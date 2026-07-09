#pragma once

#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/word2vec.hpp"

#include "tinyann/tinyann.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {

struct IndexMeta {
    std::string embedder_id;
    std::size_t dim = 0;
    std::string metric = "cosine";
    std::string index_kind = "hnsw";  // exact | hnsw
    std::size_t n_chunks = 0;
};

inline void save_meta(const std::string& path, const IndexMeta& m) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("save_meta: cannot open " + path);
    }
    out << "nanorag_index_meta_version=1\n";
    out << "embedder_id=" << m.embedder_id << "\n";
    out << "dim=" << m.dim << "\n";
    out << "metric=" << m.metric << "\n";
    out << "index_kind=" << m.index_kind << "\n";
    out << "n_chunks=" << m.n_chunks << "\n";
}

inline IndexMeta load_meta(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_meta: cannot open " + path);
    }
    IndexMeta m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if (key == "embedder_id") {
            m.embedder_id = val;
        } else if (key == "dim") {
            m.dim = static_cast<std::size_t>(std::stoull(val));
        } else if (key == "metric") {
            m.metric = val;
        } else if (key == "index_kind") {
            m.index_kind = val;
        } else if (key == "n_chunks") {
            m.n_chunks = static_cast<std::size_t>(std::stoull(val));
        }
    }
    if (m.dim == 0 || m.embedder_id.empty()) {
        throw std::runtime_error("load_meta: incomplete meta in " + path);
    }
    return m;
}

/// Load the embedder that matches index meta (hashing-v1 or word2vec-v1).
inline std::shared_ptr<Embedder> load_embedder_for_index(const std::string& dir,
                                                         const IndexMeta& meta) {
    if (meta.embedder_id == kHashingEmbedderId) {
        return std::make_shared<HashingEmbedder>(meta.dim);
    }
    if (meta.embedder_id == kWord2VecEmbedderId) {
        auto emb = std::make_shared<Word2VecEmbedder>(Word2VecEmbedder::load(dir + "/embeddings.nw2v"));
        if (emb->dim() != meta.dim) {
            throw std::runtime_error("load_embedder_for_index: word2vec dim mismatch");
        }
        return emb;
    }
    throw std::runtime_error("load_embedder_for_index: unknown embedder_id '" + meta.embedder_id +
                             "'");
}

struct RetrieveResult {
    std::vector<RetrievedChunk> chunks;
    std::string prompt;
};

/// Own-stack retriever: Embedder + tinyann + ChunkStore.
class Retriever {
public:
    Retriever(std::shared_ptr<Embedder> embedder, ChunkStore store, tinyann::HnswIndex index)
        : embedder_(std::move(embedder)), store_(std::move(store)), index_(std::move(index)) {
        if (!embedder_) {
            throw std::invalid_argument("Retriever: null embedder");
        }
        if (embedder_->dim() != index_.dimension()) {
            throw std::invalid_argument("Retriever: embedder dim != index dim");
        }
    }

    static Retriever build(std::shared_ptr<Embedder> embedder, ChunkStore store,
                           tinyann::HnswParams params = {}) {
        if (!embedder) {
            throw std::invalid_argument("Retriever::build: null embedder");
        }
        tinyann::HnswIndex index(embedder->dim(), tinyann::Metric::Cosine, params);
        for (const auto& c : store.all()) {
            index.add(c.id, embedder->embed(c.text));
        }
        return Retriever(std::move(embedder), std::move(store), std::move(index));
    }

    /// Train Word2Vec on chunk texts, then build HNSW index.
    static Retriever build_word2vec(ChunkStore store, Word2VecTrainConfig cfg = {},
                                    tinyann::HnswParams params = {}) {
        std::vector<std::string> docs;
        docs.reserve(store.size());
        for (const auto& c : store.all()) {
            docs.push_back(c.text);
        }
        auto emb = std::make_shared<Word2VecEmbedder>(Word2VecEmbedder::train(docs, cfg));
        return build(std::move(emb), std::move(store), params);
    }

    RetrieveResult retrieve(const std::string& query, std::size_t k) const {
        auto q = embedder_->embed(query);
        auto hits = index_.search(q, k);
        RetrieveResult r;
        r.chunks = hits_to_chunks(hits, store_);
        r.prompt = build_rag_prompt(query, r.chunks);
        return r;
    }

    void save(const std::string& dir) const {
        store_.save(dir + "/chunks.tsv");
        index_.save(dir + "/vectors.hnsw.tann");
        if (auto* w2v = dynamic_cast<const Word2VecEmbedder*>(embedder_.get())) {
            w2v->save(dir + "/embeddings.nw2v");
        }
        IndexMeta meta;
        meta.embedder_id = embedder_->id();
        meta.dim = embedder_->dim();
        meta.metric = "cosine";
        meta.index_kind = "hnsw";
        meta.n_chunks = store_.size();
        save_meta(dir + "/meta.txt", meta);
    }

    static Retriever load(const std::string& dir, std::shared_ptr<Embedder> embedder) {
        if (!embedder) {
            throw std::invalid_argument("Retriever::load: null embedder");
        }
        auto meta = load_meta(dir + "/meta.txt");
        if (meta.embedder_id != embedder->id()) {
            throw std::runtime_error("Retriever::load: embedder_id mismatch: index has '" +
                                     meta.embedder_id + "', got '" + embedder->id() + "'");
        }
        if (meta.dim != embedder->dim()) {
            throw std::runtime_error("Retriever::load: dim mismatch");
        }
        auto store = ChunkStore::load(dir + "/chunks.tsv");
        auto index = tinyann::HnswIndex::load(dir + "/vectors.hnsw.tann");
        if (index.dimension() != embedder->dim()) {
            throw std::runtime_error("Retriever::load: tann dim mismatch");
        }
        return Retriever(std::move(embedder), std::move(store), std::move(index));
    }

    /// Load index + matching embedder from meta (preferred).
    static Retriever open(const std::string& dir) {
        auto meta = load_meta(dir + "/meta.txt");
        auto emb = load_embedder_for_index(dir, meta);
        return load(dir, std::move(emb));
    }

    const ChunkStore& store() const { return store_; }
    const Embedder& embedder() const { return *embedder_; }
    std::size_t size() const { return store_.size(); }

private:
    std::shared_ptr<Embedder> embedder_;
    ChunkStore store_;
    tinyann::HnswIndex index_;
};

}  // namespace nanorag
