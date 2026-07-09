#pragma once

#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/word2vec.hpp"

#include "tinyann/tinyann.hpp"

#include <algorithm>
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
    std::string index_kind = "hnsw";
    /// Total rows in the chunk store (may include NO_EVIDENCE sentinel).
    std::size_t n_chunks = 0;
    /// Chunks excluding system NO_EVIDENCE (id == kNoEvidenceId).
    std::size_t n_real_chunks = 0;
    /// 1 if NO_EVIDENCE was injected at train time.
    int has_no_evidence = 0;
};

inline std::size_t count_real_chunks(const ChunkStore& store) {
    std::size_t n = 0;
    for (auto id : store.ids()) {
        if (id != kNoEvidenceId) {
            ++n;
        }
    }
    return n;
}

inline void save_meta(const std::string& path, const IndexMeta& m) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("save_meta: cannot open " + path);
    }
    out << "nanorag_index_meta_version=2\n";
    out << "embedder_id=" << m.embedder_id << "\n";
    out << "dim=" << m.dim << "\n";
    out << "metric=" << m.metric << "\n";
    out << "index_kind=" << m.index_kind << "\n";
    out << "n_chunks=" << m.n_chunks << "\n";
    out << "n_real_chunks=" << m.n_real_chunks << "\n";
    out << "has_no_evidence=" << m.has_no_evidence << "\n";
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
        } else if (key == "n_real_chunks") {
            m.n_real_chunks = static_cast<std::size_t>(std::stoull(val));
        } else if (key == "has_no_evidence") {
            m.has_no_evidence = std::stoi(val);
        }
    }
    if (m.dim == 0 || m.embedder_id.empty()) {
        throw std::runtime_error("load_meta: incomplete meta in " + path);
    }
    // Backward compat for v1 meta (no n_real_chunks).
    if (m.n_real_chunks == 0 && m.n_chunks > 0) {
        m.n_real_chunks = m.has_no_evidence ? (m.n_chunks > 0 ? m.n_chunks - 1 : 0) : m.n_chunks;
    }
    return m;
}

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
    if (meta.embedder_id == kContrastiveEmbedderId) {
        auto emb =
            std::make_shared<ContrastiveEmbedder>(ContrastiveEmbedder::load(dir + "/embeddings.nctr"));
        if (emb->dim() != meta.dim) {
            throw std::runtime_error("load_embedder_for_index: contrastive dim mismatch");
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

    static Retriever build_word2vec(ChunkStore store, Word2VecTrainConfig cfg = {},
                                    tinyann::HnswParams params = {}) {
        std::vector<std::string> docs;
        for (const auto& c : store.all()) {
            docs.push_back(c.text);
        }
        auto emb = std::make_shared<Word2VecEmbedder>(Word2VecEmbedder::train(docs, cfg));
        return build(std::move(emb), std::move(store), params);
    }

    static Retriever build_contrastive(ChunkStore store, const std::vector<TrainPair>& pairs,
                                       ContrastiveTrainConfig cfg = {},
                                       tinyann::HnswParams params = {},
                                       bool inject_no_evidence = true) {
        auto train_pairs = pairs;
        if (inject_no_evidence) {
            if (!store.contains(kNoEvidenceId)) {
                store.add({kNoEvidenceId, "system", kNoEvidenceText});
            }
            // Out-of-scope + realistic near-misses map to the sentinel.
            const char* oods[] = {
                "What is the boiling point of alcohol?",
                "What is the boiling point of ethanol?",
                "What is the melting point of iron?",
                "What is the freezing point of mercury?",
                "How many legs does a spider have?",
                "What is the capital of Germany?",
                "Who is the president of France?",
                "What is the population of Tokyo?",
                "How tall is Mount Everest?",
                "What is the speed of sound in air?",
                "Which planet has the most moons in our solar system?",
                "What is the chemical formula for methane?",
                "Who invented the chocolate pizza telescope in medieval France?",
                "What is the stock price of completely fictional company Zyblerqux?",
                "How many purple dragons live in my kitchen toaster?",
            };
            for (const char* q : oods) {
                train_pairs.push_back({q, kNoEvidenceId});
            }
        }
        auto emb = std::make_shared<ContrastiveEmbedder>(
            ContrastiveEmbedder::train(store, train_pairs, cfg));
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

    /// Top-1 among real corpus chunks only (skips NO_EVIDENCE). Empty if none.
    std::int64_t top_real_id(const std::string& query, std::size_t k = 8) const {
        auto r = retrieve(query, k);
        for (const auto& h : r.chunks) {
            if (h.id != kNoEvidenceId) {
                return h.id;
            }
        }
        return kNoEvidenceId;  // none
    }

    /// Retrieve then answerability gate + extractive/refuse.
    GroundedAnswer ask_grounded(const std::string& query, std::size_t k,
                                const GroundingConfig& gcfg = {}) const {
        if (is_blank_query(query)) {
            return answer_extractive_for_query(query, {}, gcfg);
        }
        const std::size_t fetch = std::max(k, gcfg.max_context_chunks * 3);
        auto r = retrieve(query, fetch);
        return answer_extractive_for_query(query, r.chunks, *embedder_, gcfg);
    }

    void save(const std::string& dir) const {
        store_.save(dir + "/chunks.tsv");
        index_.save(dir + "/vectors.hnsw.tann");
        if (auto* w2v = dynamic_cast<const Word2VecEmbedder*>(embedder_.get())) {
            w2v->save(dir + "/embeddings.nw2v");
        }
        if (auto* ctr = dynamic_cast<const ContrastiveEmbedder*>(embedder_.get())) {
            ctr->save(dir + "/embeddings.nctr");
        }
        IndexMeta meta;
        meta.embedder_id = embedder_->id();
        meta.dim = embedder_->dim();
        meta.metric = "cosine";
        meta.index_kind = "hnsw";
        meta.n_chunks = store_.size();
        meta.n_real_chunks = count_real_chunks(store_);
        meta.has_no_evidence = store_.contains(kNoEvidenceId) ? 1 : 0;
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

    static Retriever open(const std::string& dir) {
        auto meta = load_meta(dir + "/meta.txt");
        auto emb = load_embedder_for_index(dir, meta);
        return load(dir, std::move(emb));
    }

    const ChunkStore& store() const { return store_; }
    const Embedder& embedder() const { return *embedder_; }
    std::size_t size() const { return store_.size(); }
    std::size_t real_size() const { return count_real_chunks(store_); }

private:
    std::shared_ptr<Embedder> embedder_;
    ChunkStore store_;
    tinyann::HnswIndex index_;
};

}  // namespace nanorag
