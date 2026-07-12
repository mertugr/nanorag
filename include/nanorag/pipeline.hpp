#pragma once

#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/hybrid.hpp"
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
#include <unordered_map>
#include <utility>
#include <vector>

namespace nanorag {

/// Current index meta.txt schema version written by `save_meta`.
inline constexpr int kIndexMetaVersion = 2;
/// Oldest meta version still accepted by `load_meta` / `Retriever::load`.
inline constexpr int kIndexMetaVersionMin = 1;

struct IndexMeta {
    /// Schema version from meta.txt (`nanorag_index_meta_version`). 0 = missing (legacy).
    int meta_version = 0;
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

inline std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

/// Map meta metric string to tinyann::Metric. Throws if unsupported.
inline tinyann::Metric parse_index_metric(const std::string& metric) {
    const std::string m = to_lower_ascii(metric);
    if (m == "cosine") {
        return tinyann::Metric::Cosine;
    }
    if (m == "euclidean" || m == "l2") {
        return tinyann::Metric::Euclidean;
    }
    if (m == "inner_product" || m == "ip" || m == "dot") {
        return tinyann::Metric::InnerProduct;
    }
    throw std::runtime_error("unsupported index metric '" + metric +
                             "' (expected cosine|euclidean|inner_product)");
}

/// Fail-closed checks on meta fields alone (before loading binaries).
inline void validate_index_meta(const IndexMeta& m) {
    if (m.dim == 0 || m.embedder_id.empty()) {
        throw std::runtime_error("validate_index_meta: incomplete meta (need dim and embedder_id)");
    }
    if (m.meta_version > 0 && m.meta_version < kIndexMetaVersionMin) {
        throw std::runtime_error("validate_index_meta: meta version " + std::to_string(m.meta_version) +
                                 " is too old (min " + std::to_string(kIndexMetaVersionMin) + ")");
    }
    if (m.meta_version > kIndexMetaVersion) {
        throw std::runtime_error("validate_index_meta: meta version " + std::to_string(m.meta_version) +
                                 " is newer than supported " + std::to_string(kIndexMetaVersion) +
                                 " (upgrade nanorag)");
    }
    // This release only builds HNSW + cosine indexes.
    const std::string kind = to_lower_ascii(m.index_kind);
    if (kind != "hnsw") {
        throw std::runtime_error("validate_index_meta: unsupported index_kind '" + m.index_kind +
                                 "' (only hnsw is supported)");
    }
    const std::string metric = to_lower_ascii(m.metric);
    if (metric != "cosine") {
        throw std::runtime_error("validate_index_meta: unsupported metric '" + m.metric +
                                 "' (only cosine is supported)");
    }
    (void)parse_index_metric(m.metric);  // keep parser in sync with supported set
    if (m.has_no_evidence != 0 && m.has_no_evidence != 1) {
        throw std::runtime_error("validate_index_meta: has_no_evidence must be 0 or 1");
    }
}

/// Cross-check store + HNSW against meta after load. Fail closed on mismatch.
inline void validate_loaded_index(const IndexMeta& meta, const ChunkStore& store,
                                  const tinyann::HnswIndex& index) {
    validate_index_meta(meta);

    if (index.dimension() != meta.dim) {
        throw std::runtime_error("validate_loaded_index: tann dim " + std::to_string(index.dimension()) +
                                 " != meta dim " + std::to_string(meta.dim));
    }
    if (index.metric() != tinyann::Metric::Cosine) {
        throw std::runtime_error(
            "validate_loaded_index: tann metric is not cosine (index file/meta mismatch)");
    }
    if (meta.n_chunks > 0 && store.size() != meta.n_chunks) {
        throw std::runtime_error("validate_loaded_index: store size " + std::to_string(store.size()) +
                                 " != meta n_chunks " + std::to_string(meta.n_chunks));
    }
    if (index.size() != store.size()) {
        throw std::runtime_error("validate_loaded_index: index vector count " +
                                 std::to_string(index.size()) + " != store size " +
                                 std::to_string(store.size()));
    }
    const std::size_t real = count_real_chunks(store);
    if (meta.n_real_chunks > 0 && real != meta.n_real_chunks) {
        throw std::runtime_error("validate_loaded_index: real chunk count " + std::to_string(real) +
                                 " != meta n_real_chunks " + std::to_string(meta.n_real_chunks));
    }
    const bool has_ne = store.contains(kNoEvidenceId);
    if (meta.has_no_evidence == 1 && !has_ne) {
        throw std::runtime_error(
            "validate_loaded_index: meta has_no_evidence=1 but store has no id=-1 sentinel");
    }
    if (meta.has_no_evidence == 0 && has_ne) {
        throw std::runtime_error(
            "validate_loaded_index: store has NO_EVIDENCE sentinel but meta has_no_evidence=0");
    }
}

inline void save_meta(const std::string& path, const IndexMeta& m) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("save_meta: cannot open " + path);
    }
    out << "nanorag_index_meta_version=" << kIndexMetaVersion << "\n";
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
        if (key == "nanorag_index_meta_version") {
            m.meta_version = std::stoi(val);
        } else if (key == "embedder_id") {
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
    // Legacy files without version field: treat as v1 if missing n_real_chunks semantics.
    if (m.meta_version == 0) {
        m.meta_version = (m.n_real_chunks > 0 || m.has_no_evidence != 0) ? 2 : 1;
    }
    // Backward compat for v1 meta (no n_real_chunks).
    if (m.n_real_chunks == 0 && m.n_chunks > 0) {
        m.n_real_chunks = m.has_no_evidence ? (m.n_chunks > 0 ? m.n_chunks - 1 : 0) : m.n_chunks;
    }
    validate_index_meta(m);
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
    if (meta.embedder_id == kContrastiveEmbedderId ||
        meta.embedder_id == kContrastiveEmbedderIdV1) {
        auto emb =
            std::make_shared<ContrastiveEmbedder>(ContrastiveEmbedder::load(dir + "/embeddings.nctr"));
        if (emb->dim() != meta.dim) {
            throw std::runtime_error("load_embedder_for_index: contrastive dim mismatch");
        }
        if (emb->id() != meta.embedder_id) {
            throw std::runtime_error("load_embedder_for_index: contrastive id mismatch meta=" +
                                     meta.embedder_id + " file=" + emb->id());
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
    Retriever(std::shared_ptr<Embedder> embedder, ChunkStore store, tinyann::HnswIndex index,
              HybridConfig hybrid = {})
        : embedder_(std::move(embedder)),
          store_(std::move(store)),
          index_(std::move(index)),
          sparse_(store_),
          hybrid_(hybrid) {
        if (!embedder_) {
            throw std::invalid_argument("Retriever: null embedder");
        }
        if (embedder_->dim() != index_.dimension()) {
            throw std::invalid_argument("Retriever: embedder dim != index dim");
        }
    }

    static Retriever build(std::shared_ptr<Embedder> embedder, ChunkStore store,
                           tinyann::HnswParams params = {}, HybridConfig hybrid = {}) {
        if (!embedder) {
            throw std::invalid_argument("Retriever::build: null embedder");
        }
        tinyann::HnswIndex index(embedder->dim(), tinyann::Metric::Cosine, params);
        for (const auto& c : store.all()) {
            index.add(c.id, embedder->embed(c.text));
        }
        return Retriever(std::move(embedder), std::move(store), std::move(index), hybrid);
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
            } else {
                // Fail closed: never treat user-owned -1 as the refuse sentinel.
                const Chunk& ne = store.get(kNoEvidenceId);
                if (ne.source != "system" || ne.text != kNoEvidenceText) {
                    throw std::invalid_argument(
                        "build_contrastive: chunk id -1 is reserved for the system NO_EVIDENCE "
                        "sentinel; existing chunk does not match system text/source");
                }
            }
            // Out-of-scope + realistic near-misses map to the sentinel.
            // Oversample vs in-domain pairs so expanded synonym train does not drown refuse.
            const char* oods[] = {
                "What is the boiling point of alcohol?",
                "What is the boiling point of ethanol?",
                "What is the boiling point of ethanol under one atmosphere?",
                "What is the boiling temperature of rubbing alcohol?",
                "At one atmosphere when does pure ethanol become gas?",
                "What is the melting point of iron?",
                "What is the melting point of pure iron metal?",
                "What is the freezing point of mercury?",
                "What kelvin mark freezes liquid mercury?",
                "How many legs does a spider have?",
                "How many legs does a typical spider have?",
                "What is the capital of Germany?",
                "What city is the capital of Germany?",
                "Who is the president of France?",
                "What is the population of Tokyo?",
                "How tall is Mount Everest?",
                "What is the speed of sound in air?",
                "What is the speed of sound in dry air at sea level?",
                "Which planet has the most moons in our solar system?",
                "What is the chemical formula for methane?",
                "What is the square root of negative seventeen as a real number recipe?",
                "Who invented the chocolate pizza telescope in medieval France?",
                "What is the stock price of completely fictional company Zyblerqux?",
                "How many purple dragons live in my kitchen toaster?",
                "Which dog breed is best at herding sheep?",
                "What dog breed herds livestock best?",
                "Which canine variety is famous for sheep herding trials?",
                "What is the best sheep herding dog for alpine farms?",
            };
            // Aim for OOD mass ≈ in-domain mass (at least 2× the OOD template list).
            const std::size_t in_domain = pairs.size();
            const std::size_t n_ood_templates =
                sizeof(oods) / sizeof(oods[0]);
            int repeats = 2;
            if (in_domain > n_ood_templates && n_ood_templates > 0) {
                repeats = static_cast<int>((in_domain + n_ood_templates - 1) / n_ood_templates);
                if (repeats < 2) {
                    repeats = 2;
                }
                if (repeats > 6) {
                    repeats = 6;
                }
            }
            for (int r = 0; r < repeats; ++r) {
                for (const char* q : oods) {
                    train_pairs.push_back({q, kNoEvidenceId});
                }
            }
        }
        auto emb = std::make_shared<ContrastiveEmbedder>(
            ContrastiveEmbedder::train(store, train_pairs, cfg));
        return build(std::move(emb), std::move(store), params);
    }

    const HybridConfig& hybrid_config() const { return hybrid_; }
    void set_hybrid_config(HybridConfig cfg) { hybrid_ = cfg; }
    void set_retrieve_mode(RetrieveMode mode) { hybrid_.mode = mode; }
    RetrieveMode retrieve_mode() const { return hybrid_.mode; }

    /// Dense ANN only (ignores hybrid mode for this call).
    std::vector<tinyann::SearchResult> search_dense(const std::string& query,
                                                    std::size_t k) const {
        auto q = embedder_->embed(query);
        return index_.search(q, k);
    }

    /// Sparse BM25 only.
    std::vector<tinyann::SearchResult> search_sparse(const std::string& query,
                                                     std::size_t k) const {
        return sparse_.search(query, k);
    }

    /// Retrieve with current HybridConfig (dense / sparse / hybrid).
    ///
    /// Answerability consumes cosine-scale scores only. Sparse BM25 scores are
    /// marked non-cosine. Hybrid fusion ranks with BM25 boosts, then remaps every
    /// returned hit to dense cosine (dense pool values, or embedder re-score for
    /// sparse-only injects) so inject/fusion magnitudes cannot unlock the
    /// paraphrase path (issues #15, #11, #14).
    RetrieveResult retrieve(const std::string& query, std::size_t k) const {
        const std::size_t pool =
            std::max(k, std::max(hybrid_.candidate_pool, static_cast<std::size_t>(1)));
        std::vector<tinyann::SearchResult> hits;
        bool scores_are_cosine = true;
        if (hybrid_.mode == RetrieveMode::Dense) {
            hits = search_dense(query, k);
        } else if (hybrid_.mode == RetrieveMode::Sparse) {
            hits = search_sparse(query, k);
            scores_are_cosine = false;  // raw BM25 — not cosine
        } else {
            auto dense = search_dense(query, pool);
            auto sparse = search_sparse(query, pool);
            const float sparse_max = sparse_.max_score(query);
            hits = fuse_dense_sparse(dense, sparse, hybrid_, k, sparse_max, &sparse_, &query);
            // Report dense cosine as the score used by the answerability gate.
            // Hybrid fusion is only for *ranking*; BM25-boosted / inject fusion
            // scores must not reach min_score_without_query_support.
            std::unordered_map<std::int64_t, float> dense_score;
            for (const auto& h : dense) {
                dense_score[h.id] = h.score;
            }
            // Re-score sparse-only injects (and any non-dense fusion hits) with the
            // same embedder used for ANN so the gate sees true cosine.
            std::vector<float> qvec;
            bool have_q = false;
            for (auto& h : hits) {
                auto it = dense_score.find(h.id);
                if (it != dense_score.end()) {
                    h.score = it->second;
                    continue;
                }
                if (!have_q) {
                    qvec = embedder_->embed(query);
                    have_q = true;
                }
                if (!store_.contains(h.id)) {
                    h.score = 0.f;
                    continue;
                }
                const auto dvec = embedder_->embed(store_.get(h.id).text);
                h.score = static_cast<float>(cosine(qvec, dvec));
            }
            // Keep ranking consistent with the scores the gate will see.
            std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.id < b.id;
            });
        }
        RetrieveResult r;
        r.chunks = hits_to_chunks(hits, store_);
        if (!scores_are_cosine) {
            for (auto& c : r.chunks) {
                c.score_is_cosine = false;
            }
        }
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

    static Retriever load(const std::string& dir, std::shared_ptr<Embedder> embedder,
                          HybridConfig hybrid = {}) {
        if (!embedder) {
            throw std::invalid_argument("Retriever::load: null embedder");
        }
        auto meta = load_meta(dir + "/meta.txt");  // validates kind/metric/version
        if (meta.embedder_id != embedder->id()) {
            throw std::runtime_error("Retriever::load: embedder_id mismatch: index has '" +
                                     meta.embedder_id + "', got '" + embedder->id() + "'");
        }
        if (meta.dim != embedder->dim()) {
            throw std::runtime_error("Retriever::load: dim mismatch: meta=" +
                                     std::to_string(meta.dim) + " embedder=" +
                                     std::to_string(embedder->dim()));
        }
        auto store = ChunkStore::load(dir + "/chunks.tsv");
        auto index = tinyann::HnswIndex::load(dir + "/vectors.hnsw.tann");
        validate_loaded_index(meta, store, index);
        if (index.dimension() != embedder->dim()) {
            throw std::runtime_error("Retriever::load: tann dim mismatch");
        }
        return Retriever(std::move(embedder), std::move(store), std::move(index), hybrid);
    }

    static Retriever open(const std::string& dir, HybridConfig hybrid = {}) {
        auto meta = load_meta(dir + "/meta.txt");
        auto emb = load_embedder_for_index(dir, meta);
        return load(dir, std::move(emb), hybrid);
    }

    const ChunkStore& store() const { return store_; }
    const Embedder& embedder() const { return *embedder_; }
    const SparseBm25Index& sparse_index() const { return sparse_; }
    std::size_t size() const { return store_.size(); }
    std::size_t real_size() const { return count_real_chunks(store_); }

private:
    std::shared_ptr<Embedder> embedder_;
    ChunkStore store_;
    tinyann::HnswIndex index_;
    SparseBm25Index sparse_;
    HybridConfig hybrid_;
};

}  // namespace nanorag
