#pragma once

// Hybrid sparse (BM25) + dense (ANN) retrieval.
//
// Design (dense-primary, adaptive):
//   1) Always take a dense ANN candidate pool.
//   2) If max BM25 signal is weak → return dense ranking unchanged
//      (protects hard/zero-overlap paraphrases from sparse rank noise).
//   3) Otherwise re-rank the dense pool with BM25 boost, and inject only
//      high-confidence sparse-only hits (strong BM25).
//
// BM25 never indexes the NO_EVIDENCE sentinel (id=-1) so lexical ranking
// cannot promote the system refuse doc over real passages.

#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/grounding.hpp"  // kNoEvidenceId
#include "nanorag/prompt.hpp"

#include "tinyann/tinyann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanorag {

enum class RetrieveMode {
    Dense,   // ANN only
    Sparse,  // BM25 only
    Hybrid,  // dense-primary + BM25 boost (default)
};

enum class HybridFusion {
    DensePrimary,  // re-rank dense pool + optional strong sparse inject (default)
    RRF,           // classic reciprocal rank fusion (can hurt hard; kept for ablations)
    Weighted,      // min-max-normalized dense + sparse over union of lists
};

struct HybridConfig {
    RetrieveMode mode = RetrieveMode::Hybrid;
    HybridFusion fusion = HybridFusion::DensePrimary;
    float dense_weight = 0.55f;
    float sparse_weight = 0.45f;
    /// Scale for BM25 add-on in DensePrimary (score = dense + sparse_boost * bm25).
    float sparse_boost = 0.08f;
    /// RRF constant when fusion == RRF.
    int rrf_k = 60;
    /// Candidates from dense (and sparse inject list).
    std::size_t candidate_pool = 32;
    /// Below this max BM25, hybrid falls back to pure dense.
    float min_sparse_signal = 1.0f;
    /// BM25 must reach this to re-rank a dense hit or inject a sparse-only doc.
    /// Tuned so weak partial matches do not defeat NO_EVIDENCE on OOD queries.
    float strong_sparse_inject = 2.5f;
};

inline const char* retrieve_mode_name(RetrieveMode m) {
    switch (m) {
        case RetrieveMode::Dense:
            return "dense";
        case RetrieveMode::Sparse:
            return "sparse";
        case RetrieveMode::Hybrid:
            return "hybrid";
    }
    return "unknown";
}

inline RetrieveMode parse_retrieve_mode(const std::string& s) {
    if (s == "dense") {
        return RetrieveMode::Dense;
    }
    if (s == "sparse") {
        return RetrieveMode::Sparse;
    }
    if (s == "hybrid") {
        return RetrieveMode::Hybrid;
    }
    throw std::invalid_argument("unknown retrieve mode '" + s + "' (dense|sparse|hybrid)");
}

/// In-memory BM25 over tokenized chunks (Okapi-style). Skips NO_EVIDENCE sentinel.
class SparseBm25Index {
public:
    SparseBm25Index() = default;

    explicit SparseBm25Index(const ChunkStore& store, float k1 = 1.2f, float b = 0.75f)
        : k1_(k1), b_(b) {
        rebuild(store);
    }

    void rebuild(const ChunkStore& store) {
        docs_.clear();
        df_.clear();
        id_to_row_.clear();
        avgdl_ = 0.0;
        for (const auto& c : store.all()) {
            if (c.id == kNoEvidenceId) {
                continue;  // never index refuse sentinel
            }
            Doc d;
            d.id = c.id;
            auto toks = detail::simple_tokenize(c.text);
            d.len = static_cast<int>(toks.size());
            for (const auto& t : toks) {
                d.tf[t] += 1;
            }
            avgdl_ += static_cast<double>(d.len);
            docs_.push_back(std::move(d));
        }
        const std::size_t N = docs_.size();
        if (N == 0) {
            avgdl_ = 0;
            n_docs_ = 0;
            return;
        }
        avgdl_ /= static_cast<double>(N);
        for (std::size_t i = 0; i < docs_.size(); ++i) {
            id_to_row_[docs_[i].id] = i;
            for (const auto& kv : docs_[i].tf) {
                df_[kv.first] += 1;
            }
        }
        n_docs_ = N;
    }

    std::size_t size() const { return docs_.size(); }

    float score_id(const std::string& query, std::int64_t id) const {
        auto it = id_to_row_.find(id);
        if (it == id_to_row_.end()) {
            return 0.f;
        }
        return score_doc(query_tf(query), docs_[it->second]);
    }

    std::vector<tinyann::SearchResult> search(const std::string& query, std::size_t k) const {
        std::vector<tinyann::SearchResult> all;
        all.reserve(docs_.size());
        const auto qtf = query_tf(query);
        if (qtf.empty() || docs_.empty()) {
            return all;
        }
        for (const auto& d : docs_) {
            const float s = score_doc(qtf, d);
            if (s > 0.f) {
                all.push_back({d.id, s});
            }
        }
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id < b.id;
        });
        if (all.size() > k) {
            all.resize(k);
        }
        return all;
    }

    float max_score(const std::string& query) const {
        const auto qtf = query_tf(query);
        float mx = 0.f;
        for (const auto& d : docs_) {
            mx = std::max(mx, score_doc(qtf, d));
        }
        return mx;
    }

private:
    struct Doc {
        std::int64_t id = 0;
        int len = 0;
        std::unordered_map<std::string, int> tf;
    };

    static std::unordered_map<std::string, int> query_tf(const std::string& query) {
        std::unordered_map<std::string, int> qtf;
        for (const auto& t : detail::simple_tokenize(query)) {
            qtf[t] += 1;
        }
        return qtf;
    }

    float idf(const std::string& term) const {
        const auto it = df_.find(term);
        const double df = it == df_.end() ? 0.0 : static_cast<double>(it->second);
        const double N = static_cast<double>(std::max<std::size_t>(n_docs_, 1));
        return static_cast<float>(std::log(1.0 + (N - df + 0.5) / (df + 0.5)));
    }

    float score_doc(const std::unordered_map<std::string, int>& qtf, const Doc& d) const {
        if (d.len == 0 || avgdl_ <= 0.0) {
            return 0.f;
        }
        double score = 0.0;
        const double dl = static_cast<double>(d.len);
        for (const auto& qv : qtf) {
            auto tit = d.tf.find(qv.first);
            if (tit == d.tf.end()) {
                continue;
            }
            const double tf = static_cast<double>(tit->second);
            const double idf_t = static_cast<double>(idf(qv.first));
            const double denom = tf + k1_ * (1.0 - b_ + b_ * (dl / avgdl_));
            score += idf_t * (tf * (k1_ + 1.0) / denom) * static_cast<double>(qv.second);
        }
        return static_cast<float>(score);
    }

    float k1_ = 1.2f;
    float b_ = 0.75f;
    double avgdl_ = 0.0;
    std::size_t n_docs_ = 0;
    std::vector<Doc> docs_;
    std::unordered_map<std::string, int> df_;
    std::unordered_map<std::int64_t, std::size_t> id_to_row_;
};

namespace hybrid_detail {

inline std::vector<tinyann::SearchResult> rrf_fuse(
    const std::vector<tinyann::SearchResult>& a, const std::vector<tinyann::SearchResult>& b,
    int rrf_k, std::size_t top_k) {
    std::unordered_map<std::int64_t, double> scores;
    auto accumulate = [&](const std::vector<tinyann::SearchResult>& list) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            scores[list[i].id] += 1.0 / (static_cast<double>(rrf_k) + static_cast<double>(i + 1));
        }
    };
    accumulate(a);
    accumulate(b);
    std::vector<tinyann::SearchResult> out;
    out.reserve(scores.size());
    for (const auto& kv : scores) {
        out.push_back({kv.first, static_cast<float>(kv.second)});
    }
    std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) {
        if (x.score != y.score) {
            return x.score > y.score;
        }
        return x.id < y.id;
    });
    if (out.size() > top_k) {
        out.resize(top_k);
    }
    return out;
}

inline void min_max_norm(std::vector<float>& v) {
    if (v.empty()) {
        return;
    }
    float lo = v[0], hi = v[0];
    for (float x : v) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    const float span = hi - lo;
    if (span <= 1e-12f) {
        for (float& x : v) {
            x = 0.f;
        }
        return;
    }
    for (float& x : v) {
        x = (x - lo) / span;
    }
}

inline std::vector<tinyann::SearchResult> weighted_fuse(
    const std::vector<tinyann::SearchResult>& dense, const std::vector<tinyann::SearchResult>& sparse,
    float w_d, float w_s, std::size_t top_k) {
    std::unordered_map<std::int64_t, float> dmap, smap;
    for (const auto& h : dense) {
        dmap[h.id] = h.score;
    }
    for (const auto& h : sparse) {
        smap[h.id] = h.score;
    }
    std::vector<std::int64_t> ids;
    for (const auto& kv : dmap) {
        ids.push_back(kv.first);
    }
    for (const auto& kv : smap) {
        if (!dmap.count(kv.first)) {
            ids.push_back(kv.first);
        }
    }
    std::vector<float> dvals(ids.size(), 0.f), svals(ids.size(), 0.f);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto dit = dmap.find(ids[i]);
        auto sit = smap.find(ids[i]);
        if (dit != dmap.end()) {
            dvals[i] = dit->second;
        }
        if (sit != smap.end()) {
            svals[i] = sit->second;
        }
    }
    min_max_norm(dvals);
    min_max_norm(svals);
    std::vector<tinyann::SearchResult> out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        out.push_back({ids[i], w_d * dvals[i] + w_s * svals[i]});
    }
    std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) {
        if (x.score != y.score) {
            return x.score > y.score;
        }
        return x.id < y.id;
    });
    if (out.size() > top_k) {
        out.resize(top_k);
    }
    return out;
}

/// Dense-primary: re-rank dense hits with BM25 boost only on *strong* lexical matches.
/// Weak BM25 noise must not demote NO_EVIDENCE or flip OOD refuse behavior.
inline std::vector<tinyann::SearchResult> dense_primary_fuse(
    const std::vector<tinyann::SearchResult>& dense, const std::vector<tinyann::SearchResult>& sparse,
    const SparseBm25Index& bm25, const std::string& query, const HybridConfig& cfg, std::size_t k) {
    std::unordered_map<std::int64_t, float> hybrid;
    std::unordered_set<std::int64_t> in_dense;
    for (const auto& h : dense) {
        in_dense.insert(h.id);
        // Never BM25-boost the refuse sentinel (and it is not in the sparse index).
        if (h.id == kNoEvidenceId) {
            hybrid[h.id] = h.score;
            continue;
        }
        const float bm = bm25.score_id(query, h.id);
        // Only boost when lexical evidence is strong — weak partial matches are noise.
        if (bm >= cfg.strong_sparse_inject) {
            hybrid[h.id] = h.score + cfg.sparse_boost * bm;
        } else {
            hybrid[h.id] = h.score;
        }
    }
    for (const auto& h : sparse) {
        if (in_dense.count(h.id) || h.id == kNoEvidenceId) {
            continue;
        }
        // Inject sparse-only hits that dense missed but BM25 is confident about.
        // Ranking-only synthetic score: pipeline::retrieve re-scores injects with
        // embedder cosine before answerability (issues #11, #14). Do not treat this
        // value as cosine in the gate.
        if (h.score >= cfg.strong_sparse_inject) {
            // Place just below a typical dense top hit so inject helps R@k without
            // overtaking a high dense cosine unless dense was weak.
            const float base = dense.empty() ? 0.f : dense.front().score * 0.85f;
            hybrid[h.id] = std::max(base, cfg.sparse_boost * h.score);
        }
    }
    std::vector<tinyann::SearchResult> out;
    out.reserve(hybrid.size());
    for (const auto& kv : hybrid) {
        out.push_back({kv.first, kv.second});
    }
    std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) {
        if (x.score != y.score) {
            return x.score > y.score;
        }
        return x.id < y.id;
    });
    if (out.size() > k) {
        out.resize(k);
    }
    return out;
}

}  // namespace hybrid_detail

/// Fuse dense ANN hits with BM25. `bm25` is used for DensePrimary scoring.
inline std::vector<tinyann::SearchResult> fuse_dense_sparse(
    const std::vector<tinyann::SearchResult>& dense, const std::vector<tinyann::SearchResult>& sparse,
    const HybridConfig& cfg, std::size_t k, float sparse_max_signal,
    const SparseBm25Index* bm25 = nullptr, const std::string* query = nullptr) {
    if (cfg.mode == RetrieveMode::Dense) {
        auto out = dense;
        if (out.size() > k) {
            out.resize(k);
        }
        return out;
    }
    if (cfg.mode == RetrieveMode::Sparse) {
        auto out = sparse;
        if (out.size() > k) {
            out.resize(k);
        }
        return out;
    }
    // Hybrid: adaptive dense-only unless BM25 has a *strong* hit.
    // Using strong_sparse_inject (not the weaker min_sparse_signal) protects OOD
    // refuse: weak lexical noise must not re-rank the dense list.
    if (sparse_max_signal < cfg.strong_sparse_inject || sparse.empty()) {
        auto out = dense;
        if (out.size() > k) {
            out.resize(k);
        }
        return out;
    }
    if (cfg.fusion == HybridFusion::DensePrimary && bm25 != nullptr && query != nullptr) {
        return hybrid_detail::dense_primary_fuse(dense, sparse, *bm25, *query, cfg, k);
    }
    if (cfg.fusion == HybridFusion::RRF) {
        return hybrid_detail::rrf_fuse(dense, sparse, cfg.rrf_k, k);
    }
    return hybrid_detail::weighted_fuse(dense, sparse, cfg.dense_weight, cfg.sparse_weight, k);
}

}  // namespace nanorag
