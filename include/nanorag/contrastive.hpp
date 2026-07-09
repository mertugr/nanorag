#pragma once

// Supervised contrastive sentence embedder (pure C++17).
//
// Why not Word2Vec alone?
//   Skip-gram word averages only capture co-occurrence. They fail when a query
//   paraphrases a document with little lexical overlap. This model trains
//   token embeddings so that embed(query) ≈ embed(positive_doc) under InfoNCE,
//   using explicit (query, doc) pairs — the honest way to get paraphrase
//   retrieval without external embedding APIs.
//
// Format: .nctr (magic NCTR)

#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/text_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nanorag {

inline constexpr const char* kContrastiveEmbedderId = "contrastive-v1";
inline constexpr char kContrastiveMagic[4] = {'N', 'C', 'T', 'R'};
inline constexpr std::uint32_t kContrastiveFormatVersion = 1;

struct TrainPair {
    std::string query;
    std::int64_t pos_id = 0;
};

struct ContrastiveTrainConfig {
    std::size_t dim = 64;
    int epochs = 120;
    float lr = 0.08f;
    float temperature = 0.07f;
    std::uint32_t seed = 0xA11CEu;
    int min_count = 1;
    /// In-batch negatives: use all other docs in the store (full softmax over docs).
    bool full_doc_softmax = true;
    int max_negatives = 32;  // used when full_doc_softmax is false
};

inline std::vector<TrainPair> load_train_pairs(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_train_pairs: cannot open " + path);
    }
    std::vector<TrainPair> pairs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("load_train_pairs: expected query\\tpos_id: " + line);
        }
        TrainPair p;
        p.query = line.substr(0, tab);
        p.pos_id = std::stoll(line.substr(tab + 1));
        if (p.query.empty()) {
            throw std::runtime_error("load_train_pairs: empty query");
        }
        pairs.push_back(std::move(p));
    }
    if (pairs.empty()) {
        throw std::runtime_error("load_train_pairs: no pairs in " + path);
    }
    return pairs;
}

class ContrastiveModel {
public:
    std::size_t dim() const { return dim_; }
    std::size_t vocab_size() const { return vocab_.size(); }

    std::vector<float> embed_text(const std::string& text) const {
        return embed_tokens(detail::simple_tokenize(text));
    }

    std::vector<float> embed_tokens(const std::vector<std::string>& tokens) const {
        std::vector<float> v(dim_, 0.f);
        int n = 0;
        for (const auto& t : tokens) {
            auto it = token_to_id_.find(t);
            if (it == token_to_id_.end()) {
                continue;
            }
            const float* row = emb_.data() + static_cast<std::size_t>(it->second) * dim_;
            for (std::size_t d = 0; d < dim_; ++d) {
                v[d] += row[d];
            }
            ++n;
        }
        if (n > 0) {
            const float inv = 1.f / static_cast<float>(n);
            for (float& x : v) {
                x *= inv;
            }
            detail::l2_normalize(v);
        }
        return v;
    }

    /// Train token embeddings so paraphrastic queries rank their positive docs.
    static ContrastiveModel train(const ChunkStore& store, const std::vector<TrainPair>& pairs,
                                  ContrastiveTrainConfig cfg = {}) {
        if (cfg.dim == 0) {
            throw std::invalid_argument("ContrastiveModel: dim must be > 0");
        }
        if (store.size() == 0) {
            throw std::invalid_argument("ContrastiveModel: empty store");
        }
        if (pairs.empty()) {
            throw std::invalid_argument("ContrastiveModel: empty train pairs");
        }

        std::unordered_map<std::string, int> counts;
        auto add_text = [&](const std::string& s) {
            for (const auto& t : detail::simple_tokenize(s)) {
                counts[t] += 1;
            }
        };
        for (const auto& c : store.all()) {
            add_text(c.text);
        }
        for (const auto& p : pairs) {
            add_text(p.query);
            if (!store.contains(p.pos_id)) {
                throw std::invalid_argument("ContrastiveModel: pair pos_id missing from store: " +
                                            std::to_string(p.pos_id));
            }
        }

        ContrastiveModel m;
        m.dim_ = cfg.dim;
        for (const auto& kv : counts) {
            if (kv.second >= cfg.min_count) {
                m.token_to_id_[kv.first] = static_cast<int>(m.vocab_.size());
                m.vocab_.push_back(kv.first);
            }
        }
        if (m.vocab_.empty()) {
            throw std::invalid_argument("ContrastiveModel: empty vocab");
        }

        const int V = static_cast<int>(m.vocab_.size());
        m.emb_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
        std::mt19937 rng(cfg.seed);
        std::uniform_real_distribution<float> ur(-0.5f, 0.5f);
        const float scale = 1.f / static_cast<float>(m.dim_);
        for (float& x : m.emb_) {
            x = ur(rng) * scale;
        }

        // Cache doc embeddings recomputed each epoch (small corpora).
        const auto chunks = store.all();
        std::unordered_map<std::int64_t, std::size_t> id_to_row;
        std::vector<std::int64_t> doc_ids;
        doc_ids.reserve(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            id_to_row[chunks[i].id] = i;
            doc_ids.push_back(chunks[i].id);
        }

        auto embed_ids = [&](const std::vector<int>& ids) {
            std::vector<float> v(m.dim_, 0.f);
            if (ids.empty()) {
                return v;
            }
            for (int id : ids) {
                const float* row = m.emb_.data() + static_cast<std::size_t>(id) * m.dim_;
                for (std::size_t d = 0; d < m.dim_; ++d) {
                    v[d] += row[d];
                }
            }
            const float inv = 1.f / static_cast<float>(ids.size());
            for (float& x : v) {
                x *= inv;
            }
            detail::l2_normalize(v);
            return v;
        };

        auto tokenize_ids = [&](const std::string& text) {
            std::vector<int> ids;
            for (const auto& t : detail::simple_tokenize(text)) {
                auto it = m.token_to_id_.find(t);
                if (it != m.token_to_id_.end()) {
                    ids.push_back(it->second);
                }
            }
            return ids;
        };

        // Pre-tokenize pairs and docs
        struct PairTok {
            std::vector<int> q;
            std::vector<int> pos;
            std::int64_t pos_id;
        };
        std::vector<PairTok> pt;
        pt.reserve(pairs.size());
        for (const auto& p : pairs) {
            PairTok x;
            x.q = tokenize_ids(p.query);
            x.pos = tokenize_ids(store.get(p.pos_id).text);
            x.pos_id = p.pos_id;
            if (x.q.empty() || x.pos.empty()) {
                throw std::invalid_argument(
                    "ContrastiveModel: pair has empty tokens after vocab filter");
            }
            pt.push_back(std::move(x));
        }

        std::vector<std::vector<int>> doc_tok(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            doc_tok[i] = tokenize_ids(chunks[i].text);
        }

        // InfoNCE: for each query, scores against all docs (or sampled negs + pos).
        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            const float lr =
                cfg.lr * (1.f - 0.9f * static_cast<float>(epoch) / static_cast<float>(cfg.epochs));
            std::shuffle(pt.begin(), pt.end(), rng);

            for (const auto& pair : pt) {
                auto qv = embed_ids(pair.q);
                // Collect candidate docs: all docs or pos + random negs
                std::vector<std::size_t> cand_rows;
                if (cfg.full_doc_softmax) {
                    for (std::size_t i = 0; i < chunks.size(); ++i) {
                        cand_rows.push_back(i);
                    }
                } else {
                    cand_rows.push_back(id_to_row.at(pair.pos_id));
                    std::uniform_int_distribution<std::size_t> ud(0, chunks.size() - 1);
                    while (static_cast<int>(cand_rows.size()) < cfg.max_negatives + 1) {
                        std::size_t j = ud(rng);
                        if (chunks[j].id == pair.pos_id) {
                            continue;
                        }
                        if (std::find(cand_rows.begin(), cand_rows.end(), j) != cand_rows.end()) {
                            continue;
                        }
                        cand_rows.push_back(j);
                    }
                }

                std::vector<std::vector<float>> dvs;
                dvs.reserve(cand_rows.size());
                std::vector<float> logits;
                logits.reserve(cand_rows.size());
                int pos_index = -1;
                for (std::size_t ci = 0; ci < cand_rows.size(); ++ci) {
                    const std::size_t row = cand_rows[ci];
                    if (chunks[row].id == pair.pos_id) {
                        pos_index = static_cast<int>(ci);
                    }
                    auto dv = embed_ids(doc_tok[row]);
                    float dot = 0.f;
                    for (std::size_t d = 0; d < m.dim_; ++d) {
                        dot += qv[d] * dv[d];
                    }
                    logits.push_back(dot / cfg.temperature);
                    dvs.push_back(std::move(dv));
                }
                if (pos_index < 0) {
                    throw std::logic_error("ContrastiveModel: positive missing from candidates");
                }

                // softmax
                float mx = *std::max_element(logits.begin(), logits.end());
                std::vector<float> prob(logits.size());
                float sum = 0.f;
                for (std::size_t i = 0; i < logits.size(); ++i) {
                    prob[i] = std::exp(logits[i] - mx);
                    sum += prob[i];
                }
                for (float& p : prob) {
                    p /= sum;
                }

                // dL/d(logit_i) = prob_i - 1_{i=pos}
                // logit = cos/T ≈ dot/T since L2-normalized
                // We backprop through mean-pool embeddings (approx: ignore L2 norm Jacobian
                // for stability on small models — common practical simplification).
                std::vector<float> gq(m.dim_, 0.f);
                std::vector<std::vector<float>> gd(dvs.size(), std::vector<float>(m.dim_, 0.f));
                for (std::size_t i = 0; i < logits.size(); ++i) {
                    float g_logit = prob[i];
                    if (static_cast<int>(i) == pos_index) {
                        g_logit -= 1.f;
                    }
                    g_logit /= cfg.temperature;
                    for (std::size_t d = 0; d < m.dim_; ++d) {
                        gq[d] += g_logit * dvs[i][d];
                        gd[i][d] += g_logit * qv[d];
                    }
                }

                auto apply_token_grads = [&](const std::vector<int>& toks,
                                             const std::vector<float>& g_out) {
                    if (toks.empty()) {
                        return;
                    }
                    const float inv = 1.f / static_cast<float>(toks.size());
                    for (int tid : toks) {
                        float* row = m.emb_.data() + static_cast<std::size_t>(tid) * m.dim_;
                        for (std::size_t d = 0; d < m.dim_; ++d) {
                            row[d] -= lr * g_out[d] * inv;
                        }
                    }
                };

                apply_token_grads(pair.q, gq);
                for (std::size_t i = 0; i < cand_rows.size(); ++i) {
                    apply_token_grads(doc_tok[cand_rows[i]], gd[i]);
                }
            }
        }

        // L2-normalize rows for stable mean pooling at inference.
        for (int i = 0; i < V; ++i) {
            float* row = m.emb_.data() + static_cast<std::size_t>(i) * m.dim_;
            double ss = 0;
            for (std::size_t d = 0; d < m.dim_; ++d) {
                ss += static_cast<double>(row[d]) * row[d];
            }
            if (ss > 0) {
                const float inv = static_cast<float>(1.0 / std::sqrt(ss));
                for (std::size_t d = 0; d < m.dim_; ++d) {
                    row[d] *= inv;
                }
            }
        }
        return m;
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("ContrastiveModel::save: cannot open " + path);
        }
        out.write(kContrastiveMagic, 4);
        write_u32(out, kContrastiveFormatVersion);
        write_u32(out, static_cast<std::uint32_t>(dim_));
        write_u32(out, static_cast<std::uint32_t>(vocab_.size()));
        for (const auto& w : vocab_) {
            if (w.size() > 65535) {
                throw std::runtime_error("ContrastiveModel::save: token too long");
            }
            write_u16(out, static_cast<std::uint16_t>(w.size()));
            out.write(w.data(), static_cast<std::streamsize>(w.size()));
            const float* row = emb_.data() + static_cast<std::size_t>(token_to_id_.at(w)) * dim_;
            out.write(reinterpret_cast<const char*>(row),
                      static_cast<std::streamsize>(dim_ * sizeof(float)));
        }
    }

    static ContrastiveModel load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("ContrastiveModel::load: cannot open " + path);
        }
        char magic[4];
        in.read(magic, 4);
        if (in.gcount() != 4 || magic[0] != 'N' || magic[1] != 'C' || magic[2] != 'T' ||
            magic[3] != 'R') {
            throw std::runtime_error("ContrastiveModel::load: bad magic");
        }
        if (read_u32(in) != kContrastiveFormatVersion) {
            throw std::runtime_error("ContrastiveModel::load: bad version");
        }
        ContrastiveModel m;
        m.dim_ = read_u32(in);
        const auto V = read_u32(in);
        m.emb_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
        m.vocab_.reserve(V);
        for (std::uint32_t i = 0; i < V; ++i) {
            const auto len = read_u16(in);
            std::string w(len, '\0');
            in.read(w.data(), len);
            m.token_to_id_[w] = static_cast<int>(i);
            m.vocab_.push_back(std::move(w));
            in.read(reinterpret_cast<char*>(m.emb_.data() + static_cast<std::size_t>(i) * m.dim_),
                    static_cast<std::streamsize>(m.dim_ * sizeof(float)));
            if (!in) {
                throw std::runtime_error("ContrastiveModel::load: truncated");
            }
        }
        return m;
    }

private:
    std::size_t dim_ = 0;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<float> emb_;

    static void write_u32(std::ostream& out, std::uint32_t v) {
        out.write(reinterpret_cast<const char*>(&v), 4);
    }
    static void write_u16(std::ostream& out, std::uint16_t v) {
        out.write(reinterpret_cast<const char*>(&v), 2);
    }
    static std::uint32_t read_u32(std::istream& in) {
        std::uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        if (!in) {
            throw std::runtime_error("ContrastiveModel: short u32");
        }
        return v;
    }
    static std::uint16_t read_u16(std::istream& in) {
        std::uint16_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 2);
        if (!in) {
            throw std::runtime_error("ContrastiveModel: short u16");
        }
        return v;
    }
};

class ContrastiveEmbedder final : public Embedder {
public:
    explicit ContrastiveEmbedder(ContrastiveModel model) : model_(std::move(model)) {}

    std::size_t dim() const override { return model_.dim(); }
    std::string id() const override { return kContrastiveEmbedderId; }
    std::vector<float> embed(const std::string& text) const override {
        return model_.embed_text(text);
    }

    const ContrastiveModel& model() const { return model_; }
    void save(const std::string& path) const { model_.save(path); }

    static ContrastiveEmbedder load(const std::string& path) {
        return ContrastiveEmbedder(ContrastiveModel::load(path));
    }

    static ContrastiveEmbedder train(const ChunkStore& store, const std::vector<TrainPair>& pairs,
                                     ContrastiveTrainConfig cfg = {}) {
        return ContrastiveEmbedder(ContrastiveModel::train(store, pairs, cfg));
    }

private:
    ContrastiveModel model_;
};

}  // namespace nanorag
