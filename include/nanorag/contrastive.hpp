#pragma once

// Supervised contrastive dual encoder (pure C++17).
//
// contrastive-v2: same stable mean-pool InfoNCE as v1, plus:
//   - default dim 128, momentum SGD, cosine LR schedule
//   - hashed char n-grams (3/4) + token bigrams in the mean pool
//   - expanded synonymy training pairs (data/)
//
// Format: .nctr magic NCTR; version 2 (v1 still loadable as identity-feature model).

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

inline constexpr const char* kContrastiveEmbedderId = "contrastive-v2";
inline constexpr const char* kContrastiveEmbedderIdV1 = "contrastive-v1";
inline constexpr char kContrastiveMagic[4] = {'N', 'C', 'T', 'R'};
inline constexpr std::uint32_t kContrastiveFormatVersionV1 = 1;
inline constexpr std::uint32_t kContrastiveFormatVersion = 2;

struct TrainPair {
    std::string query;
    std::int64_t pos_id = 0;
};

struct ContrastiveTrainConfig {
    std::size_t dim = 128;
    int epochs = 420;
    float lr = 0.08f;
    float momentum = 0.9f;
    float temperature = 0.05f;
    std::uint32_t seed = 0xA11CEu;
    int min_count = 1;
    bool full_doc_softmax = true;
    int max_negatives = 32;
    std::size_t ngram_buckets = 4096;
    std::size_t bigram_buckets = 4096;
    float ngram_weight = 0.35f;
    float bigram_weight = 0.45f;
    /// Accepted for API compat; tables use dim for both when emb_dim==0.
    std::size_t emb_dim = 0;
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

namespace ctr_detail {
inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
inline std::uint64_t hash_str(const std::string& s, std::uint64_t salt) {
    return mix64(detail::fnv1a64(s) ^ salt);
}
inline void char_ngrams(const std::string& tok, std::size_t buckets, std::vector<std::size_t>& out) {
    if (buckets == 0 || tok.empty()) {
        return;
    }
    const std::string s = std::string("^") + tok + "$";
    for (int n : {3, 4}) {
        if (static_cast<int>(s.size()) < n) {
            continue;
        }
        for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= s.size(); ++i) {
            out.push_back(static_cast<std::size_t>(
                hash_str(s.substr(i, static_cast<std::size_t>(n)), 0xC4A7ull) % buckets));
        }
    }
}
}  // namespace ctr_detail

class ContrastiveModel {
public:
    std::size_t dim() const { return dim_; }
    std::size_t vocab_size() const { return vocab_.size(); }
    std::uint32_t format_version() const { return format_version_; }

    std::vector<float> embed_text(const std::string& text) const {
        return embed_tokens(detail::simple_tokenize(text));
    }

    std::vector<float> embed_tokens(const std::vector<std::string>& tokens) const {
        return pool_tokens(tokens);
    }

    static ContrastiveModel train(const ChunkStore& store, const std::vector<TrainPair>& pairs,
                                  ContrastiveTrainConfig cfg = {}) {
        if (cfg.dim == 0) {
            throw std::invalid_argument("ContrastiveModel: dim must be > 0");
        }
        if (cfg.emb_dim != 0) {
            cfg.dim = cfg.emb_dim;  // single width tables
        }
        if (store.size() == 0 || pairs.empty()) {
            throw std::invalid_argument("ContrastiveModel: empty store or pairs");
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
                throw std::invalid_argument("ContrastiveModel: pair pos_id missing: " +
                                            std::to_string(p.pos_id));
            }
        }

        ContrastiveModel m;
        m.format_version_ = kContrastiveFormatVersion;
        m.dim_ = cfg.dim;
        m.ngram_buckets_ = cfg.ngram_buckets;
        m.bigram_buckets_ = cfg.bigram_buckets;
        m.ngram_weight_ = cfg.ngram_weight;
        m.bigram_weight_ = cfg.bigram_weight;

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
        m.ngram_emb_.assign(m.ngram_buckets_ * m.dim_, 0.f);
        m.bigram_emb_.assign(m.bigram_buckets_ * m.dim_, 0.f);

        std::mt19937 rng(cfg.seed);
        std::uniform_real_distribution<float> ur(-0.5f, 0.5f);
        const float scale = 1.f / static_cast<float>(m.dim_);
        for (float& x : m.emb_) {
            x = ur(rng) * scale;
        }
        for (float& x : m.ngram_emb_) {
            x = ur(rng) * scale * 0.5f;
        }
        for (float& x : m.bigram_emb_) {
            x = ur(rng) * scale * 0.5f;
        }

        std::vector<float> vel_emb(m.emb_.size(), 0.f);
        std::vector<float> vel_ng(m.ngram_emb_.size(), 0.f);
        std::vector<float> vel_bi(m.bigram_emb_.size(), 0.f);

        const auto chunks = store.all();
        std::unordered_map<std::int64_t, std::size_t> id_to_row;
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            id_to_row[chunks[i].id] = i;
        }

        struct PairTok {
            std::vector<std::string> q;
            std::vector<std::string> pos;
            std::int64_t pos_id = 0;
        };
        std::vector<PairTok> pt;
        pt.reserve(pairs.size());
        for (const auto& p : pairs) {
            PairTok x;
            x.q = detail::simple_tokenize(p.query);
            x.pos = detail::simple_tokenize(store.get(p.pos_id).text);
            x.pos_id = p.pos_id;
            if (x.q.empty() || x.pos.empty()) {
                throw std::invalid_argument("ContrastiveModel: empty tokens after tokenize");
            }
            pt.push_back(std::move(x));
        }
        std::vector<std::vector<std::string>> doc_tok(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            doc_tok[i] = detail::simple_tokenize(chunks[i].text);
        }

        auto apply = [&](std::vector<float>& param, std::vector<float>& vel,
                         const std::vector<float>& grad, float lr) {
            for (std::size_t i = 0; i < param.size(); ++i) {
                vel[i] = cfg.momentum * vel[i] + grad[i];
                param[i] -= lr * vel[i];
            }
        };

        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            const float t =
                static_cast<float>(epoch) / static_cast<float>(std::max(1, cfg.epochs - 1));
            const float lr = cfg.lr * (0.15f + 0.85f * 0.5f * (1.f + std::cos(3.14159265f * t)));
            std::shuffle(pt.begin(), pt.end(), rng);

            for (const auto& pair : pt) {
                std::vector<std::size_t> cand_rows;
                if (cfg.full_doc_softmax) {
                    cand_rows.resize(chunks.size());
                    for (std::size_t i = 0; i < chunks.size(); ++i) {
                        cand_rows[i] = i;
                    }
                } else {
                    cand_rows.push_back(id_to_row.at(pair.pos_id));
                    // Never ask for more distinct rows than the store holds — a small
                    // corpus would otherwise spin forever waiting for an unreachable
                    // (max_negatives + 1)-th distinct row.
                    const std::size_t want =
                        std::min(static_cast<std::size_t>(std::max(0, cfg.max_negatives)) + 1,
                                 chunks.size());
                    std::uniform_int_distribution<std::size_t> ud(0, chunks.size() - 1);
                    while (cand_rows.size() < want) {
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

                auto qv = m.pool_tokens(pair.q);
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
                    auto dv = m.pool_tokens(doc_tok[row]);
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

                std::vector<float> g_emb(m.emb_.size(), 0.f);
                std::vector<float> g_ng(m.ngram_emb_.size(), 0.f);
                std::vector<float> g_bi(m.bigram_emb_.size(), 0.f);
                m.accum_grad(pair.q, gq, g_emb, g_ng, g_bi);
                for (std::size_t i = 0; i < cand_rows.size(); ++i) {
                    m.accum_grad(doc_tok[cand_rows[i]], gd[i], g_emb, g_ng, g_bi);
                }
                apply(m.emb_, vel_emb, g_emb, lr);
                if (!m.ngram_emb_.empty()) {
                    apply(m.ngram_emb_, vel_ng, g_ng, lr);
                }
                if (!m.bigram_emb_.empty()) {
                    apply(m.bigram_emb_, vel_bi, g_bi, lr);
                }
            }
        }

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
        write_u32(out, static_cast<std::uint32_t>(ngram_buckets_));
        write_u32(out, static_cast<std::uint32_t>(bigram_buckets_));
        write_f32(out, ngram_weight_);
        write_f32(out, bigram_weight_);
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
        write_buf(out, ngram_emb_);
        write_buf(out, bigram_emb_);
        out.flush();
        if (!out) {
            throw std::runtime_error("ContrastiveModel::save: write failed");
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
        const auto ver = read_u32(in);
        if (ver == kContrastiveFormatVersionV1) {
            ContrastiveModel m;
            m.format_version_ = ver;
            m.dim_ = read_u32(in);
            m.ngram_buckets_ = 0;
            m.bigram_buckets_ = 0;
            const auto V = read_u32(in);
            m.emb_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
            for (std::uint32_t i = 0; i < V; ++i) {
                const auto len = read_u16(in);
                std::string w(len, '\0');
                in.read(w.data(), static_cast<std::streamsize>(len));
                m.token_to_id_[w] = static_cast<int>(i);
                m.vocab_.push_back(std::move(w));
                in.read(reinterpret_cast<char*>(m.emb_.data() + static_cast<std::size_t>(i) * m.dim_),
                        static_cast<std::streamsize>(m.dim_ * sizeof(float)));
                if (!in) {
                    throw std::runtime_error("ContrastiveModel::load v1 truncated");
                }
            }
            return m;
        }
        if (ver != kContrastiveFormatVersion) {
            throw std::runtime_error("ContrastiveModel::load: bad version");
        }
        ContrastiveModel m;
        m.format_version_ = ver;
        m.dim_ = read_u32(in);
        m.ngram_buckets_ = read_u32(in);
        m.bigram_buckets_ = read_u32(in);
        m.ngram_weight_ = read_f32(in);
        m.bigram_weight_ = read_f32(in);
        const auto V = read_u32(in);
        m.emb_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
        for (std::uint32_t i = 0; i < V; ++i) {
            const auto len = read_u16(in);
            std::string w(len, '\0');
            in.read(w.data(), static_cast<std::streamsize>(len));
            m.token_to_id_[w] = static_cast<int>(i);
            m.vocab_.push_back(std::move(w));
            in.read(reinterpret_cast<char*>(m.emb_.data() + static_cast<std::size_t>(i) * m.dim_),
                    static_cast<std::streamsize>(m.dim_ * sizeof(float)));
            if (!in) {
                throw std::runtime_error("ContrastiveModel::load truncated");
            }
        }
        m.ngram_emb_.assign(m.ngram_buckets_ * m.dim_, 0.f);
        m.bigram_emb_.assign(m.bigram_buckets_ * m.dim_, 0.f);
        read_buf(in, m.ngram_emb_);
        read_buf(in, m.bigram_emb_);
        return m;
    }

private:
    std::uint32_t format_version_ = kContrastiveFormatVersion;
    std::size_t dim_ = 0;
    std::size_t ngram_buckets_ = 0;
    std::size_t bigram_buckets_ = 0;
    float ngram_weight_ = 0.35f;
    float bigram_weight_ = 0.45f;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<float> emb_;
    std::vector<float> ngram_emb_;
    std::vector<float> bigram_emb_;

    std::vector<float> pool_tokens(const std::vector<std::string>& tokens) const {
        std::vector<float> v(dim_, 0.f);
        float wsum = 0.f;
        for (const auto& t : tokens) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) {
                const float* row = emb_.data() + static_cast<std::size_t>(it->second) * dim_;
                for (std::size_t d = 0; d < dim_; ++d) {
                    v[d] += row[d];
                }
                wsum += 1.f;
            }
            if (ngram_buckets_ > 0 && !ngram_emb_.empty()) {
                std::vector<std::size_t> ngs;
                ctr_detail::char_ngrams(t, ngram_buckets_, ngs);
                for (std::size_t b : ngs) {
                    const float* row = ngram_emb_.data() + b * dim_;
                    for (std::size_t d = 0; d < dim_; ++d) {
                        v[d] += ngram_weight_ * row[d];
                    }
                    wsum += ngram_weight_;
                }
            }
        }
        if (bigram_buckets_ > 0 && !bigram_emb_.empty()) {
            for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
                const std::string bg = tokens[i] + "_" + tokens[i + 1];
                const std::size_t b =
                    static_cast<std::size_t>(ctr_detail::hash_str(bg, 0xB16Aull) % bigram_buckets_);
                const float* row = bigram_emb_.data() + b * dim_;
                for (std::size_t d = 0; d < dim_; ++d) {
                    v[d] += bigram_weight_ * row[d];
                }
                wsum += bigram_weight_;
            }
        }
        if (wsum > 0.f) {
            const float inv = 1.f / wsum;
            for (float& x : v) {
                x *= inv;
            }
            detail::l2_normalize(v);
        }
        return v;
    }

    void accum_grad(const std::vector<std::string>& tokens, const std::vector<float>& g_out,
                    std::vector<float>& g_emb, std::vector<float>& g_ng,
                    std::vector<float>& g_bi) const {
        // Approximate: ignore L2 jacobian; distribute mean-pool grads.
        float wsum = 0.f;
        for (const auto& t : tokens) {
            if (token_to_id_.count(t)) {
                wsum += 1.f;
            }
            if (ngram_buckets_ > 0) {
                std::vector<std::size_t> ngs;
                ctr_detail::char_ngrams(t, ngram_buckets_, ngs);
                wsum += ngram_weight_ * static_cast<float>(ngs.size());
            }
        }
        if (bigram_buckets_ > 0 && tokens.size() > 1) {
            wsum += bigram_weight_ * static_cast<float>(tokens.size() - 1);
        }
        if (!(wsum > 0.f)) {
            return;
        }
        const float inv = 1.f / wsum;
        for (const auto& t : tokens) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) {
                float* grow = g_emb.data() + static_cast<std::size_t>(it->second) * dim_;
                for (std::size_t d = 0; d < dim_; ++d) {
                    grow[d] += g_out[d] * inv;
                }
            }
            if (ngram_buckets_ > 0 && !g_ng.empty()) {
                std::vector<std::size_t> ngs;
                ctr_detail::char_ngrams(t, ngram_buckets_, ngs);
                for (std::size_t b : ngs) {
                    float* grow = g_ng.data() + b * dim_;
                    for (std::size_t d = 0; d < dim_; ++d) {
                        grow[d] += ngram_weight_ * g_out[d] * inv;
                    }
                }
            }
        }
        if (bigram_buckets_ > 0 && !g_bi.empty()) {
            for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
                const std::string bg = tokens[i] + "_" + tokens[i + 1];
                const std::size_t b =
                    static_cast<std::size_t>(ctr_detail::hash_str(bg, 0xB16Aull) % bigram_buckets_);
                float* grow = g_bi.data() + b * dim_;
                for (std::size_t d = 0; d < dim_; ++d) {
                    grow[d] += bigram_weight_ * g_out[d] * inv;
                }
            }
        }
    }

    static void write_u32(std::ostream& o, std::uint32_t v) {
        o.write(reinterpret_cast<const char*>(&v), 4);
    }
    static void write_u16(std::ostream& o, std::uint16_t v) {
        o.write(reinterpret_cast<const char*>(&v), 2);
    }
    static void write_f32(std::ostream& o, float v) {
        o.write(reinterpret_cast<const char*>(&v), 4);
    }
    static void write_buf(std::ostream& o, const std::vector<float>& v) {
        if (!v.empty()) {
            o.write(reinterpret_cast<const char*>(v.data()),
                    static_cast<std::streamsize>(v.size() * sizeof(float)));
        }
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
    static float read_f32(std::istream& in) {
        float v = 0;
        in.read(reinterpret_cast<char*>(&v), 4);
        if (!in) {
            throw std::runtime_error("ContrastiveModel: short f32");
        }
        return v;
    }
    static void read_buf(std::istream& in, std::vector<float>& v) {
        if (v.empty()) {
            return;
        }
        in.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(float)));
        if (!in) {
            throw std::runtime_error("ContrastiveModel: short buf");
        }
    }
};

class ContrastiveEmbedder final : public Embedder {
public:
    explicit ContrastiveEmbedder(ContrastiveModel model) : model_(std::move(model)) {}

    std::size_t dim() const override { return model_.dim(); }
    std::string id() const override {
        return model_.format_version() == kContrastiveFormatVersionV1 ? kContrastiveEmbedderIdV1
                                                                      : kContrastiveEmbedderId;
    }
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
