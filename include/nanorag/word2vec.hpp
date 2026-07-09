#pragma once

// In-house skip-gram Word2Vec with negative sampling.
// Trained on the ingest corpus in pure C++17 — no external ML libs.

#include "nanorag/embedder.hpp"

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

inline constexpr const char* kWord2VecEmbedderId = "word2vec-v1";
inline constexpr char kWord2VecMagic[4] = {'N', 'W', '2', 'V'};
inline constexpr std::uint32_t kWord2VecFormatVersion = 1;

struct Word2VecTrainConfig {
    std::size_t dim = 64;
    int epochs = 40;
    int window = 5;
    int negative = 5;
    float lr = 0.05f;
    int min_count = 1;
    std::uint32_t seed = 0xC0FFEEu;
    /// Repeat each document this many times (small corpora need more signal).
    int doc_repeat = 8;
};

class Word2VecModel {
public:
    std::size_t dim() const { return dim_; }
    std::size_t vocab_size() const { return vocab_.size(); }
    const std::vector<std::string>& vocab() const { return vocab_; }

    bool has(const std::string& token) const { return token_to_id_.count(token) != 0; }

    /// Mean of in-vocab word vectors, L2-normalized. Empty/OOV-only → zero vector.
    std::vector<float> embed_tokens(const std::vector<std::string>& tokens) const {
        std::vector<float> v(dim_, 0.f);
        int n = 0;
        for (const auto& t : tokens) {
            auto it = token_to_id_.find(t);
            if (it == token_to_id_.end()) {
                continue;
            }
            const float* row = row_ptr(it->second);
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

    std::vector<float> embed_text(const std::string& text) const {
        return embed_tokens(detail::simple_tokenize(text));
    }

    static Word2VecModel train(const std::vector<std::string>& documents,
                               Word2VecTrainConfig cfg = {}) {
        if (cfg.dim == 0) {
            throw std::invalid_argument("Word2Vec: dim must be > 0");
        }
        if (documents.empty()) {
            throw std::invalid_argument("Word2Vec: empty corpus");
        }

        // Tokenize + count
        std::unordered_map<std::string, int> counts;
        std::vector<std::vector<std::string>> docs_tok;
        docs_tok.reserve(documents.size() * static_cast<std::size_t>(std::max(1, cfg.doc_repeat)));
        for (int r = 0; r < std::max(1, cfg.doc_repeat); ++r) {
            for (const auto& doc : documents) {
                auto toks = detail::simple_tokenize(doc);
                if (toks.empty()) {
                    continue;
                }
                for (const auto& t : toks) {
                    counts[t] += 1;
                }
                docs_tok.push_back(std::move(toks));
            }
        }
        if (docs_tok.empty()) {
            throw std::invalid_argument("Word2Vec: no tokens in corpus");
        }

        Word2VecModel m;
        m.dim_ = cfg.dim;
        for (const auto& kv : counts) {
            if (kv.second >= cfg.min_count) {
                m.token_to_id_[kv.first] = static_cast<int>(m.vocab_.size());
                m.vocab_.push_back(kv.first);
            }
        }
        if (m.vocab_.empty()) {
            throw std::invalid_argument("Word2Vec: vocab empty after min_count");
        }

        const int V = static_cast<int>(m.vocab_.size());
        m.input_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
        m.output_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);

        std::mt19937 rng(cfg.seed);
        std::uniform_real_distribution<float> ur(-0.5f, 0.5f);
        const float scale = 1.f / static_cast<float>(m.dim_);
        for (float& x : m.input_) {
            x = ur(rng) * scale;
        }
        // output_ stays 0 (common init)

        // Negative sampling table ~ unigram^0.75
        std::vector<double> freqs(static_cast<std::size_t>(V), 0.0);
        for (const auto& t : m.vocab_) {
            freqs[static_cast<std::size_t>(m.token_to_id_.at(t))] =
                std::pow(static_cast<double>(counts[t]), 0.75);
        }
        std::vector<int> table;
        table.reserve(100000);
        double sum = 0;
        for (double f : freqs) {
            sum += f;
        }
        for (int i = 0; i < V; ++i) {
            const int n = std::max(1, static_cast<int>(freqs[static_cast<std::size_t>(i)] / sum * 100000.0));
            for (int k = 0; k < n; ++k) {
                table.push_back(i);
            }
        }
        std::uniform_int_distribution<std::size_t> tdist(0, table.size() - 1);

        auto sigmoid = [](float x) {
            if (x > 12.f) {
                return 1.f;
            }
            if (x < -12.f) {
                return 0.f;
            }
            return 1.f / (1.f + std::exp(-x));
        };

        auto train_pair = [&](int center, int context, float label, float lr) {
            float* in = m.row_mut(m.input_, center);
            float* out = m.row_mut(m.output_, context);
            float dot = 0.f;
            for (std::size_t d = 0; d < m.dim_; ++d) {
                dot += in[d] * out[d];
            }
            const float pred = sigmoid(dot);
            const float err = (label - pred) * lr;
            // grad for input and output
            std::vector<float> in_grad(m.dim_, 0.f);
            for (std::size_t d = 0; d < m.dim_; ++d) {
                in_grad[d] = err * out[d];
                out[d] += err * in[d];
            }
            for (std::size_t d = 0; d < m.dim_; ++d) {
                in[d] += in_grad[d];
            }
        };

        // Filter docs to in-vocab ids
        std::vector<std::vector<int>> corpus;
        corpus.reserve(docs_tok.size());
        for (const auto& toks : docs_tok) {
            std::vector<int> ids;
            ids.reserve(toks.size());
            for (const auto& t : toks) {
                auto it = m.token_to_id_.find(t);
                if (it != m.token_to_id_.end()) {
                    ids.push_back(it->second);
                }
            }
            if (ids.size() >= 2) {
                corpus.push_back(std::move(ids));
            }
        }
        if (corpus.empty()) {
            throw std::invalid_argument("Word2Vec: no training pairs");
        }

        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            const float lr = cfg.lr * (1.f - static_cast<float>(epoch) / static_cast<float>(cfg.epochs));
            std::shuffle(corpus.begin(), corpus.end(), rng);
            for (const auto& sent : corpus) {
                const int n = static_cast<int>(sent.size());
                for (int i = 0; i < n; ++i) {
                    const int center = sent[static_cast<std::size_t>(i)];
                    const int left = std::max(0, i - cfg.window);
                    const int right = std::min(n - 1, i + cfg.window);
                    for (int j = left; j <= right; ++j) {
                        if (j == i) {
                            continue;
                        }
                        const int ctx = sent[static_cast<std::size_t>(j)];
                        train_pair(center, ctx, 1.f, lr);
                        for (int neg = 0; neg < cfg.negative; ++neg) {
                            int ni = table[tdist(rng)];
                            if (ni == center || ni == ctx) {
                                continue;
                            }
                            train_pair(center, ni, 0.f, lr);
                        }
                    }
                }
            }
        }

        // Final L2-normalize each input word vector for stable averaging.
        for (int i = 0; i < V; ++i) {
            float* row = m.row_mut(m.input_, i);
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
            throw std::runtime_error("Word2VecModel::save: cannot open " + path);
        }
        out.write(kWord2VecMagic, 4);
        write_u32(out, kWord2VecFormatVersion);
        write_u32(out, static_cast<std::uint32_t>(dim_));
        write_u32(out, static_cast<std::uint32_t>(vocab_.size()));
        for (const auto& w : vocab_) {
            if (w.size() > 65535) {
                throw std::runtime_error("Word2VecModel::save: token too long");
            }
            write_u16(out, static_cast<std::uint16_t>(w.size()));
            out.write(w.data(), static_cast<std::streamsize>(w.size()));
            const float* row = row_ptr(token_to_id_.at(w));
            out.write(reinterpret_cast<const char*>(row),
                      static_cast<std::streamsize>(dim_ * sizeof(float)));
        }
        if (!out) {
            throw std::runtime_error("Word2VecModel::save: write failed");
        }
    }

    static Word2VecModel load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Word2VecModel::load: cannot open " + path);
        }
        char magic[4];
        in.read(magic, 4);
        if (in.gcount() != 4 || magic[0] != kWord2VecMagic[0] || magic[1] != kWord2VecMagic[1] ||
            magic[2] != kWord2VecMagic[2] || magic[3] != kWord2VecMagic[3]) {
            throw std::runtime_error("Word2VecModel::load: bad magic");
        }
        const auto ver = read_u32(in);
        if (ver != kWord2VecFormatVersion) {
            throw std::runtime_error("Word2VecModel::load: unsupported version");
        }
        Word2VecModel m;
        m.dim_ = read_u32(in);
        const auto V = read_u32(in);
        if (m.dim_ == 0 || V == 0) {
            throw std::runtime_error("Word2VecModel::load: empty model");
        }
        m.input_.assign(static_cast<std::size_t>(V) * m.dim_, 0.f);
        m.output_.clear();
        m.vocab_.reserve(V);
        for (std::uint32_t i = 0; i < V; ++i) {
            const auto len = read_u16(in);
            std::string w(len, '\0');
            in.read(w.data(), len);
            if (!in) {
                throw std::runtime_error("Word2VecModel::load: truncated token");
            }
            m.token_to_id_[w] = static_cast<int>(i);
            m.vocab_.push_back(std::move(w));
            in.read(reinterpret_cast<char*>(m.row_mut(m.input_, static_cast<int>(i))),
                    static_cast<std::streamsize>(m.dim_ * sizeof(float)));
            if (!in) {
                throw std::runtime_error("Word2VecModel::load: truncated vectors");
            }
        }
        return m;
    }

private:
    std::size_t dim_ = 0;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<float> input_;   // [V, dim] row-major
    std::vector<float> output_;  // training only

    const float* row_ptr(int id) const {
        return input_.data() + static_cast<std::size_t>(id) * dim_;
    }
    float* row_mut(std::vector<float>& mat, int id) {
        return mat.data() + static_cast<std::size_t>(id) * dim_;
    }

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
            throw std::runtime_error("Word2VecModel: short read u32");
        }
        return v;
    }
    static std::uint16_t read_u16(std::istream& in) {
        std::uint16_t v = 0;
        in.read(reinterpret_cast<char*>(&v), 2);
        if (!in) {
            throw std::runtime_error("Word2VecModel: short read u16");
        }
        return v;
    }
};

/// Embedder backed by a trained Word2VecModel (mean pooling + L2).
class Word2VecEmbedder final : public Embedder {
public:
    explicit Word2VecEmbedder(Word2VecModel model) : model_(std::move(model)) {}

    std::size_t dim() const override { return model_.dim(); }
    std::string id() const override { return kWord2VecEmbedderId; }

    std::vector<float> embed(const std::string& text) const override {
        return model_.embed_text(text);
    }

    const Word2VecModel& model() const { return model_; }

    void save(const std::string& path) const { model_.save(path); }

    static Word2VecEmbedder load(const std::string& path) {
        return Word2VecEmbedder(Word2VecModel::load(path));
    }

    static Word2VecEmbedder train(const std::vector<std::string>& documents,
                                  Word2VecTrainConfig cfg = {}) {
        return Word2VecEmbedder(Word2VecModel::train(documents, cfg));
    }

private:
    Word2VecModel model_;
};

}  // namespace nanorag
