#pragma once

// nanorag embedders — all in-house; no external embedding services or model hubs.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanorag {

/// Stable id written into index meta so query-time embedder must match ingest.
inline constexpr const char* kHashingEmbedderId = "hashing-v1";

class Embedder {
public:
    virtual ~Embedder() = default;
    virtual std::size_t dim() const = 0;
    virtual std::string id() const = 0;
    virtual std::vector<float> embed(const std::string& text) const = 0;

    virtual std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) const {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) {
            out.push_back(embed(t));
        }
        return out;
    }
};

namespace detail {

inline void l2_normalize(std::vector<float>& v) {
    double ss = 0.0;
    for (float x : v) {
        ss += static_cast<double>(x) * static_cast<double>(x);
    }
    if (ss <= 0.0) {
        return;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(ss));
    for (float& x : v) {
        x *= inv;
    }
}

inline std::uint64_t fnv1a64(const std::string& s) {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

/// Conservative plural fold only: cats→cat, graphs→graph.
/// Does NOT strip -ed/-ing (those mangled shared→shar, class→clas).
inline std::string light_stem(std::string t) {
    if (t.size() > 4 && t.back() == 's' && t[t.size() - 2] != 's') {
        // avoid ss→s; keep short tokens
        t.pop_back();
    }
    return t;
}

/// Lowercase ASCII + alnum tokens; non-ASCII bytes kept (UTF-8 multi-byte as substrings).
inline std::vector<std::string> simple_tokenize(const std::string& text) {
    std::vector<std::string> toks;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            toks.push_back(light_stem(std::move(cur)));
            cur.clear();
        }
    };
    for (unsigned char c : text) {
        if (std::isalnum(c) || (c & 0x80)) {
            if (c < 128) {
                cur.push_back(static_cast<char>(std::tolower(c)));
            } else {
                cur.push_back(static_cast<char>(c));
            }
        } else {
            flush();
        }
    }
    flush();
    return toks;
}

}  // namespace detail

/// Feature-hashing bag-of-tokens embedder. Deterministic, L2-normalized for cosine.
class HashingEmbedder final : public Embedder {
public:
    explicit HashingEmbedder(std::size_t dim = 512) : dim_(dim) {
        if (dim_ == 0) {
            throw std::invalid_argument("HashingEmbedder: dim must be > 0");
        }
    }

    std::size_t dim() const override { return dim_; }
    std::string id() const override { return kHashingEmbedderId; }

    std::vector<float> embed(const std::string& text) const override {
        std::vector<float> v(dim_, 0.f);
        const auto toks = detail::simple_tokenize(text);
        if (toks.empty()) {
            return v;
        }
        for (const auto& tok : toks) {
            const std::uint64_t h = detail::fnv1a64(tok);
            const std::size_t i = static_cast<std::size_t>(h % dim_);
            const float sign = ((h >> 32) & 1ull) ? 1.f : -1.f;
            v[i] += sign;
        }
        detail::l2_normalize(v);
        return v;
    }

private:
    std::size_t dim_;
};

}  // namespace nanorag
