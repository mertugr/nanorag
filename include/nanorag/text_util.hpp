#pragma once

#include "nanorag/embedder.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

namespace nanorag {

/// Jaccard similarity over token sets (after simple_tokenize / light_stem).
inline double token_jaccard(const std::string& a, const std::string& b) {
    const auto ta = detail::simple_tokenize(a);
    const auto tb = detail::simple_tokenize(b);
    if (ta.empty() && tb.empty()) {
        return 1.0;
    }
    if (ta.empty() || tb.empty()) {
        return 0.0;
    }
    std::unordered_set<std::string> sa(ta.begin(), ta.end());
    std::unordered_set<std::string> sb(tb.begin(), tb.end());
    std::size_t inter = 0;
    for (const auto& t : sa) {
        if (sb.count(t)) {
            ++inter;
        }
    }
    const std::size_t uni = sa.size() + sb.size() - inter;
    return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

inline double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0 || nb <= 0) {
        return 0.0;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace nanorag
