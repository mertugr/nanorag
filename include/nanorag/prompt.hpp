#pragma once

#include "nanorag/chunk_store.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "tinyann/tinyann.hpp"

namespace nanorag {

struct RetrievedChunk {
    std::int64_t id = 0;
    float score = 0.f;
    std::string source;
    std::string text;
    /// True when `score` is dense cosine (or equivalent in [~-1,1]).
    /// False for BM25 / fusion ranking scores that must not drive paraphrase gates.
    bool score_is_cosine = true;
};

// Forward declaration — full grounded prompt lives in grounding.hpp to avoid cycles.
// Legacy helper kept for raw dumps; prefer build_grounded_prompt via grounding.hpp.
inline std::string build_rag_prompt(const std::string& question,
                                    const std::vector<RetrievedChunk>& hits) {
    std::ostringstream oss;
    oss << "Use ONLY the context. If insufficient, answer exactly: I don't know. "
           "Cite every claim as [#id] using only context ids.\n\n";
    oss << "Context:\n";
    if (hits.empty()) {
        oss << "(no retrieved passages)\n";
    } else {
        for (const auto& h : hits) {
            oss << "[#" << h.id << " source=" << h.source << " score=" << h.score << "]\n";
            oss << h.text << "\n\n";
        }
    }
    oss << "Question: " << question << "\n\nAnswer:";
    return oss.str();
}

inline std::vector<RetrievedChunk> hits_to_chunks(const std::vector<tinyann::SearchResult>& hits,
                                                  const ChunkStore& store) {
    std::vector<RetrievedChunk> out;
    out.reserve(hits.size());
    for (const auto& h : hits) {
        RetrievedChunk rc;
        rc.id = h.id;
        rc.score = h.score;
        if (store.contains(h.id)) {
            const Chunk& c = store.get(h.id);
            rc.source = c.source;
            rc.text = c.text;
        } else {
            rc.source = "?";
            rc.text = "(missing chunk text for id)";
        }
        out.push_back(std::move(rc));
    }
    return out;
}

}  // namespace nanorag
