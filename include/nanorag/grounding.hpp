#pragma once

// Grounding + citations for nanorag.
//
// Policy:
//   1) Retrieve k candidates (scores = query–doc similarity)
//   2) Keep hits that pass the relevance gate (score + shuffle baseline)
//   3) If none remain → answer exactly "I don't know" (no generation)
//   4) Otherwise answer only from those passages and cite [#id]
//   5) validate_grounding() proves citations + lexical support from context

#include "nanorag/embedder.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/text_util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanorag {

inline constexpr const char* kDontKnowAnswer = "I don't know";
/// Sentinel chunk id trained as the target for out-of-scope queries.
inline constexpr std::int64_t kNoEvidenceId = -1;
inline constexpr const char* kNoEvidenceText =
    "NO_EVIDENCE: the corpus contains no answer to this question.";

struct GroundingConfig {
    /// Absolute minimum retrieval score (cosine-like).
    float min_score = 0.25f;
    /// Require score(query,doc) - score(shuffle(query),doc) >= margin (spurious match filter).
    float min_score_margin = 0.05f;
    bool use_shuffle_baseline = true;
    std::uint32_t shuffle_seed = 0xBEE5u;
    std::size_t max_context_chunks = 3;
    bool require_citations = true;
    /// Min fraction of answer content tokens that must appear in context (0–1).
    float min_content_support = 0.25f;
};

struct GroundingCheck {
    bool ok = false;
    bool refused = false;
    std::string reason;
    std::vector<std::int64_t> cited_ids;
    std::vector<std::int64_t> illegal_ids;
    double content_support = 0.0;
};

struct GroundedAnswer {
    bool refused = false;
    std::string answer;
    std::string mode;  // refuse | extractive | generated | extractive_fallback
    std::vector<RetrievedChunk> candidates;
    std::vector<RetrievedChunk> used;
    GroundingCheck check;
    std::string prompt;
};

inline std::string normalize_ws(std::string s) {
    std::string out;
    out.reserve(s.size());
    bool sp = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!sp) {
                out.push_back(' ');
                sp = true;
            }
        } else {
            out.push_back(c);
            sp = false;
        }
    }
    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

inline std::string to_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

inline bool is_dont_know(const std::string& answer) {
    auto t = to_lower_copy(normalize_ws(answer));
    if (t == "i don't know" || t == "i do not know") {
        return true;
    }
    if (t.rfind("i don't know", 0) == 0 || t.rfind("i do not know", 0) == 0) {
        return true;
    }
    return false;
}

inline std::string shuffle_query_tokens(const std::string& query, std::uint32_t seed) {
    auto toks = detail::simple_tokenize(query);
    if (toks.size() < 2) {
        return query;
    }
    std::mt19937 rng(seed);
    std::shuffle(toks.begin(), toks.end(), rng);
    std::ostringstream oss;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        if (i) {
            oss << ' ';
        }
        oss << toks[i];
    }
    return oss.str();
}

/// Score-only filter (for unit tests with canned candidates, no embedder).
inline bool is_no_evidence_chunk(const RetrievedChunk& h) {
    return h.id == kNoEvidenceId;
}

/// True if retrieval indicates no answer: NO_EVIDENCE is ranked first and no
/// real passage clears min_score.
inline bool should_refuse_no_evidence(const std::vector<RetrievedChunk>& hits, float min_score) {
    if (hits.empty() || !is_no_evidence_chunk(hits.front())) {
        return false;
    }
    for (std::size_t i = 1; i < hits.size(); ++i) {
        if (!is_no_evidence_chunk(hits[i]) && hits[i].score >= min_score) {
            return false;  // a real passage still clears the bar
        }
    }
    return true;
}

inline std::vector<RetrievedChunk> filter_relevant_by_score(const std::vector<RetrievedChunk>& hits,
                                                            float min_score, std::size_t max_n) {
    if (should_refuse_no_evidence(hits, min_score)) {
        return {};
    }
    std::vector<RetrievedChunk> out;
    for (const auto& h : hits) {
        if (is_no_evidence_chunk(h)) {
            continue;
        }
        if (h.score >= min_score) {
            out.push_back(h);
            if (out.size() >= max_n) {
                break;
            }
        }
    }
    return out;
}

/// Full relevance gate: min score + optional shuffle baseline (needs embedder).
inline std::vector<RetrievedChunk> filter_relevant(const std::vector<RetrievedChunk>& hits,
                                                   const std::string& query,
                                                   const Embedder* embedder,
                                                   const GroundingConfig& cfg) {
    (void)query;
    (void)embedder;
    // OOD: trained NO_EVIDENCE sentinel (id=-1). Only refuse when it wins and no
    // real passage clears min_score (bag-of-words cannot use order baselines).
    if (should_refuse_no_evidence(hits, cfg.min_score)) {
        return {};
    }
    std::vector<RetrievedChunk> out;
    for (const auto& h : hits) {
        if (is_no_evidence_chunk(h)) {
            continue;
        }
        if (h.score < cfg.min_score) {
            continue;
        }
        out.push_back(h);
        if (out.size() >= cfg.max_context_chunks) {
            break;
        }
    }
    return out;
}

inline std::vector<std::int64_t> extract_citation_ids(const std::string& answer) {
    static const std::regex re(R"(\[#?\s*(-?\d+)\s*\])");
    std::vector<std::int64_t> ids;
    std::sregex_iterator it(answer.begin(), answer.end(), re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        ids.push_back(std::stoll((*it)[1].str()));
    }
    std::unordered_set<std::int64_t> seen;
    std::vector<std::int64_t> uniq;
    for (auto id : ids) {
        if (seen.insert(id).second) {
            uniq.push_back(id);
        }
    }
    return uniq;
}

inline std::unordered_set<std::int64_t> allowed_ids(const std::vector<RetrievedChunk>& used) {
    std::unordered_set<std::int64_t> s;
    for (const auto& h : used) {
        s.insert(h.id);
    }
    return s;
}

inline std::string context_blob(const std::vector<RetrievedChunk>& used) {
    std::ostringstream oss;
    for (const auto& h : used) {
        oss << h.text << ' ';
    }
    return oss.str();
}

inline double content_support_ratio(const std::string& answer,
                                    const std::vector<RetrievedChunk>& used) {
    std::string cleaned = std::regex_replace(answer, std::regex(R"(\[#?\s*-?\d+\s*\])"), " ");
    auto atoks = detail::simple_tokenize(cleaned);
    static const std::unordered_set<std::string> stop = {
        "a",    "an",   "the",  "is",   "are",  "was",  "were", "be",   "been", "to",   "of",
        "in",   "on",   "for",  "and",  "or",   "as",   "by",   "with", "from", "that", "this",
        "it",   "its",  "at",   "do",   "does", "did",  "not",  "no",   "yes",  "based",
        "context", "according", "source", "sources", "passage", "passages"};
    std::vector<std::string> content;
    for (const auto& t : atoks) {
        if (stop.count(t) || t.size() <= 1) {
            continue;
        }
        content.push_back(t);
    }
    if (content.empty()) {
        return 1.0;
    }
    auto ctx_toks = detail::simple_tokenize(context_blob(used));
    std::unordered_set<std::string> ctx(ctx_toks.begin(), ctx_toks.end());
    int hit = 0;
    for (const auto& t : content) {
        if (ctx.count(t)) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(content.size());
}

inline GroundingCheck validate_grounding(const std::string& answer,
                                         const std::vector<RetrievedChunk>& used,
                                         const GroundingConfig& cfg = {}) {
    GroundingCheck c;
    c.cited_ids = extract_citation_ids(answer);
    const bool refuse = is_dont_know(answer);
    c.refused = refuse;

    if (used.empty()) {
        if (!refuse) {
            c.ok = false;
            c.reason = "no relevant context but answer is not \"" + std::string(kDontKnowAnswer) +
                       "\"";
            return c;
        }
        if (!c.cited_ids.empty()) {
            c.ok = false;
            c.reason = "refusal must not cite chunk ids";
            return c;
        }
        c.ok = true;
        c.reason = "ok: refused with empty context";
        return c;
    }

    if (refuse) {
        c.ok = true;
        c.reason = "ok: explicit refusal despite context";
        return c;
    }

    const auto allow = allowed_ids(used);
    for (auto id : c.cited_ids) {
        if (!allow.count(id)) {
            c.illegal_ids.push_back(id);
        }
    }
    if (!c.illegal_ids.empty()) {
        c.ok = false;
        c.reason = "cited id not in retrieved context";
        return c;
    }
    if (cfg.require_citations && c.cited_ids.empty()) {
        c.ok = false;
        c.reason = "missing required [#id] citations";
        return c;
    }

    c.content_support = content_support_ratio(answer, used);
    if (c.content_support + 1e-9 < cfg.min_content_support) {
        c.ok = false;
        c.reason = "answer not supported by context tokens (support=" +
                   std::to_string(c.content_support) + ")";
        return c;
    }

    c.ok = true;
    c.reason = "ok: citations valid and content supported";
    return c;
}

inline std::string grounded_system_rules() {
    return "You are a retrieval-grounded assistant.\n"
           "RULES (mandatory):\n"
           "1) Use ONLY the context passages below. Do not use outside knowledge.\n"
           "2) If the context is empty or does not contain the answer, reply exactly: "
           "I don't know\n"
           "3) Every factual sentence MUST end with a citation of a context id like [#3].\n"
           "4) Only cite ids that appear in the context. Never invent ids.\n"
           "5) Prefer short answers. Do not add unstated details.\n";
}

inline std::string build_grounded_prompt(const std::string& question,
                                         const std::vector<RetrievedChunk>& used) {
    std::ostringstream oss;
    oss << grounded_system_rules() << "\n";
    oss << "Context:\n";
    if (used.empty()) {
        oss << "(no relevant passages — you must answer: I don't know)\n";
    } else {
        for (const auto& h : used) {
            oss << "[#" << h.id << " source=" << h.source << " score=" << h.score << "]\n";
            oss << h.text << "\n\n";
        }
    }
    oss << "Question: " << question << "\n\nAnswer:";
    return oss.str();
}

inline std::string build_extractive_answer(const std::vector<RetrievedChunk>& used) {
    if (used.empty()) {
        return kDontKnowAnswer;
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < used.size(); ++i) {
        if (i) {
            oss << ' ';
        }
        auto text = normalize_ws(used[i].text);
        oss << text;
        if (!text.empty() && text.back() != '.') {
            oss << '.';
        }
        oss << " [#" << used[i].id << "]";
    }
    return oss.str();
}

inline GroundedAnswer make_answer_from_used(const std::string& question,
                                            const std::vector<RetrievedChunk>& candidates,
                                            std::vector<RetrievedChunk> used,
                                            const GroundingConfig& cfg) {
    GroundedAnswer ga;
    ga.candidates = candidates;
    ga.used = std::move(used);
    ga.prompt = build_grounded_prompt(question, ga.used);
    if (ga.used.empty()) {
        ga.refused = true;
        ga.mode = "refuse";
        ga.answer = kDontKnowAnswer;
    } else {
        ga.refused = false;
        ga.mode = "extractive";
        ga.answer = build_extractive_answer(ga.used);
    }
    ga.check = validate_grounding(ga.answer, ga.used, cfg);
    return ga;
}

/// Score-only path (tests / no embedder).
inline GroundedAnswer answer_extractive_for_query(const std::string& question,
                                                  const std::vector<RetrievedChunk>& candidates,
                                                  const GroundingConfig& cfg = {}) {
    auto used = filter_relevant_by_score(candidates, cfg.min_score, cfg.max_context_chunks);
    return make_answer_from_used(question, candidates, std::move(used), cfg);
}

/// Preferred path with embedder for shuffle-baseline relevance.
inline GroundedAnswer answer_extractive_for_query(const std::string& question,
                                                  const std::vector<RetrievedChunk>& candidates,
                                                  const Embedder& embedder,
                                                  const GroundingConfig& cfg = {}) {
    auto used = filter_relevant(candidates, question, &embedder, cfg);
    return make_answer_from_used(question, candidates, std::move(used), cfg);
}

inline GroundedAnswer finalize_with_model_text(const std::string& question,
                                               const std::vector<RetrievedChunk>& candidates,
                                               const std::string& model_text,
                                               const Embedder* embedder,
                                               const GroundingConfig& cfg = {}) {
    std::vector<RetrievedChunk> used =
        embedder ? filter_relevant(candidates, question, embedder, cfg)
                 : filter_relevant_by_score(candidates, cfg.min_score, cfg.max_context_chunks);

    GroundedAnswer ga;
    ga.candidates = candidates;
    ga.used = std::move(used);
    ga.prompt = build_grounded_prompt(question, ga.used);

    if (ga.used.empty()) {
        ga.refused = true;
        ga.mode = "refuse";
        ga.answer = kDontKnowAnswer;
        ga.check = validate_grounding(ga.answer, ga.used, cfg);
        return ga;
    }

    auto check = validate_grounding(model_text, ga.used, cfg);
    if (check.ok) {
        ga.answer = normalize_ws(model_text);
        ga.mode = "generated";
        ga.refused = is_dont_know(ga.answer);
        ga.check = check;
        return ga;
    }

    ga.mode = "extractive_fallback";
    ga.answer = build_extractive_answer(ga.used);
    ga.refused = false;
    ga.check = validate_grounding(ga.answer, ga.used, cfg);
    ga.check.reason = std::string("model failed grounding (") + check.reason +
                      "); used extractive fallback";
    return ga;
}

// Back-compat overload without embedder.
inline GroundedAnswer finalize_with_model_text(const std::string& question,
                                               const std::vector<RetrievedChunk>& candidates,
                                               const std::string& model_text,
                                               const GroundingConfig& cfg = {}) {
    return finalize_with_model_text(question, candidates, model_text, nullptr, cfg);
}

}  // namespace nanorag
