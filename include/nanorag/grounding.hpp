#pragma once

// Grounding + citations for nanorag.
//
// Policy:
//   1) Retrieve k candidates (query–doc similarity)
//   2) Keep hits that pass answerability (lexical query support and/or high score;
//      NO_EVIDENCE sentinel forces refuse when it wins without answerable real docs)
//   3) Empty / unanswerable → exact "I don't know"
//   4) Otherwise extractive evidence passages with [#id] citations
//   5) validate_grounding: legal cites + answer content tokens ⊆ context
//
// Extractive mode quotes evidence; it is not free-form abstractive QA.
// Near-miss queries (alcohol BP when corpus only has water) must refuse.

#include "nanorag/embedder.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/text_util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanorag {

inline constexpr const char* kDontKnowAnswer = "I don't know";
/// Sentinel chunk id trained as the target for out-of-scope / unanswerable queries.
inline constexpr std::int64_t kNoEvidenceId = -1;
inline constexpr const char* kNoEvidenceText =
    "NO_EVIDENCE: the corpus contains no answer to this question.";

struct GroundingConfig {
    /// Minimum retrieval score (cosine-like) for a hit to be considered.
    float min_score = 0.25f;
    /// Without query↔passage lexical support, require this higher score (paraphrase path).
    float min_score_without_query_support = 0.55f;
    /// Min fraction of query content tokens found in the passage for lexical path.
    float min_query_support = 0.20f;
    /// Min absolute query-content hits for the lexical path.
    int min_query_support_hits = 1;
    std::size_t max_context_chunks = 3;
    bool require_citations = true;
    /// Min fraction of answer content tokens that must appear in context.
    float min_content_support = 0.25f;
};

/// Single source of defaults for CLI, smoke, eval, and library callers.
inline GroundingConfig default_grounding_config() {
    GroundingConfig c;
    c.min_score = 0.25f;
    c.min_score_without_query_support = 0.55f;
    c.min_query_support = 0.20f;
    c.min_query_support_hits = 1;
    c.max_context_chunks = 3;
    c.require_citations = true;
    c.min_content_support = 0.25f;
    return c;
}

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

inline bool is_blank_query(const std::string& query) {
    return normalize_ws(query).empty();
}

inline bool is_no_evidence_chunk(const RetrievedChunk& h) { return h.id == kNoEvidenceId; }

inline bool is_stop_token(const std::string& t) {
    static const std::unordered_set<std::string> stop = {
        "a",    "an",   "the",  "is",   "are",  "was",  "were", "be",   "been", "to",   "of",
        "in",   "on",   "for",  "and",  "or",   "as",   "by",   "with", "from", "that", "this",
        "it",   "its",  "at",   "do",   "does", "did",  "not",  "no",   "yes",  "what", "which",
        "who",  "whom", "whose","when", "where","why",  "how",  "many", "much", "can",  "could",
        "would","should","will","may",  "might","than", "then", "into", "about","over", "under",
        "between","my", "your", "our",  "their","his",  "her",  "there","here", "has",  "have",
        "had",  "point",  // generic in "boiling point" without the substance
        "based","context","according","source","sources","passage","passages",
    };
    return stop.count(t) != 0 || t.size() <= 1;
}

inline std::vector<std::string> content_tokens(const std::string& text) {
    std::vector<std::string> out;
    for (const auto& tok : detail::simple_tokenize(text)) {
        if (!is_stop_token(tok)) {
            out.push_back(tok);
        }
    }
    return out;
}

/// Fraction of query content tokens found in `passage`.
inline double query_support_in_passage(const std::string& query, const std::string& passage,
                                       int* hits_out = nullptr) {
    auto q = content_tokens(query);
    if (q.empty()) {
        if (hits_out) {
            *hits_out = 0;
        }
        return 0.0;
    }
    auto ptoks = content_tokens(passage);
    std::unordered_set<std::string> pset(ptoks.begin(), ptoks.end());
    int hits = 0;
    for (const auto& tok : q) {
        if (pset.count(tok)) {
            ++hits;
        }
    }
    if (hits_out) {
        *hits_out = hits;
    }
    return static_cast<double>(hits) / static_cast<double>(q.size());
}

/// Per-passage answerability (blocks near-misses like alcohol/ethanol→water).
inline bool passage_is_answerable(const std::string& query, const RetrievedChunk& h,
                                  const GroundingConfig& cfg) {
    if (is_no_evidence_chunk(h)) {
        return false;
    }
    if (h.score < cfg.min_score) {
        return false;
    }
    const auto qtoks = content_tokens(query);
    const auto ptoks = content_tokens(h.text);
    std::unordered_set<std::string> pset(ptoks.begin(), ptoks.end());
    int hits = 0;
    bool missing_distinctive = false;
    for (const auto& tok : qtoks) {
        if (pset.count(tok)) {
            ++hits;
        } else if (tok.size() >= 5) {
            // Distinctive query term absent from passage (e.g. ethanol, alcohol, iron).
            missing_distinctive = true;
        }
    }
    const double qsup =
        qtoks.empty() ? 0.0 : static_cast<double>(hits) / static_cast<double>(qtoks.size());

    // If a long content token is missing, shared generics like "atmosphere" must not
    // unlock the lexical path — only a strong embedding score may pass (paraphrase).
    if (missing_distinctive) {
        return h.score >= cfg.min_score_without_query_support;
    }
    if (hits >= cfg.min_query_support_hits && qsup + 1e-12 >= cfg.min_query_support) {
        return true;
    }
    return h.score >= cfg.min_score_without_query_support;
}

/// Refuse when NO_EVIDENCE is ranked first and no real answerable passage exists.
inline bool should_refuse_no_evidence(const std::vector<RetrievedChunk>& hits,
                                      const std::string& query, const GroundingConfig& cfg) {
    if (hits.empty() || !is_no_evidence_chunk(hits.front())) {
        return false;
    }
    for (std::size_t i = 1; i < hits.size(); ++i) {
        if (passage_is_answerable(query, hits[i], cfg)) {
            return false;
        }
    }
    return true;
}

inline std::vector<RetrievedChunk> filter_relevant(const std::vector<RetrievedChunk>& hits,
                                                   const std::string& query,
                                                   const Embedder* /*embedder*/,
                                                   const GroundingConfig& cfg) {
    if (is_blank_query(query)) {
        return {};
    }
    if (should_refuse_no_evidence(hits, query, cfg)) {
        return {};
    }
    std::vector<RetrievedChunk> out;
    for (const auto& h : hits) {
        if (!passage_is_answerable(query, h, cfg)) {
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
    try {
        std::sregex_iterator it(answer.begin(), answer.end(), re);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            ids.push_back(std::stoll((*it)[1].str()));
        }
    } catch (const std::exception&) {
        // malformed number → ignore that match
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
    // Strip the extractive prefix if present
    const std::string prefix = "evidence from retrieved passages:";
    auto lower = to_lower_copy(cleaned);
    if (lower.rfind(prefix, 0) == 0) {
        cleaned = cleaned.substr(prefix.size());
    }
    auto atoks = content_tokens(cleaned);
    if (atoks.empty()) {
        return 1.0;
    }
    auto ctx_toks = content_tokens(context_blob(used));
    std::unordered_set<std::string> ctx(ctx_toks.begin(), ctx_toks.end());
    int hit = 0;
    for (const auto& t : atoks) {
        if (ctx.count(t)) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(atoks.size());
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

/// Quote retrieved evidence with citations. Not abstractive QA — labeled as such.
inline std::string build_extractive_answer(const std::vector<RetrievedChunk>& used) {
    if (used.empty()) {
        return kDontKnowAnswer;
    }
    std::ostringstream oss;
    oss << "Evidence from retrieved passages:";
    for (const auto& h : used) {
        auto text = normalize_ws(h.text);
        oss << " ";
        oss << text;
        if (!text.empty() && text.back() != '.') {
            oss << '.';
        }
        oss << " [#" << h.id << "]";
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

inline GroundedAnswer answer_extractive_for_query(const std::string& question,
                                                  const std::vector<RetrievedChunk>& candidates,
                                                  const GroundingConfig& cfg = {}) {
    if (is_blank_query(question)) {
        return make_answer_from_used(question, candidates, {}, cfg);
    }
    auto used = filter_relevant(candidates, question, nullptr, cfg);
    return make_answer_from_used(question, candidates, std::move(used), cfg);
}

inline GroundedAnswer answer_extractive_for_query(const std::string& question,
                                                  const std::vector<RetrievedChunk>& candidates,
                                                  const Embedder& embedder,
                                                  const GroundingConfig& cfg = {}) {
    if (is_blank_query(question)) {
        return make_answer_from_used(question, candidates, {}, cfg);
    }
    auto used = filter_relevant(candidates, question, &embedder, cfg);
    return make_answer_from_used(question, candidates, std::move(used), cfg);
}

inline GroundedAnswer finalize_with_model_text(const std::string& question,
                                               const std::vector<RetrievedChunk>& candidates,
                                               const std::string& model_text,
                                               const Embedder* embedder,
                                               const GroundingConfig& cfg = {}) {
    if (is_blank_query(question)) {
        return make_answer_from_used(question, candidates, {}, cfg);
    }
    std::vector<RetrievedChunk> used = filter_relevant(candidates, question, embedder, cfg);

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

inline GroundedAnswer finalize_with_model_text(const std::string& question,
                                               const std::vector<RetrievedChunk>& candidates,
                                               const std::string& model_text,
                                               const GroundingConfig& cfg = {}) {
    return finalize_with_model_text(question, candidates, model_text, nullptr, cfg);
}

}  // namespace nanorag
