#pragma once

// Phase 2 evaluation foundation: labeled sets, metrics, ablations.
// Eval queries must be disjoint from training pairs (enforced by assert_disjoint).
//
// Retrieval is reported on two held-out splits:
//   - easy (retrieval.tsv): paraphrases may share gold keywords — smoke / lexical path
//   - hard (retrieval_hard.tsv): ZERO content-token overlap with gold — honest semantic R@k/MRR
// Easy R@1≈1 does not measure dual-encoder quality; hard does (and may be low for BOW).

#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/text_util.hpp"
#include "nanorag/word2vec.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanorag {
namespace eval {

struct LabeledQuery {
    std::string query;
    std::int64_t gold_id = 0;
};

struct RefuseQuery {
    std::string query;
};

inline std::string normalize_query(std::string s) {
    // lowercase + collapse whitespace
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
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

inline std::unordered_set<std::string> load_train_query_set(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_train_query_set: cannot open " + path);
    }
    std::unordered_set<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto tab = line.find('\t');
        const std::string q = tab == std::string::npos ? line : line.substr(0, tab);
        out.insert(normalize_query(q));
    }
    return out;
}

inline std::vector<LabeledQuery> load_labeled(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_labeled: cannot open " + path);
    }
    std::vector<LabeledQuery> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("load_labeled: expected query\\tgold_id: " + line);
        }
        LabeledQuery q;
        q.query = line.substr(0, tab);
        q.gold_id = std::stoll(line.substr(tab + 1));
        if (q.gold_id == kNoEvidenceId) {
            throw std::runtime_error("load_labeled: gold must not be NO_EVIDENCE");
        }
        out.push_back(std::move(q));
    }
    if (out.empty()) {
        throw std::runtime_error("load_labeled: empty set " + path);
    }
    return out;
}

inline std::vector<RefuseQuery> load_refuse(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_refuse: cannot open " + path);
    }
    std::vector<RefuseQuery> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        out.push_back({line});
    }
    if (out.empty()) {
        throw std::runtime_error("load_refuse: empty set " + path);
    }
    return out;
}

/// Fail if any eval query (normalized) appears in train.
inline void assert_disjoint(const std::unordered_set<std::string>& train,
                            const std::vector<std::string>& eval_queries,
                            const std::string& eval_name) {
    std::vector<std::string> bad;
    for (const auto& q : eval_queries) {
        if (train.count(normalize_query(q))) {
            bad.push_back(q);
        }
    }
    if (!bad.empty()) {
        std::ostringstream oss;
        oss << "eval/train query overlap in " << eval_name << " (" << bad.size() << "):\n";
        for (const auto& q : bad) {
            oss << "  - " << q << "\n";
        }
        throw std::runtime_error(oss.str());
    }
}

inline void assert_labeled_disjoint(const std::unordered_set<std::string>& train,
                                    const std::vector<LabeledQuery>& labeled,
                                    const std::string& name) {
    std::vector<std::string> qs;
    qs.reserve(labeled.size());
    for (const auto& q : labeled) {
        qs.push_back(q.query);
    }
    assert_disjoint(train, qs, name);
}

inline void assert_refuse_disjoint(const std::unordered_set<std::string>& train,
                                   const std::vector<RefuseQuery>& refuse,
                                   const std::string& name) {
    std::vector<std::string> qs;
    qs.reserve(refuse.size());
    for (const auto& q : refuse) {
        qs.push_back(q.query);
    }
    assert_disjoint(train, qs, name);
}

/// Content tokens shared by query and gold passage (same filter as answerability).
inline std::vector<std::string> keyword_overlap(const std::string& query,
                                                const std::string& passage) {
    const auto q = content_tokens(query);
    const auto p = content_tokens(passage);
    std::unordered_set<std::string> pset(p.begin(), p.end());
    std::vector<std::string> inter;
    std::unordered_set<std::string> seen;
    for (const auto& t : q) {
        if (pset.count(t) && seen.insert(t).second) {
            inter.push_back(t);
        }
    }
    return inter;
}

inline bool has_zero_keyword_overlap(const std::string& query, const std::string& passage) {
    return keyword_overlap(query, passage).empty();
}

/// Fail unless every labeled query has zero content-token overlap with its gold chunk.
inline void assert_zero_keyword_overlap(const ChunkStore& store,
                                        const std::vector<LabeledQuery>& labeled,
                                        const std::string& name) {
    std::ostringstream oss;
    int bad = 0;
    for (const auto& q : labeled) {
        const auto& gold = store.get(q.gold_id);
        auto ov = keyword_overlap(q.query, gold.text);
        if (!ov.empty()) {
            ++bad;
            oss << "  - id=" << q.gold_id << " overlap={";
            for (std::size_t i = 0; i < ov.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << ov[i];
            }
            oss << "} | " << q.query << "\n";
        }
    }
    if (bad) {
        throw std::runtime_error(name + ": " + std::to_string(bad) +
                                 " queries share content tokens with gold:\n" + oss.str());
    }
}

/// Character n-gram (3/4) set over content tokens — catches morph leakage (feline/felis).
inline std::unordered_set<std::string> content_char_ngrams(const std::string& text) {
    std::unordered_set<std::string> out;
    for (const auto& tok : content_tokens(text)) {
        const std::string s = std::string("^") + tok + "$";
        for (int n : {3, 4}) {
            if (static_cast<int>(s.size()) < n) {
                continue;
            }
            for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= s.size(); ++i) {
                out.insert(s.substr(i, static_cast<std::size_t>(n)));
            }
        }
    }
    return out;
}

/// Fraction of query content-char-ngrams that also appear in passage.
inline double ngram_query_coverage(const std::string& query, const std::string& passage) {
    const auto q = content_char_ngrams(query);
    if (q.empty()) {
        return 0.0;
    }
    const auto p = content_char_ngrams(passage);
    int hit = 0;
    for (const auto& g : q) {
        if (p.count(g)) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(q.size());
}

inline double content_token_jaccard(const std::string& a, const std::string& b) {
    const auto ta = content_tokens(a);
    const auto tb = content_tokens(b);
    std::unordered_set<std::string> sa(ta.begin(), ta.end());
    std::unordered_set<std::string> sb(tb.begin(), tb.end());
    if (sa.empty() && sb.empty()) {
        return 1.0;
    }
    if (sa.empty() || sb.empty()) {
        return 0.0;
    }
    std::size_t inter = 0;
    for (const auto& t : sa) {
        if (sb.count(t)) {
            ++inter;
        }
    }
    return static_cast<double>(inter) / static_cast<double>(sa.size() + sb.size() - inter);
}

/// Near-dup score on **content-token** sequences (not char LCS).
/// Char LCS falsely flags unrelated "Which … ?" templates as similar.
inline double sequence_similarity(const std::string& a_in, const std::string& b_in) {
    const auto a = content_tokens(a_in);
    const auto b = content_tokens(b_in);
    if (a.empty() && b.empty()) {
        return 1.0;
    }
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::size_t> prev(m + 1, 0), cur(m + 1, 0);
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = std::max(prev[j], cur[j - 1]);
            }
        }
        prev.swap(cur);
        std::fill(cur.begin(), cur.end(), 0);
    }
    const double lcs = static_cast<double>(prev[m]);
    return 2.0 * lcs / static_cast<double>(n + m);
}

struct HardIntegrityLimits {
    double max_train_seq_sim = 0.50;
    double max_train_content_jaccard = 0.25;
    double max_gold_ngram_coverage = 0.15;
    int min_distinctive_token_len = 4;
};

/// Fail if hard queries leak via train near-duplicates / shared synonym tokens / ngram morph.
inline void assert_hard_no_synonym_leak(const ChunkStore& store,
                                        const std::vector<LabeledQuery>& hard,
                                        const std::vector<TrainPair>& train_pairs,
                                        const std::string& name,
                                        HardIntegrityLimits lim = {}) {
    std::ostringstream oss;
    int bad = 0;
    for (const auto& hq : hard) {
        const auto& gold = store.get(hq.gold_id);
        std::vector<std::string> reasons;

        if (!keyword_overlap(hq.query, gold.text).empty()) {
            reasons.push_back("gold content-token overlap");
        }
        const double ncov = ngram_query_coverage(hq.query, gold.text);
        if (ncov > lim.max_gold_ngram_coverage + 1e-12) {
            reasons.push_back("gold ngram_qcov=" + std::to_string(ncov));
        }

        double best_seq = 0, best_j = 0;
        std::string best_seq_q, best_j_q;
        std::unordered_set<std::string> same_id_train_toks;
        for (const auto& tp : train_pairs) {
            const double s = sequence_similarity(hq.query, tp.query);
            const double j = content_token_jaccard(hq.query, tp.query);
            if (s > best_seq) {
                best_seq = s;
                best_seq_q = tp.query;
            }
            if (j > best_j) {
                best_j = j;
                best_j_q = tp.query;
            }
            if (tp.pos_id == hq.gold_id) {
                for (const auto& t : content_tokens(tp.query)) {
                    if (static_cast<int>(t.size()) >= lim.min_distinctive_token_len) {
                        same_id_train_toks.insert(t);
                    }
                }
            }
        }
        if (best_seq + 1e-12 >= lim.max_train_seq_sim) {
            reasons.push_back("train seq_sim=" + std::to_string(best_seq) + " ~ " + best_seq_q);
        }
        if (best_j + 1e-12 >= lim.max_train_content_jaccard) {
            reasons.push_back("train content_jaccard=" + std::to_string(best_j) + " ~ " +
                              best_j_q);
        }
        std::vector<std::string> shared;
        for (const auto& t : content_tokens(hq.query)) {
            if (static_cast<int>(t.size()) >= lim.min_distinctive_token_len &&
                same_id_train_toks.count(t)) {
                shared.push_back(t);
            }
        }
        if (!shared.empty()) {
            std::ostringstream s;
            s << "same-id train distinctive tokens={";
            for (std::size_t i = 0; i < shared.size(); ++i) {
                if (i) {
                    s << ",";
                }
                s << shared[i];
            }
            s << "}";
            reasons.push_back(s.str());
        }

        if (!reasons.empty()) {
            ++bad;
            oss << "  - id=" << hq.gold_id << " | " << hq.query << "\n";
            for (const auto& r : reasons) {
                oss << "      * " << r << "\n";
            }
        }
    }
    if (bad) {
        throw std::runtime_error(name + ": " + std::to_string(bad) +
                                 " hard queries fail synonym-leak integrity:\n" + oss.str());
    }
}

// ---------------------------------------------------------------------------
// Retrieval metrics (real docs only — skip NO_EVIDENCE)
// ---------------------------------------------------------------------------

inline std::vector<std::int64_t> ranked_real_ids(const Retriever& ret, const std::string& query,
                                                 std::size_t k) {
    auto r = ret.retrieve(query, k + 4);  // overfetch if sentinel present
    std::vector<std::int64_t> ids;
    for (const auto& h : r.chunks) {
        if (h.id == kNoEvidenceId) {
            continue;
        }
        ids.push_back(h.id);
        if (ids.size() >= k) {
            break;
        }
    }
    return ids;
}

inline bool recall_at_k(const std::vector<std::int64_t>& ranked, std::int64_t gold, std::size_t k) {
    const std::size_t n = std::min(k, ranked.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (ranked[i] == gold) {
            return true;
        }
    }
    return false;
}

/// Reciprocal rank of gold (0 if missing). Rank is 1-based.
inline double reciprocal_rank(const std::vector<std::int64_t>& ranked, std::int64_t gold) {
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        if (ranked[i] == gold) {
            return 1.0 / static_cast<double>(i + 1);
        }
    }
    return 0.0;
}

struct RetrievalMetrics {
    double recall_at_1 = 0;
    double recall_at_3 = 0;
    double recall_at_5 = 0;
    double mrr = 0;
    int n = 0;
};

inline RetrievalMetrics eval_retrieval(const Retriever& ret,
                                       const std::vector<LabeledQuery>& labeled,
                                       std::size_t max_k = 5) {
    RetrievalMetrics m;
    m.n = static_cast<int>(labeled.size());
    if (m.n == 0) {
        return m;
    }
    double r1 = 0, r3 = 0, r5 = 0, mrr = 0;
    for (const auto& q : labeled) {
        auto ranked = ranked_real_ids(ret, q.query, max_k);
        if (recall_at_k(ranked, q.gold_id, 1)) {
            r1 += 1;
        }
        if (recall_at_k(ranked, q.gold_id, 3)) {
            r3 += 1;
        }
        if (recall_at_k(ranked, q.gold_id, 5)) {
            r5 += 1;
        }
        mrr += reciprocal_rank(ranked, q.gold_id);
    }
    const double inv = 1.0 / static_cast<double>(m.n);
    m.recall_at_1 = r1 * inv;
    m.recall_at_3 = r3 * inv;
    m.recall_at_5 = r5 * inv;
    m.mrr = mrr * inv;
    return m;
}

// ---------------------------------------------------------------------------
// Refuse metrics
// ---------------------------------------------------------------------------

struct RefuseMetrics {
    double idk_rate = 0;       // exact I don't know
    double empty_used_rate = 0;
    double pass_rate = 0;      // IDK + empty used + validator ok
    int n = 0;
    std::vector<std::string> failures;
};

inline RefuseMetrics eval_refuse(const Retriever& ret, const std::vector<RefuseQuery>& refuse,
                                 const GroundingConfig& gcfg = default_grounding_config()) {
    RefuseMetrics m;
    m.n = static_cast<int>(refuse.size());
    if (m.n == 0) {
        return m;
    }
    int idk = 0, empty = 0, pass = 0;
    for (const auto& q : refuse) {
        auto ga = ret.ask_grounded(q.query, 5, gcfg);
        const bool is_idk = ga.answer == kDontKnowAnswer || is_dont_know(ga.answer);
        const bool is_empty = ga.used.empty();
        const bool ok = is_idk && is_empty && ga.refused && ga.check.ok;
        if (is_idk) {
            ++idk;
        }
        if (is_empty) {
            ++empty;
        }
        if (ok) {
            ++pass;
        } else {
            m.failures.push_back(q.query + " | answer=" + ga.answer +
                                 " used=" + std::to_string(ga.used.size()));
        }
    }
    const double inv = 1.0 / static_cast<double>(m.n);
    m.idk_rate = idk * inv;
    m.empty_used_rate = empty * inv;
    m.pass_rate = pass * inv;
    return m;
}

// ---------------------------------------------------------------------------
// Grounding metrics (on retrieval-labeled queries that should answer)
// ---------------------------------------------------------------------------

struct GroundingMetrics {
    double answer_rate = 0;          // not refused
    double validator_ok_rate = 0;    // check.ok among answered
    double gold_cited_rate = 0;      // gold id appears in citations among answered
    double illegal_cite_rate = 0;    // any illegal id among answered
    double full_pass_rate = 0;       // answered + ok + gold cited + no illegal
    int n = 0;
    int answered = 0;
    std::vector<std::string> failures;
};

inline GroundingMetrics eval_grounding(const Retriever& ret,
                                       const std::vector<LabeledQuery>& labeled,
                                       const GroundingConfig& gcfg = default_grounding_config()) {
    GroundingMetrics m;
    m.n = static_cast<int>(labeled.size());
    if (m.n == 0) {
        return m;
    }
    int answered = 0, vok = 0, gold_c = 0, illegal = 0, full = 0;
    for (const auto& q : labeled) {
        auto ga = ret.ask_grounded(q.query, 5, gcfg);
        if (ga.refused) {
            m.failures.push_back("refused: " + q.query);
            continue;
        }
        ++answered;
        if (ga.check.ok) {
            ++vok;
        }
        if (!ga.check.illegal_ids.empty()) {
            ++illegal;
        }
        const auto cites = extract_citation_ids(ga.answer);
        const bool has_gold =
            std::find(cites.begin(), cites.end(), q.gold_id) != cites.end();
        if (has_gold) {
            ++gold_c;
        }
        if (ga.check.ok && has_gold && ga.check.illegal_ids.empty()) {
            ++full;
        } else {
            m.failures.push_back("grounding: " + q.query + " | " + ga.check.reason);
        }
    }
    m.answered = answered;
    const double inv_n = 1.0 / static_cast<double>(m.n);
    m.answer_rate = answered * inv_n;
    m.full_pass_rate = full * inv_n;
    if (answered > 0) {
        const double inv_a = 1.0 / static_cast<double>(answered);
        m.validator_ok_rate = vok * inv_a;
        m.gold_cited_rate = gold_c * inv_a;
        m.illegal_cite_rate = illegal * inv_a;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Full report + ablations
// ---------------------------------------------------------------------------

struct AblationRow {
    std::string name;
    RetrievalMetrics retrieval_easy;
    RetrievalMetrics retrieval_hard;
};

struct EvalReport {
    /// Lexical-friendly held-out paraphrases (may share gold keywords). Smoke only.
    RetrievalMetrics retrieval_easy;
    /// Zero content-token overlap with gold — honest semantic retrieval measure.
    RetrievalMetrics retrieval_hard;
    RefuseMetrics refuse;
    GroundingMetrics grounding;
    std::vector<AblationRow> ablations;
    bool disjoint_ok = false;
    bool hard_zero_keyword_ok = false;
};

struct EvalPaths {
    std::string chunks;
    std::string train_pairs;
    std::string retrieval;       // easy
    std::string retrieval_hard;  // zero keyword overlap
    std::string refuse;
};

inline EvalPaths default_demo_paths(const std::string& root) {
    // root = .../data/demo
    EvalPaths p;
    p.chunks = root + "/chunks.tsv";
    p.train_pairs = root + "/train_pairs.tsv";
    p.retrieval = root + "/eval/retrieval.tsv";
    p.retrieval_hard = root + "/eval/retrieval_hard.tsv";
    p.refuse = root + "/eval/refuse.tsv";
    return p;
}

inline Retriever build_contrastive_from_paths(const EvalPaths& paths,
                                              ContrastiveTrainConfig cfg = {},
                                              bool inject_no_evidence = true) {
    auto store = ChunkStore::load(paths.chunks);
    auto pairs = load_train_pairs(paths.train_pairs);
    // contrastive-v2 + hard-neg training defaults
    if (cfg.dim == 0) {
        cfg.dim = 128;
    }
    if (cfg.epochs <= 0) {
        cfg.epochs = 420;
    }
    return Retriever::build_contrastive(store, pairs, cfg, {}, inject_no_evidence);
}

inline EvalReport run_full_eval(const Retriever& primary, const EvalPaths& paths,
                                const GroundingConfig& gcfg = default_grounding_config(),
                                bool run_ablations = true) {
    auto train = load_train_query_set(paths.train_pairs);
    auto easy = load_labeled(paths.retrieval);
    auto hard = load_labeled(paths.retrieval_hard);
    auto refuse = load_refuse(paths.refuse);
    assert_labeled_disjoint(train, easy, "retrieval_easy");
    assert_labeled_disjoint(train, hard, "retrieval_hard");
    assert_refuse_disjoint(train, refuse, "refuse");
    // Hard must also be disjoint from easy query strings (no double-counting).
    {
        std::unordered_set<std::string> easy_set;
        for (const auto& q : easy) {
            easy_set.insert(normalize_query(q.query));
        }
        std::vector<std::string> hard_qs;
        for (const auto& q : hard) {
            hard_qs.push_back(q.query);
        }
        assert_disjoint(easy_set, hard_qs, "retrieval_hard vs retrieval_easy");
    }

    auto store = ChunkStore::load(paths.chunks);
    assert_zero_keyword_overlap(store, hard, "retrieval_hard");
    // Hard must not be solvable via train near-dups / shared synonym tokens / morph ngrams.
    auto train_pairs = load_train_pairs(paths.train_pairs);
    assert_hard_no_synonym_leak(store, hard, train_pairs, "retrieval_hard");

    EvalReport rep;
    rep.disjoint_ok = true;
    rep.hard_zero_keyword_ok = true;
    rep.retrieval_easy = eval_retrieval(primary, easy, 5);
    rep.retrieval_hard = eval_retrieval(primary, hard, 5);
    rep.refuse = eval_refuse(primary, refuse, gcfg);
    // Grounding uses easy set (lexical support path); hard is retrieval-only for now.
    rep.grounding = eval_grounding(primary, easy, gcfg);

    if (run_ablations) {
        // Hashing
        {
            auto emb = std::make_shared<HashingEmbedder>(128);
            auto ret = Retriever::build(emb, store);
            AblationRow row;
            row.name = "hashing-v1";
            row.retrieval_easy = eval_retrieval(ret, easy, 5);
            row.retrieval_hard = eval_retrieval(ret, hard, 5);
            rep.ablations.push_back(std::move(row));
        }
        // Word2Vec
        {
            Word2VecTrainConfig wcfg;
            wcfg.dim = 64;
            wcfg.epochs = 80;
            wcfg.doc_repeat = 12;
            wcfg.seed = 42;
            auto ret = Retriever::build_word2vec(store, wcfg);
            AblationRow row;
            row.name = "word2vec-v1";
            row.retrieval_easy = eval_retrieval(ret, easy, 5);
            row.retrieval_hard = eval_retrieval(ret, hard, 5);
            rep.ablations.push_back(std::move(row));
        }
        // Contrastive row (primary expected to be contrastive-v2)
        {
            AblationRow row;
            row.name = primary.embedder().id();
            row.retrieval_easy = rep.retrieval_easy;
            row.retrieval_hard = rep.retrieval_hard;
            rep.ablations.push_back(std::move(row));
        }
    }
    return rep;
}

inline std::string format_report(const EvalReport& r) {
    std::ostringstream oss;
    oss << "=== nanorag eval report ===\n";
    oss << "disjoint_ok=" << (r.disjoint_ok ? "true" : "false")
        << " hard_zero_keyword_ok=" << (r.hard_zero_keyword_ok ? "true" : "false") << "\n";
    oss << "[retrieval_easy] n=" << r.retrieval_easy.n
        << " R@1=" << r.retrieval_easy.recall_at_1
        << " R@3=" << r.retrieval_easy.recall_at_3
        << " R@5=" << r.retrieval_easy.recall_at_5
        << " MRR=" << r.retrieval_easy.mrr
        << "  # may share gold keywords — NOT semantic quality\n";
    oss << "[retrieval_hard] n=" << r.retrieval_hard.n
        << " R@1=" << r.retrieval_hard.recall_at_1
        << " R@3=" << r.retrieval_hard.recall_at_3
        << " R@5=" << r.retrieval_hard.recall_at_5
        << " MRR=" << r.retrieval_hard.mrr
        << "  # zero content-token overlap with gold — honest R@k/MRR\n";
    oss << "[refuse] n=" << r.refuse.n
        << " idk_rate=" << r.refuse.idk_rate
        << " empty_used=" << r.refuse.empty_used_rate
        << " pass_rate=" << r.refuse.pass_rate << "\n";
    if (!r.refuse.failures.empty()) {
        oss << "  refuse failures (" << r.refuse.failures.size() << "):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(5, r.refuse.failures.size()); ++i) {
            oss << "    - " << r.refuse.failures[i] << "\n";
        }
    }
    oss << "[grounding] n=" << r.grounding.n
        << " answer_rate=" << r.grounding.answer_rate
        << " full_pass=" << r.grounding.full_pass_rate
        << " gold_cited|ans=" << r.grounding.gold_cited_rate
        << " validator_ok|ans=" << r.grounding.validator_ok_rate
        << " illegal|ans=" << r.grounding.illegal_cite_rate
        << "  # on retrieval_easy labels\n";
    if (!r.grounding.failures.empty()) {
        oss << "  grounding failures (" << r.grounding.failures.size() << "):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(5, r.grounding.failures.size()); ++i) {
            oss << "    - " << r.grounding.failures[i] << "\n";
        }
    }
    if (!r.ablations.empty()) {
        oss << "[ablations] easy_R@1 / hard_R@1 / hard_MRR\n";
        for (const auto& a : r.ablations) {
            oss << "  " << a.name
                << " easy_R@1=" << a.retrieval_easy.recall_at_1
                << " hard_R@1=" << a.retrieval_hard.recall_at_1
                << " hard_R@5=" << a.retrieval_hard.recall_at_5
                << " hard_MRR=" << a.retrieval_hard.mrr << "\n";
        }
    }
    return oss.str();
}

/// Gates for automated tests / CI.
/// High R@k floors apply only to retrieval_easy (lexical smoke).
/// Hard metrics are reported for honesty; optional floors default to 0 (BOW may fail hard).
struct QualityGates {
    double min_easy_recall_at_1 = 0.85;
    double min_easy_mrr = 0.88;
    /// Honest semantic split — leave 0 until embedders improve; still reported.
    double min_hard_recall_at_1 = 0.0;
    double min_hard_mrr = 0.0;
    double min_refuse_pass_rate = 0.90;
    double min_grounding_full_pass = 0.85;
    bool require_hard_zero_keyword = true;
    /// On easy set only (hashing is competitive when keywords leak).
    bool require_contrastive_beats_hashing_easy = true;
};

inline std::string check_gates(const EvalReport& r, const QualityGates& g = {}) {
    std::ostringstream oss;
    int fails = 0;
    auto fail = [&](const std::string& msg) {
        oss << "GATE FAIL: " << msg << "\n";
        ++fails;
    };
    if (!r.disjoint_ok) {
        fail("train/eval not disjoint");
    }
    if (g.require_hard_zero_keyword && !r.hard_zero_keyword_ok) {
        fail("retrieval_hard must have zero content-token overlap with gold");
    }
    if (r.retrieval_hard.n <= 0) {
        fail("retrieval_hard set is empty");
    }
    if (r.retrieval_easy.recall_at_1 + 1e-9 < g.min_easy_recall_at_1) {
        fail("easy R@1 " + std::to_string(r.retrieval_easy.recall_at_1) + " < " +
             std::to_string(g.min_easy_recall_at_1));
    }
    if (r.retrieval_easy.mrr + 1e-9 < g.min_easy_mrr) {
        fail("easy MRR " + std::to_string(r.retrieval_easy.mrr) + " < " +
             std::to_string(g.min_easy_mrr));
    }
    if (r.retrieval_hard.recall_at_1 + 1e-9 < g.min_hard_recall_at_1) {
        fail("hard R@1 " + std::to_string(r.retrieval_hard.recall_at_1) + " < " +
             std::to_string(g.min_hard_recall_at_1));
    }
    if (r.retrieval_hard.mrr + 1e-9 < g.min_hard_mrr) {
        fail("hard MRR " + std::to_string(r.retrieval_hard.mrr) + " < " +
             std::to_string(g.min_hard_mrr));
    }
    if (r.refuse.pass_rate + 1e-9 < g.min_refuse_pass_rate) {
        fail("refuse pass_rate " + std::to_string(r.refuse.pass_rate) + " < " +
             std::to_string(g.min_refuse_pass_rate));
    }
    if (r.grounding.full_pass_rate + 1e-9 < g.min_grounding_full_pass) {
        fail("grounding full_pass " + std::to_string(r.grounding.full_pass_rate) + " < " +
             std::to_string(g.min_grounding_full_pass));
    }
    if (g.require_contrastive_beats_hashing_easy) {
        double h_r1 = -1;
        double c_r1 = r.retrieval_easy.recall_at_1;
        for (const auto& a : r.ablations) {
            if (a.name == "hashing-v1") {
                h_r1 = a.retrieval_easy.recall_at_1;
            }
        }
        if (h_r1 >= 0 && c_r1 + 1e-9 < h_r1) {
            fail("contrastive easy R@1 should be >= hashing easy R@1");
        }
    }
    if (fails == 0) {
        return "";
    }
    return oss.str();
}

}  // namespace eval
}  // namespace nanorag
