#pragma once

// Phase 2 evaluation foundation: labeled sets, metrics, ablations.
// Eval queries must be disjoint from training pairs (enforced by assert_disjoint).

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
    RetrievalMetrics retrieval;
};

struct EvalReport {
    RetrievalMetrics retrieval;
    RefuseMetrics refuse;
    GroundingMetrics grounding;
    std::vector<AblationRow> ablations;
    bool disjoint_ok = false;
};

struct EvalPaths {
    std::string chunks;
    std::string train_pairs;
    std::string retrieval;
    std::string refuse;
};

inline EvalPaths default_demo_paths(const std::string& root) {
    // root = .../data/demo
    EvalPaths p;
    p.chunks = root + "/chunks.tsv";
    p.train_pairs = root + "/train_pairs.tsv";
    p.retrieval = root + "/eval/retrieval.tsv";
    p.refuse = root + "/eval/refuse.tsv";
    return p;
}

inline Retriever build_contrastive_from_paths(const EvalPaths& paths,
                                              ContrastiveTrainConfig cfg = {},
                                              bool inject_no_evidence = true) {
    auto store = ChunkStore::load(paths.chunks);
    auto pairs = load_train_pairs(paths.train_pairs);
    cfg.dim = cfg.dim ? cfg.dim : 64;
    if (cfg.epochs <= 0) {
        cfg.epochs = 280;
    }
    return Retriever::build_contrastive(store, pairs, cfg, {}, inject_no_evidence);
}

inline EvalReport run_full_eval(const Retriever& primary, const EvalPaths& paths,
                                const GroundingConfig& gcfg = default_grounding_config(),
                                bool run_ablations = true) {
    auto train = load_train_query_set(paths.train_pairs);
    auto labeled = load_labeled(paths.retrieval);
    auto refuse = load_refuse(paths.refuse);
    assert_labeled_disjoint(train, labeled, "retrieval");
    assert_refuse_disjoint(train, refuse, "refuse");

    EvalReport rep;
    rep.disjoint_ok = true;
    rep.retrieval = eval_retrieval(primary, labeled, 5);
    rep.refuse = eval_refuse(primary, refuse, gcfg);
    rep.grounding = eval_grounding(primary, labeled, gcfg);

    if (run_ablations) {
        auto store = ChunkStore::load(paths.chunks);
        // Hashing
        {
            auto emb = std::make_shared<HashingEmbedder>(128);
            auto ret = Retriever::build(emb, store);
            AblationRow row;
            row.name = "hashing-v1";
            row.retrieval = eval_retrieval(ret, labeled, 5);
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
            row.retrieval = eval_retrieval(ret, labeled, 5);
            rep.ablations.push_back(std::move(row));
        }
        // Contrastive row (primary expected to be contrastive)
        {
            AblationRow row;
            row.name = "contrastive-v1";
            row.retrieval = rep.retrieval;
            rep.ablations.push_back(std::move(row));
        }
    }
    return rep;
}

inline std::string format_report(const EvalReport& r) {
    std::ostringstream oss;
    oss << "=== nanorag eval report ===\n";
    oss << "disjoint_ok=" << (r.disjoint_ok ? "true" : "false") << "\n";
    oss << "[retrieval] n=" << r.retrieval.n
        << " R@1=" << r.retrieval.recall_at_1
        << " R@3=" << r.retrieval.recall_at_3
        << " R@5=" << r.retrieval.recall_at_5
        << " MRR=" << r.retrieval.mrr << "\n";
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
        << " illegal|ans=" << r.grounding.illegal_cite_rate << "\n";
    if (!r.grounding.failures.empty()) {
        oss << "  grounding failures (" << r.grounding.failures.size() << "):\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(5, r.grounding.failures.size()); ++i) {
            oss << "    - " << r.grounding.failures[i] << "\n";
        }
    }
    if (!r.ablations.empty()) {
        oss << "[ablations]\n";
        for (const auto& a : r.ablations) {
            oss << "  " << a.name
                << " R@1=" << a.retrieval.recall_at_1
                << " R@5=" << a.retrieval.recall_at_5
                << " MRR=" << a.retrieval.mrr << "\n";
        }
    }
    return oss.str();
}

/// Gates for automated tests / CI (adjust as quality improves).
struct QualityGates {
    double min_contrastive_recall_at_1 = 0.85;
    double min_contrastive_mrr = 0.88;
    double min_refuse_pass_rate = 0.95;
    double min_grounding_full_pass = 0.85;
    /// Ablations should not dominate contrastive on R@1 by this margin (soft).
    bool require_contrastive_beats_hashing = true;
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
    if (r.retrieval.recall_at_1 + 1e-9 < g.min_contrastive_recall_at_1) {
        fail("R@1 " + std::to_string(r.retrieval.recall_at_1) + " < " +
             std::to_string(g.min_contrastive_recall_at_1));
    }
    if (r.retrieval.mrr + 1e-9 < g.min_contrastive_mrr) {
        fail("MRR " + std::to_string(r.retrieval.mrr) + " < " + std::to_string(g.min_contrastive_mrr));
    }
    if (r.refuse.pass_rate + 1e-9 < g.min_refuse_pass_rate) {
        fail("refuse pass_rate " + std::to_string(r.refuse.pass_rate) + " < " +
             std::to_string(g.min_refuse_pass_rate));
    }
    if (r.grounding.full_pass_rate + 1e-9 < g.min_grounding_full_pass) {
        fail("grounding full_pass " + std::to_string(r.grounding.full_pass_rate) + " < " +
             std::to_string(g.min_grounding_full_pass));
    }
    if (g.require_contrastive_beats_hashing) {
        double h_r1 = -1;
        double c_r1 = r.retrieval.recall_at_1;
        for (const auto& a : r.ablations) {
            if (a.name == "hashing-v1") {
                h_r1 = a.retrieval.recall_at_1;
            }
        }
        if (h_r1 >= 0 && c_r1 + 1e-9 < h_r1) {
            fail("contrastive R@1 should be >= hashing R@1");
        }
    }
    if (fails == 0) {
        return "";
    }
    return oss.str();
}

}  // namespace eval
}  // namespace nanorag
