#include "nanorag/eval.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_fails = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n";  \
            ++g_fails;                                                              \
        }                                                                           \
    } while (0)

static std::string demo_root() {
    if (const char* env = std::getenv("NANORAG_DEMO_DATA")) {
        return env;
    }
    // Prefer source-tree data next to binary's relative path via cwd (ctest WORKING_DIRECTORY)
    const fs::path p = fs::path("data/demo");
    if (fs::exists(p / "train_pairs.tsv") && fs::exists(p / "eval/retrieval.tsv")) {
        return p.string();
    }
    // Fallback: relative to this source file is not available; try common locations
    for (const auto& cand : {fs::path("../data/demo"), fs::path("../../data/demo")}) {
        if (fs::exists(cand / "train_pairs.tsv")) {
            return cand.string();
        }
    }
    throw std::runtime_error("cannot find data/demo (set NANORAG_DEMO_DATA or run from repo root)");
}

int main() {
    using namespace nanorag;
    using namespace nanorag::eval;

    // --- pure metric unit tests ---
    {
        std::vector<std::int64_t> ranked = {3, 1, 7, 0};
        CHECK(recall_at_k(ranked, 3, 1));
        CHECK(recall_at_k(ranked, 1, 2));
        CHECK(!recall_at_k(ranked, 0, 2));
        CHECK(recall_at_k(ranked, 0, 4));
        CHECK(std::abs(reciprocal_rank(ranked, 3) - 1.0) < 1e-9);
        CHECK(std::abs(reciprocal_rank(ranked, 1) - 0.5) < 1e-9);
        CHECK(std::abs(reciprocal_rank(ranked, 7) - 1.0 / 3.0) < 1e-9);
        CHECK(reciprocal_rank(ranked, 99) == 0.0);
    }

    {
        CHECK(normalize_query("  Hello   World  ") == "hello world");
        auto train = std::unordered_set<std::string>{normalize_query("Alpha Beta")};
        try {
            assert_disjoint(train, {"alpha beta"}, "unit");
            std::cerr << "FAIL expected overlap throw\n";
            ++g_fails;
        } catch (const std::exception&) {
            // expected
        }
        assert_disjoint(train, {"gamma delta"}, "unit-ok");
    }

    // --- hard grounding unit cases (no training) ---
    {
        GroundingConfig cfg = default_grounding_config();
        std::vector<RetrievedChunk> cand = {
            {10, 0.40f, "animals",
             "Felis catus is a small carnivorous mammal that shares dwellings with people."},
            {30, 0.10f, "science",
             "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K."},
        };
        auto alcohol =
            answer_extractive_for_query("What is the boiling point of alcohol?", cand, cfg);
        CHECK(alcohol.refused);
        CHECK(alcohol.answer == std::string(kDontKnowAnswer));

        auto iron = answer_extractive_for_query("What is the melting point of iron?", cand, cfg);
        CHECK(iron.refused);

        auto mercury =
            answer_extractive_for_query("What is the freezing point of mercury?", cand, cfg);
        CHECK(mercury.refused);

        // Entity swap: water BP query with only cat evidence → refuse
        auto wrong =
            answer_extractive_for_query("At one atmosphere when does pure water become gas?", cand, cfg);
        CHECK(wrong.refused);

        // Lexical support path
        auto mammal = answer_extractive_for_query("Which mammal lives with people?", cand, cfg);
        CHECK(!mammal.refused);
        auto cites = extract_citation_ids(mammal.answer);
        CHECK(std::find(cites.begin(), cites.end(), 10) != cites.end());

        // Citation validity
        std::vector<RetrievedChunk> used = {cand[0]};
        CHECK(!validate_grounding("Aliens [#99]", used, cfg).ok);
        CHECK(!validate_grounding("Felis catus is a small carnivorous mammal.", used, cfg).ok);
        CHECK(validate_grounding("Felis catus is a small carnivorous mammal. [#10]", used, cfg).ok);
        CHECK(validate_grounding(kDontKnowAnswer, {}, cfg).ok);
        CHECK(!validate_grounding("Something invented.", {}, cfg).ok);
    }

    // --- full labeled harness on demo data ---
    {
        const std::string root = demo_root();
        auto paths = default_demo_paths(root);

        // Disjointness must hold on disk
        auto train = load_train_query_set(paths.train_pairs);
        auto easy = load_labeled(paths.retrieval);
        auto hard = load_labeled(paths.retrieval_hard);
        auto refuse = load_refuse(paths.refuse);
        assert_labeled_disjoint(train, easy, "retrieval_easy");
        assert_labeled_disjoint(train, hard, "retrieval_hard");
        assert_refuse_disjoint(train, refuse, "refuse");
        CHECK(easy.size() >= 12);
        CHECK(hard.size() >= 12);
        CHECK(refuse.size() >= 10);

        auto store = ChunkStore::load(paths.chunks);

        // Easy set: soft Jaccard (may share keywords; not a semantic measure)
        for (const auto& q : easy) {
            const double j = token_jaccard(q.query, store.get(q.gold_id).text);
            if (j > 0.45) {
                std::cerr << "FAIL easy high jaccard j=" << j << " q=" << q.query << "\n";
                ++g_fails;
            }
        }

        // Hard set: must have ZERO content-token overlap with gold
        try {
            assert_zero_keyword_overlap(store, hard, "unit-hard");
        } catch (const std::exception& e) {
            std::cerr << "FAIL hard zero-keyword: " << e.what() << "\n";
            ++g_fails;
        }
        for (const auto& q : hard) {
            if (!has_zero_keyword_overlap(q.query, store.get(q.gold_id).text)) {
                std::cerr << "FAIL hard overlap: " << q.query << "\n";
                ++g_fails;
            }
        }

        // Unit: keyword_overlap detects leaks
        {
            auto ov = keyword_overlap("pure H2O liquid gas", store.get(6).text);
            CHECK(!ov.empty());
            CHECK(has_zero_keyword_overlap("dihydrogen monoxide freeze weather", store.get(6).text));
        }

        ContrastiveTrainConfig cfg;
        cfg.dim = 64;
        cfg.epochs = 320;
        cfg.lr = 0.09f;
        cfg.temperature = 0.05f;
        cfg.seed = 7;
        auto primary = build_contrastive_from_paths(paths, cfg, /*inject_no_evidence=*/true);

        auto report = run_full_eval(primary, paths, default_grounding_config(),
                                    /*run_ablations=*/true);
        std::cout << format_report(report);

        CHECK(report.hard_zero_keyword_ok);
        CHECK(report.retrieval_hard.n >= 12);
        // Easy smoke can look strong; hard is the honest signal (BOW may be low)
        CHECK(report.retrieval_easy.recall_at_1 + 1e-9 >= 0.80);

        // Hard refuse cases must be in the failure-free set
        for (const char* hard_q : {"What is the boiling point of alcohol?",
                                   "What is the melting point of iron?",
                                   "What is the capital of Germany?"}) {
            auto ga = primary.ask_grounded(hard_q, 5, default_grounding_config());
            if (!ga.refused || ga.answer != std::string(kDontKnowAnswer)) {
                std::cerr << "HARD refuse FAIL: " << hard_q << " -> " << ga.answer << "\n";
                ++g_fails;
            }
        }

        // Gates: high floors on easy smoke only; hard floors stay 0 for BOW
        QualityGates gates;
        gates.min_easy_recall_at_1 = 0.80;
        gates.min_easy_mrr = 0.82;
        gates.min_hard_recall_at_1 = 0.0;
        gates.min_hard_mrr = 0.0;
        gates.min_refuse_pass_rate = 0.90;
        gates.min_grounding_full_pass = 0.75;
        const std::string gate_err = check_gates(report, gates);
        if (!gate_err.empty()) {
            std::cerr << gate_err;
            ++g_fails;
        }

        double hash_easy = -1, hash_hard = -1;
        for (const auto& a : report.ablations) {
            if (a.name == "hashing-v1") {
                hash_easy = a.retrieval_easy.recall_at_1;
                hash_hard = a.retrieval_hard.recall_at_1;
            }
        }
        CHECK(hash_easy >= 0);
        CHECK(hash_hard >= 0);
        CHECK(report.ablations.size() >= 3);
        // Document honesty: if hard R@1 is perfect, something is likely leaking keywords
        if (report.retrieval_hard.recall_at_1 > 0.99 + 1e-9) {
            std::cerr << "WARN: hard R@1 is 1.0 — verify zero-keyword still holds "
                         "(unexpected for bag-of-words)\n";
        }
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_eval_harness OK\n";
    return 0;
}
