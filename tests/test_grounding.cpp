#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/text_util.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

static int g_fails = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n";  \
            ++g_fails;                                                              \
        }                                                                           \
    } while (0)

static nanorag::ChunkStore corpus() {
    nanorag::ChunkStore store;
    store.add({10, "animals",
               "Felis catus is a small carnivorous mammal that shares dwellings with people and "
               "produces a low continuous vocalization when content."});
    store.add({20, "systems",
               "Tinyann implements hierarchical navigable small world graphs for in-memory "
               "nearest neighbor search in C++."});
    store.add({30, "science",
               "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K."});
    return store;
}

int main() {
    // --- citation parser ---
    {
        auto ids = nanorag::extract_citation_ids("Claim one [#10]. Also [20] and [#10] again.");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 10);
        CHECK(ids[1] == 20);
    }

    // --- query support utility: alcohol must not support water / cat text ---
    {
        const char* water =
            "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K.";
        const char* cat =
            "Felis catus is a small carnivorous mammal that shares dwellings with people.";
        int hits = 0;
        double s = nanorag::query_support_in_passage("What is the boiling point of alcohol?", water,
                                                     &hits);
        CHECK(hits == 0);
        CHECK(s < 0.05);
        s = nanorag::query_support_in_passage("What is the boiling point of alcohol?", cat, &hits);
        CHECK(hits == 0);
        // water query should support water passage (atmosphere / pure / H2O variants)
        s = nanorag::query_support_in_passage(
            "At one atmosphere when does pure water become gas?", water, &hits);
        // "atmosphere", "pure" should hit; water/gas may not
        CHECK(hits >= 1);
    }

    // --- near-miss: high-ish score but wrong topic → refuse ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.20f;
        cfg.min_score_without_query_support = 0.55f;
        cfg.min_query_support = 0.20f;
        // Spurious: embedding "likes" cat for alcohol question
        std::vector<nanorag::RetrievedChunk> cand = {
            {10, 0.39f, "animals",
             "Felis catus is a small carnivorous mammal that shares dwellings with people."},
            {30, 0.08f, "science",
             "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K."},
        };
        auto ga = nanorag::answer_extractive_for_query("What is the boiling point of alcohol?", cand,
                                                       cfg);
        CHECK(ga.refused);
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
        CHECK(ga.used.empty());
        CHECK(ga.check.ok);
    }

    // --- lexical support path: mammal/people query on cat passage ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.20f;
        cfg.min_score_without_query_support = 0.90f;  // force lexical path
        std::vector<nanorag::RetrievedChunk> cand = {
            {10, 0.40f, "animals",
             "Felis catus is a small carnivorous mammal that shares dwellings with people."},
            {20, 0.35f, "systems", "Tinyann implements hierarchical navigable small world graphs."},
        };
        auto ga = nanorag::answer_extractive_for_query("Which mammal lives with people?", cand, cfg);
        CHECK(!ga.refused);
        CHECK(ga.used.size() == 1);
        CHECK(ga.used[0].id == 10);
        CHECK(ga.check.ok);
        auto cites = nanorag::extract_citation_ids(ga.answer);
        CHECK(!cites.empty() && cites[0] == 10);
        // Tinyann must not be used (no mammal/people support, score < 0.90)
        for (const auto& u : ga.used) {
            CHECK(u.id != 20);
        }
    }

    // --- validator: illegal / missing / hallucinated ---
    {
        nanorag::GroundingConfig cfg;
        std::vector<nanorag::RetrievedChunk> used = {
            {10, 0.9f, "a", "Felis catus is a small carnivorous mammal."}};
        CHECK(!nanorag::validate_grounding("Cats are aliens [#999]", used, cfg).ok);
        CHECK(!nanorag::validate_grounding("Felis catus is a small carnivorous mammal.", used, cfg)
                   .ok);
        CHECK(!nanorag::validate_grounding("Quantum pineapple engines power starships [#10]", used,
                                           cfg)
                   .ok);
        CHECK(nanorag::validate_grounding("Felis catus is a small carnivorous mammal. [#10]", used,
                                          cfg)
                  .ok);
        CHECK(nanorag::validate_grounding(nanorag::kDontKnowAnswer, {}, cfg).ok);
        CHECK(!nanorag::validate_grounding("The moon is cheese.", {}, cfg).ok);
    }

    // --- model finalize: empty context forces I don't know ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.9f;
        std::vector<nanorag::RetrievedChunk> cand = {{1, 0.1f, "x", "noise"}};
        auto ga = nanorag::finalize_with_model_text("Secret?", cand, "The password is 12345 [#1]",
                                                    nullptr, cfg);
        CHECK(ga.refused);
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
    }

    // --- blank query refuses ---
    {
        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        auto ga = nanorag::answer_extractive_for_query("", {}, cfg);
        CHECK(ga.refused);
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
        CHECK(ga.check.ok);
    }

    // --- issue #12: is_dont_know is exact match only (not prefix) ---
    {
        CHECK(nanorag::is_dont_know("I don't know"));
        CHECK(nanorag::is_dont_know("  I do not know  "));
        CHECK(nanorag::is_dont_know("i don't know"));
        // Hallucinated prefix must NOT count as refuse (issue #12).
        CHECK(!nanorag::is_dont_know("I don't know. Water boils at 100C [#999]"));
        CHECK(!nanorag::is_dont_know("I do not know — the answer is cats [#10]"));
        CHECK(!nanorag::is_dont_know("I don't know something"));

        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        std::vector<nanorag::RetrievedChunk> used = {
            {10, 0.9f, "a", "Felis catus is a small carnivorous mammal."}};
        // Prefix + hallucinated claim + illegal cite must fail validation.
        auto bad = nanorag::validate_grounding(
            "I don't know. Water boils at 100C [#999]", used, cfg);
        CHECK(!bad.ok);
        CHECK(!bad.refused);

        // finalize_with_model_text must not accept the hallucinated refuse as success.
        std::vector<nanorag::RetrievedChunk> cand = used;
        auto ga = nanorag::finalize_with_model_text(
            "What is the boiling point of water?", cand,
            "I don't know. Water boils at 100C [#999]", nullptr, cfg);
        // Either forced exact refuse (empty used) or extractive fallback — never the
        // hallucinated string as a grounded "success".
        CHECK(ga.answer != std::string("I don't know. Water boils at 100C [#999]"));
        if (ga.refused) {
            CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
            CHECK(ga.check.ok);
        } else {
            CHECK(ga.check.ok);
            CHECK(ga.mode == "extractive" || ga.mode == "extractive_fallback" ||
                  ga.mode == "generated");
        }
    }

    // --- issue #15: BM25-scale scores must not unlock paraphrase path ---
    {
        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        const char* water =
            "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K.";
        const char* q = "What is the boiling point of ethanol under one atmosphere?";
        // score_is_cosine=false simulates --retrieve sparse (raw BM25).
        nanorag::RetrievedChunk bm25_hit{30, 3.5f, "science", water, /*score_is_cosine=*/false};
        CHECK(!nanorag::passage_is_answerable(q, bm25_hit, cfg));
        auto ga = nanorag::answer_extractive_for_query(q, {bm25_hit}, cfg);
        CHECK(ga.refused);
        CHECK(ga.used.empty());
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
    }

    // --- issues #11/#14: fusion inject magnitude (~0.85×dense top) is not cosine ---
    {
        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        const char* water =
            "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K.";
        const char* q = "What is the boiling point of ethanol under one atmosphere?";
        // Pre-fix behavior: fusion inject score 0.75 with default score_is_cosine=true
        // would incorrectly pass. After fix, non-cosine or low cosine refuses.
        nanorag::RetrievedChunk inject_as_ranking{30, 0.75f, "science", water,
                                                  /*score_is_cosine=*/false};
        CHECK(!nanorag::passage_is_answerable(q, inject_as_ranking, cfg));
        // True low cosine (what pipeline remaps injects to for near-miss topics).
        nanorag::RetrievedChunk inject_cosine_low{30, 0.15f, "science", water,
                                                  /*score_is_cosine=*/true};
        CHECK(!nanorag::passage_is_answerable(q, inject_cosine_low, cfg));
        // High real cosine still allows paraphrase path (by design).
        nanorag::RetrievedChunk cosine_para{30, 0.92f, "science", water, true};
        // ethanol missing → needs paraphrase score; 0.92 >= 0.60
        CHECK(nanorag::passage_is_answerable(q, cosine_para, cfg));
    }

    // --- non-cosine scores can still pass via full lexical support ---
    // Query terms must appear in the passage (no stemmer); avoid "water" vs "H2O".
    {
        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        const char* water =
            "Under one atmosphere pure H2O becomes solid at 273.15 K and gas at 373.15 K.";
        const char* q = "At one atmosphere when does pure H2O become gas?";
        // "become" (len>=5) is missing vs "becomes" → missing_distinctive unless we
        // only use tokens present in the passage.
        const char* q_lex = "pure H2O becomes gas under one atmosphere";
        nanorag::RetrievedChunk bm25_lex{30, 4.2f, "science", water, /*score_is_cosine=*/false};
        CHECK(nanorag::passage_is_answerable(q_lex, bm25_lex, cfg));
        auto ga = nanorag::answer_extractive_for_query(q_lex, {bm25_lex}, cfg);
        CHECK(!ga.refused);
        CHECK(!ga.used.empty());
        CHECK(ga.used[0].id == 30);
        // Near-miss with BM25 still refuses (ethanol missing + non-cosine).
        (void)q;
        nanorag::RetrievedChunk ood{30, 5.0f, "science", water, false};
        CHECK(!nanorag::passage_is_answerable(
            "What is the boiling point of ethanol under one atmosphere?", ood, cfg));
    }

    // --- sparse retrieve marks scores non-cosine (issue #15 end-to-end) ---
    {
        auto store = corpus();
        std::vector<nanorag::TrainPair> pairs = {
            {"At one atmosphere when does pure H2O become gas?", 30},
            {"Which small companion mammal vibrates softly when relaxed?", 10},
        };
        nanorag::ContrastiveTrainConfig tcfg;
        tcfg.dim = 32;
        tcfg.epochs = 40;
        tcfg.seed = 3;
        tcfg.hard_neg_k = 0;
        tcfg.query_query_weight = 0.0f;
        auto ret = nanorag::Retriever::build_contrastive(store, pairs, tcfg);
        ret.set_retrieve_mode(nanorag::RetrieveMode::Sparse);
        nanorag::GroundingConfig gcfg = nanorag::default_grounding_config();
        auto ga = ret.ask_grounded("What is the boiling point of ethanol under one atmosphere?", 5,
                                   gcfg);
        if (!ga.refused || !ga.used.empty()) {
            std::cerr << "SPARSE OOD FAIL refused=" << ga.refused << " used=" << ga.used.size()
                      << " answer=" << ga.answer << "\n";
            for (const auto& u : ga.used) {
                std::cerr << "  used id=" << u.id << " score=" << u.score
                          << " cosine=" << u.score_is_cosine << "\n";
            }
            ++g_fails;
        }
        // Confirm retrieve marks non-cosine on sparse hits.
        auto rr = ret.retrieve("atmosphere pure H2O gas", 3);
        for (const auto& c : rr.chunks) {
            CHECK(!c.score_is_cosine);
        }
    }

    // --- end-to-end: paraphrase ok, realistic near-miss refuse ---
    {
        auto store = corpus();
        std::vector<nanorag::TrainPair> pairs = {
            {"Which small companion mammal vibrates softly when relaxed?", 10},
            {"What domestic predator shares living space with humans?", 10},
            {"What pure C++ toolkit finds neighbors with graph search?", 20},
            {"Which library ships hierarchical navigable small world indexes?", 20},
            {"At one atmosphere when does pure water become ice on kelvin scale?", 30},
            {"At one atmosphere when does pure water become gas?", 30},
        };
        nanorag::ContrastiveTrainConfig tcfg;
        tcfg.dim = 64;
        tcfg.epochs = 220;
        tcfg.seed = 7;
        tcfg.hard_neg_k = 3;
        tcfg.hard_neg_start_epoch = 60;
        tcfg.hard_neg_loss_weight = 0.15f;
        tcfg.query_query_weight = 0.0f;
        auto ret = nanorag::Retriever::build_contrastive(store, pairs, tcfg);

        nanorag::GroundingConfig gcfg = nanorag::default_grounding_config();

        auto water = ret.ask_grounded("At one atmosphere when does pure H2O become gas?", 5, gcfg);
        CHECK(!water.refused);
        CHECK(water.check.ok);
        auto wc = nanorag::extract_citation_ids(water.answer);
        CHECK(std::find(wc.begin(), wc.end(), 30) != wc.end());

        // THE bug case: alcohol ≠ water/cat/planet
        const char* hard_ood[] = {
            "What is the boiling point of alcohol?",
            "What is the boiling point of ethanol?",
            "What is the melting point of iron?",
            "How many legs does a spider have?",
            "What is the capital of Germany?",
        };
        for (const char* q : hard_ood) {
            auto ga = ret.ask_grounded(q, 5, gcfg);
            if (!ga.refused || ga.answer != std::string(nanorag::kDontKnowAnswer) || !ga.used.empty()) {
                std::cerr << "HARD OOD FAIL q=" << q << " refused=" << ga.refused
                          << " answer=" << ga.answer << " used=" << ga.used.size() << "\n";
                for (const auto& u : ga.used) {
                    std::cerr << "  used id=" << u.id << " score=" << u.score << "\n";
                }
                ++g_fails;
            }
        }

        // Absurd OOD still refuses
        auto pizza = ret.ask_grounded("Who invented the chocolate pizza telescope?", 5, gcfg);
        CHECK(pizza.refused);
        CHECK(pizza.answer == std::string(nanorag::kDontKnowAnswer));
    }

    // --- issue #17: empty content-token answers fail grounding (not vacuous pass) ---
    {
        nanorag::GroundingConfig cfg = nanorag::default_grounding_config();
        CHECK(cfg.require_citations);
        std::vector<nanorag::RetrievedChunk> used = {
            {6, 0.9f, "a", "Felis catus is a small carnivorous mammal."}};
        // Citation-only / stopword-only answers have no content tokens after strip.
        CHECK(nanorag::content_support_ratio("Yes. [#6]", used) == 0.0);
        CHECK(nanorag::content_support_ratio("[#6]", used) == 0.0);
        CHECK(nanorag::content_support_ratio("Yes. No. [#6]", used) == 0.0);
        auto yes = nanorag::validate_grounding("Yes. [#6]", used, cfg);
        CHECK(!yes.ok);
        auto bare = nanorag::validate_grounding("[#6]", used, cfg);
        CHECK(!bare.ok);
        // Exact refuse still ok (handled before content_support).
        CHECK(nanorag::validate_grounding(nanorag::kDontKnowAnswer, {}, cfg).ok);
        CHECK(nanorag::validate_grounding(nanorag::kDontKnowAnswer, used, cfg).ok);
        // Real content still passes when grounded.
        CHECK(nanorag::validate_grounding("Felis catus is a small carnivorous mammal. [#6]", used,
                                          cfg)
                  .ok);
    }

    // --- issue #16: user chunk id -1 rejected; system sentinel allowed ---
    {
        nanorag::ChunkStore s;
        bool rejected = false;
        try {
            s.add({-1, "user", "this is a real user passage that collides with sentinel"});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK(rejected);
        CHECK(!s.contains(-1));

        bool rejected_other_neg = false;
        try {
            s.add({-2, "user", "other negative id"});
        } catch (const std::invalid_argument&) {
            rejected_other_neg = true;
        }
        CHECK(rejected_other_neg);

        // Official system sentinel inject succeeds.
        s.add({nanorag::kNoEvidenceId, "system", nanorag::kNoEvidenceText});
        CHECK(s.contains(nanorag::kNoEvidenceId));
        CHECK(s.get(nanorag::kNoEvidenceId).text == std::string(nanorag::kNoEvidenceText));

        // build_contrastive injects sentinel when missing.
        nanorag::ChunkStore s2;
        s2.add({0, "a", "cats purr when content"});
        std::vector<nanorag::TrainPair> pairs = {{"which pet purrs", 0}};
        nanorag::ContrastiveTrainConfig tcfg;
        tcfg.dim = 8;
        tcfg.epochs = 1;
        tcfg.seed = 1;
        tcfg.hard_neg_k = 0;
        auto ret = nanorag::Retriever::build_contrastive(s2, pairs, tcfg, {}, true);
        CHECK(ret.store().contains(nanorag::kNoEvidenceId));
        CHECK(ret.store().get(nanorag::kNoEvidenceId).source == "system");
        CHECK(ret.store().get(nanorag::kNoEvidenceId).text ==
              std::string(nanorag::kNoEvidenceText));
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_grounding OK (near-miss + paraphrase)\n";
    return 0;
}
