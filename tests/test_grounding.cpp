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
        tcfg.dim = 48;
        tcfg.epochs = 200;
        tcfg.seed = 7;
        auto ret = nanorag::Retriever::build_contrastive(store, pairs, tcfg);

        nanorag::GroundingConfig gcfg;
        gcfg.min_score = 0.20f;
        gcfg.min_score_without_query_support = 0.55f;
        gcfg.min_query_support = 0.15f;

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

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_grounding OK (near-miss + paraphrase)\n";
    return 0;
}
