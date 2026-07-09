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
               "Under one atmosphere pure H2O becomes solid at 273.15 K."});
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

    // --- refuse when no relevant hits ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.5f;
        std::vector<nanorag::RetrievedChunk> cand = {
            {99, 0.1f, "x", "totally unrelated low score passage about ships"}};
        auto ga = nanorag::answer_extractive_for_query("What is a cat?", cand, cfg);
        CHECK(ga.refused);
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
        CHECK(ga.check.ok);
        CHECK(ga.used.empty());
    }

    // --- extractive answer cites only used ids and is supported ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.2f;
        std::vector<nanorag::RetrievedChunk> cand = {
            {10, 0.9f, "animals",
             "Felis catus is a small carnivorous mammal that shares dwellings with people."},
            {20, 0.15f, "systems", "Tinyann implements hierarchical navigable small world graphs."},
        };
        auto ga = nanorag::answer_extractive_for_query("Which mammal lives with people?", cand, cfg);
        CHECK(!ga.refused);
        CHECK(ga.mode == "extractive");
        CHECK(ga.used.size() == 1);
        CHECK(ga.used[0].id == 10);
        CHECK(ga.check.ok);
        auto cites = nanorag::extract_citation_ids(ga.answer);
        CHECK(cites.size() == 1);
        CHECK(cites[0] == 10);
        CHECK(ga.answer.find("Felis") != std::string::npos);
        // Illegal citation must fail validation
        auto bad = nanorag::validate_grounding("Cats are aliens [#999]", ga.used, cfg);
        CHECK(!bad.ok);
        CHECK(!bad.illegal_ids.empty());
        // Missing citation fails
        auto missing = nanorag::validate_grounding("Felis catus is a small carnivorous mammal.", ga.used, cfg);
        CHECK(!missing.ok);
        // Ungrounded content fails support
        auto hallu = nanorag::validate_grounding(
            "Quantum pineapple engines power starships [#10]", ga.used, cfg);
        CHECK(!hallu.ok);
    }

    // --- model finalize: ungrounded model text → extractive fallback ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.2f;
        std::vector<nanorag::RetrievedChunk> cand = {
            {10, 0.85f, "a", "Felis catus is a small carnivorous mammal."}};
        auto ga = nanorag::finalize_with_model_text(
            "What is felis?", cand, "I think cats come from Mars and eat lasers.", cfg);
        CHECK(ga.mode == "extractive_fallback");
        CHECK(ga.check.ok);
        CHECK(nanorag::extract_citation_ids(ga.answer)[0] == 10);
    }

    // --- model finalize: empty context forces I don't know even if model bluffs ---
    {
        nanorag::GroundingConfig cfg;
        cfg.min_score = 0.9f;
        std::vector<nanorag::RetrievedChunk> cand = {{1, 0.1f, "x", "noise"}};
        auto ga = nanorag::finalize_with_model_text(
            "Secret?", cand, "The password is 12345 [#1]", cfg);
        CHECK(ga.refused);
        CHECK(ga.answer == std::string(nanorag::kDontKnowAnswer));
        CHECK(ga.check.ok);
    }

    // --- end-to-end retriever grounding on paraphrase + OOD refuse ---
    {
        auto store = corpus();
        std::vector<nanorag::TrainPair> pairs = {
            {"Which small companion mammal vibrates softly when relaxed?", 10},
            {"What domestic predator shares living space with humans?", 10},
            {"What pure C++ toolkit finds neighbors with graph search?", 20},
            {"Which library ships hierarchical navigable small world indexes?", 20},
            {"At one atmosphere when does pure water become ice on kelvin scale?", 30},
        };
        nanorag::ContrastiveTrainConfig tcfg;
        tcfg.dim = 48;
        tcfg.epochs = 180;
        tcfg.seed = 7;
        auto ret = nanorag::Retriever::build_contrastive(store, pairs, tcfg);

        nanorag::GroundingConfig gcfg;
        gcfg.min_score = 0.20f;
        gcfg.max_context_chunks = 2;

        auto in_domain = ret.ask_grounded("Which feline housemate vibrates when comfortable?", 5, gcfg);
        CHECK(!in_domain.refused);
        CHECK(in_domain.check.ok);
        auto cites = nanorag::extract_citation_ids(in_domain.answer);
        CHECK(!cites.empty());
        CHECK(cites[0] == 10 || (cites.size() && std::find(cites.begin(), cites.end(), 10) != cites.end()));
        CHECK(in_domain.answer.find("Felis") != std::string::npos ||
              in_domain.answer.find("felis") != std::string::npos ||
              in_domain.answer.find("carnivorous") != std::string::npos);

        // Out-of-domain: should refuse (no relevant passage about pizza telescopes)
        auto ood = ret.ask_grounded(
            "Who invented the chocolate pizza telescope in medieval France?", 5, gcfg);
        CHECK(ood.refused);
        CHECK(ood.answer == std::string(nanorag::kDontKnowAnswer));
        CHECK(ood.check.ok);
        CHECK(ood.used.empty());
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_grounding OK\n";
    return 0;
}
