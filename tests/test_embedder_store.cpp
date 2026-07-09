#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/word2vec.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static int g_fails = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n";  \
            ++g_fails;                                                              \
        }                                                                           \
    } while (0)

static nanorag::ChunkStore make_corpus() {
    nanorag::ChunkStore store;
    store.add({0, "animals",
               "cats meow purr and are small house pets that drink milk and chase mice"});
    store.add({1, "animals",
               "dogs bark wag and are house pets that go for walks and fetch balls"});
    store.add({2, "systems",
               "tinyann is a c++ vector similarity search library with hnsw ivf and exact indexes"});
    store.add({3, "systems",
               "nanollm is a from scratch llama style language model inference engine with chat"});
    store.add({4, "science",
               "water freezes at zero celsius and boils at one hundred under standard pressure"});
    store.add({5, "science",
               "the earth orbits the sun once per year and has one moon satellite"});
    return store;
}

int main() {
    // --- hashing still works ---
    {
        nanorag::HashingEmbedder emb(128);
        auto a = emb.embed("hello world");
        auto b = emb.embed("hello world");
        CHECK(a.size() == 128);
        double dot = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i] == b[i]);
            dot += a[i] * b[i];
        }
        CHECK(dot > 0.99);
    }

    // --- chunk store roundtrip ---
    {
        nanorag::ChunkStore store;
        store.add({1, "s", "alpha"});
        store.add({2, "s", "beta"});
        const auto dir = fs::temp_directory_path() / "nanorag_test_store";
        fs::create_directories(dir);
        const auto path = (dir / "chunks.tsv").string();
        store.save(path);
        auto loaded = nanorag::ChunkStore::load(path);
        CHECK(loaded.size() == 2);
        CHECK(loaded.get(2).text == "beta");
    }

    // --- Word2Vec train + retrieval quality ---
    {
        auto store = make_corpus();
        nanorag::Word2VecTrainConfig cfg;
        cfg.dim = 48;
        cfg.epochs = 80;
        cfg.doc_repeat = 16;
        cfg.window = 5;
        cfg.negative = 5;
        cfg.lr = 0.05f;
        cfg.seed = 12345;

        auto ret = nanorag::Retriever::build_word2vec(store, cfg);
        CHECK(ret.embedder().id() == std::string(nanorag::kWord2VecEmbedderId));

        auto cats = ret.retrieve("animal that meows and purrs", 1);
        CHECK(!cats.chunks.empty());
        CHECK(cats.chunks[0].id == 0);
        CHECK(cats.chunks[0].text.find("cat") != std::string::npos);

        auto ann = ret.retrieve("hnsw vector similarity search library", 1);
        CHECK(!ann.chunks.empty());
        CHECK(ann.chunks[0].id == 2);
        CHECK(ann.chunks[0].text.find("tinyann") != std::string::npos);

        auto water = ret.retrieve("freezing boiling temperature of water", 1);
        CHECK(!water.chunks.empty());
        CHECK(water.chunks[0].id == 4);

        // Grounding: prompt contains retrieved source text
        CHECK(cats.prompt.find(cats.chunks[0].text) != std::string::npos);
        CHECK(cats.prompt.find("[#0") != std::string::npos);

        // save / load preserves ranking
        const auto dir = fs::temp_directory_path() / "nanorag_test_w2v_index";
        fs::create_directories(dir);
        ret.save(dir.string());
        auto loaded = nanorag::Retriever::open(dir.string());
        CHECK(loaded.embedder().id() == std::string(nanorag::kWord2VecEmbedderId));
        auto cats2 = loaded.retrieve("animal that meows and purrs", 1);
        CHECK(!cats2.chunks.empty());
        CHECK(cats2.chunks[0].id == 0);
        auto ann2 = loaded.retrieve("hnsw vector similarity search library", 1);
        CHECK(!ann2.chunks.empty());
        CHECK(ann2.chunks[0].id == 2);
    }

    // --- Word2Vec model file roundtrip ---
    {
        std::vector<std::string> docs = {
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
            "alpha zeta shared vocabulary tokens here",
        };
        nanorag::Word2VecTrainConfig cfg;
        cfg.dim = 16;
        cfg.epochs = 30;
        cfg.doc_repeat = 10;
        cfg.seed = 7;
        auto m = nanorag::Word2VecModel::train(docs, cfg);
        const auto path = (fs::temp_directory_path() / "nanorag_test.nw2v").string();
        m.save(path);
        auto m2 = nanorag::Word2VecModel::load(path);
        CHECK(m2.dim() == m.dim());
        CHECK(m2.vocab_size() == m.vocab_size());
        auto e1 = m.embed_text("alpha beta");
        auto e2 = m2.embed_text("alpha beta");
        double err = 0;
        for (std::size_t i = 0; i < e1.size(); ++i) {
            err += std::abs(e1[i] - e2[i]);
        }
        CHECK(err < 1e-5);
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "nanorag_tests OK\n";
    return 0;
}
