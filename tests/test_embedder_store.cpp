#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/pipeline.hpp"

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

int main() {
    {
        nanorag::HashingEmbedder emb(128);
        auto a = emb.embed("hello world");
        auto b = emb.embed("hello world");
        auto c = emb.embed("completely different tokens here");
        CHECK(a.size() == 128);
        double dot_ab = 0, dot_ac = 0, na = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i] == b[i]);
            dot_ab += a[i] * b[i];
            dot_ac += a[i] * c[i];
            na += a[i] * a[i];
        }
        CHECK(std::abs(na - 1.0) < 1e-4 || na == 0.0);
        CHECK(dot_ab > 0.99);
        CHECK(dot_ac < dot_ab);
    }

    {
        nanorag::ChunkStore store;
        store.add({1, "s", "alpha"});
        store.add({2, "s", "beta"});
        CHECK(store.size() == 2);
        CHECK(store.get(1).text == "alpha");

        const auto dir = fs::temp_directory_path() / "nanorag_test_store";
        fs::create_directories(dir);
        const auto path = (dir / "chunks.tsv").string();
        store.save(path);
        auto loaded = nanorag::ChunkStore::load(path);
        CHECK(loaded.size() == 2);
        CHECK(loaded.get(2).text == "beta");
    }

    {
        auto emb = std::make_shared<nanorag::HashingEmbedder>(64);
        nanorag::ChunkStore store;
        store.add({10, "d", "vector search with tinyann library"});
        store.add({20, "d", "cooking pasta with tomato sauce"});
        auto ret = nanorag::Retriever::build(emb, store);
        auto r = ret.retrieve("tinyann vector similarity", 1);
        CHECK(!r.chunks.empty());
        CHECK(r.chunks[0].id == 10);

        const auto dir = fs::temp_directory_path() / "nanorag_test_index";
        fs::create_directories(dir);
        ret.save(dir.string());
        auto emb2 = std::make_shared<nanorag::HashingEmbedder>(64);
        auto loaded = nanorag::Retriever::load(dir.string(), emb2);
        auto r2 = loaded.retrieve("tinyann vector similarity", 1);
        CHECK(!r2.chunks.empty());
        CHECK(r2.chunks[0].id == 10);
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "nanorag_tests OK\n";
    return 0;
}
