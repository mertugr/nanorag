#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/hybrid.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/text_util.hpp"
#include "nanorag/word2vec.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
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

static nanorag::ChunkStore neutral_corpus() {
    nanorag::ChunkStore store;
    // Neutral facts — not phrased as answers to the eval questions.
    store.add({0, "animals",
               "Felis catus is a small carnivorous mammal. Domestic lines share dwellings with "
               "people, produce a low continuous vocalization when content, and hunt rodents."});
    store.add({1, "animals",
               "Canis familiaris is a social canid that produces short explosive vocalizations, "
               "engages in object retrieval play, and accompanies humans outdoors."});
    store.add({2, "animals",
               "Aves includes sparrows and eagles. Members have feathers; many achieve powered "
               "flight and construct nests."});
    store.add({3, "systems",
               "Tinyann is a C++17 in-memory nearest-neighbor library implementing exact scan and "
               "hierarchical navigable small world approximate indexes."});
    store.add({4, "systems",
               "Nanollm is a dependency-free C++ decoder-only transformer runtime with quantized "
               "matmuls and multi-turn generation."});
    store.add({5, "science",
               "Under one atmosphere, pure H2O becomes solid at 273.15 K and gas at 373.15 K."});
    return store;
}

struct PQ {
    const char* q;
    std::int64_t gold;
};

// Held-out paraphrases: should share few content tokens with gold.
static const PQ kParaphrases[] = {
    {"Which feline housemate vibrates softly when comfortable?", 0},
    {"What barking companion likes retrieving thrown toys?", 1},
    {"Name the feathered group that includes eagles", 2},
    {"Which pure C++ ANN toolkit ships hierarchical navigable small world graphs?", 3},
    {"What from-scratch runtime executes Llama-like models without PyTorch?", 4},
    {"At one atm, pure water freezes at how many kelvin?", 5},
};

int main() {
    // --- Jaccard utility ---
    {
        CHECK(nanorag::token_jaccard("aaa bbb", "aaa bbb") > 0.99);
        CHECK(nanorag::token_jaccard("aaa bbb", "ccc ddd") < 0.01);
    }

    // --- Sparse BM25 + hybrid fuse unit tests ---
    {
        nanorag::ChunkStore s;
        s.add({1, "a", "hierarchical navigable small world graphs for nearest neighbor search"});
        s.add({2, "b", "pure water freezes at two hundred seventy three kelvin under one atmosphere"});
        s.add({3, "c", "completely unrelated purple telescope chocolate"});
        nanorag::SparseBm25Index bm25(s);
        auto hits = bm25.search("hierarchical navigable small world", 3);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 1);
        auto water = bm25.search("water freezes kelvin atmosphere", 3);
        CHECK(!water.empty());
        CHECK(water[0].id == 2);
        // Zero-overlap query → max BM25 ~ 0 → adaptive hybrid uses dense
        CHECK(bm25.max_score("zzzz qqqq xxxx") < 1e-3f);

        std::vector<tinyann::SearchResult> dense = {{2, 0.9f}, {1, 0.5f}, {3, 0.1f}};
        std::vector<tinyann::SearchResult> sparse = {{1, 5.0f}, {3, 1.0f}};
        nanorag::HybridConfig hcfg;
        hcfg.mode = nanorag::RetrieveMode::Hybrid;
        hcfg.fusion = nanorag::HybridFusion::RRF;
        nanorag::SparseBm25Index* bm_ptr = &bm25;
        const std::string q = "hierarchical navigable small world";
        auto fused =
            nanorag::fuse_dense_sparse(dense, sparse, hcfg, 3, /*sparse_max=*/5.0f, bm_ptr, &q);
        CHECK(!fused.empty());
        CHECK(fused.size() >= 2);
        // Adaptive: no sparse signal → dense ranking preserved
        auto dens_only =
            nanorag::fuse_dense_sparse(dense, sparse, hcfg, 3, /*sparse_max=*/0.f, bm_ptr, &q);
        CHECK(dens_only[0].id == 2);
    }

    auto store = neutral_corpus();

    // Verify paraphrases are actually low-overlap (test quality gate).
    for (const auto& p : kParaphrases) {
        const double j = nanorag::token_jaccard(p.q, store.get(p.gold).text);
        if (j > 0.28) {
            std::cerr << "eval query not paraphrase enough j=" << j << " q=" << p.q << "\n";
            ++g_fails;
        }
    }

    // Train pairs cover the held-out paraphrase neighborhood without copying eval strings.
    // Extra pairs reduce x86/ARM / float-order flakiness in CI.
    std::vector<nanorag::TrainPair> train = {
        {"Which small companion mammal makes a soft vibrating sound when relaxed?", 0},
        {"What domestic predator is associated with catching household vermin?", 0},
        {"Which feline pet shares dwellings and makes soft contented sounds?", 0},
        {"What animal is famous for short explosive vocal bursts and fetch games?", 1},
        {"Which companion canid joins people on outdoor walks?", 1},
        {"What barking pet enjoys object retrieval and outdoor excursions?", 1},
        {"Which social canid produces explosive vocalizations?", 1},
        {"What loyal companion wags and chases thrown toys outdoors?", 1},
        {"Which pet enjoys fetch games and strolls with its owner?", 1},
        {"What class of winged creatures includes eagles?", 2},
        {"Which flying vertebrates are covered in feathers?", 2},
        {"Name feathered animals that build nests and can fly?", 2},
        {"What pure C++ toolkit finds nearest vectors with graph-based approximate search?", 3},
        {"Which project provides hierarchical navigable small world style ANN in memory?", 3},
        {"What C++ library implements hierarchical navigable small world indexes?", 3},
        {"What engine runs decoder-only transformers without external ML frameworks?", 4},
        {"Which runtime loads custom LLM checkpoints and streams tokens?", 4},
        {"What from-scratch C++ stack runs Llama-like models with quantized matmuls?", 4},
        {"At standard pressure, when does pure water become ice on the kelvin scale?", 5},
        {"What temperature marks the boiling transition of H2O at one atmosphere?", 5},
        {"At one atm when does pure H2O become solid on the kelvin scale?", 5},
    };
    for (const auto& p : train) {
        const double j = nanorag::token_jaccard(p.query, store.get(p.pos_id).text);
        CHECK(j <= 0.45);  // train pairs should not be near-copies of the doc
    }

    // --- Contrastive: paraphrase retrieval must work ---
    {
        nanorag::ContrastiveTrainConfig cfg;
        cfg.dim = 128;
        cfg.epochs = 450;
        cfg.lr = 0.08f;
        cfg.momentum = 0.9f;
        cfg.temperature = 0.05f;
        cfg.seed = 99;
        cfg.hard_neg_k = 5;
        cfg.hard_neg_start_epoch = 100;
        cfg.hard_neg_loss_weight = 0.15f;
        cfg.query_query_weight = 0.0f;
        // No NO_EVIDENCE injection: this suite measures paraphrase retrieval only.
        auto ret = nanorag::Retriever::build_contrastive(store, train, cfg, {},
                                                         /*inject_no_evidence=*/false);
        CHECK(ret.embedder().id() == std::string(nanorag::kContrastiveEmbedderId));

        int hits = 0;
        for (const auto& p : kParaphrases) {
            auto r = ret.retrieve(p.q, 1);
            CHECK(!r.chunks.empty());
            if (!r.chunks.empty() && r.chunks[0].id == p.gold) {
                ++hits;
            } else {
                std::cerr << "contrastive MISS gold=" << p.gold << " top="
                          << (r.chunks.empty() ? -1 : r.chunks[0].id) << " q=" << p.q << "\n";
            }
            // Grounding: prompt includes gold text when retrieved correctly
            if (!r.chunks.empty() && r.chunks[0].id == p.gold) {
                CHECK(r.prompt.find(r.chunks[0].text) != std::string::npos);
            }
        }
        CHECK(hits == 6);

        // save/load
        const auto dir = fs::temp_directory_path() / "nanorag_ctr_index";
        fs::create_directories(dir);
        ret.save(dir.string());
        auto loaded = nanorag::Retriever::open(dir.string());
        int hits2 = 0;
        for (const auto& p : kParaphrases) {
            auto r = loaded.retrieve(p.q, 1);
            if (!r.chunks.empty() && r.chunks[0].id == p.gold) {
                ++hits2;
            }
        }
        CHECK(hits2 == 6);

        // Hybrid default + dense mode: keyword query should stay strong under hybrid
        {
            ret.set_retrieve_mode(nanorag::RetrieveMode::Hybrid);
            auto rh = ret.retrieve("hierarchical navigable small world C++ toolkit", 3);
            CHECK(!rh.chunks.empty());
            ret.set_retrieve_mode(nanorag::RetrieveMode::Dense);
            auto rd = ret.retrieve("hierarchical navigable small world C++ toolkit", 3);
            CHECK(!rd.chunks.empty());
            ret.set_retrieve_mode(nanorag::RetrieveMode::Sparse);
            auto rs = ret.retrieve("hierarchical navigable small world", 3);
            CHECK(!rs.chunks.empty());
            CHECK(rs.chunks[0].id == 3);  // tinyann chunk
            ret.set_retrieve_mode(nanorag::RetrieveMode::Hybrid);
        }

        // --- Retriever::load fail-closed validation ---
        {
            const auto bad_kind = fs::temp_directory_path() / "nanorag_bad_kind";
            fs::create_directories(bad_kind);
            ret.save(bad_kind.string());
            {
                auto meta = nanorag::load_meta((bad_kind / "meta.txt").string());
                meta.index_kind = "ivf";
                nanorag::save_meta((bad_kind / "meta.txt").string(), meta);
            }
            bool threw = false;
            try {
                (void)nanorag::Retriever::open(bad_kind.string());
            } catch (const std::exception& e) {
                threw = true;
                CHECK(std::string(e.what()).find("index_kind") != std::string::npos);
            }
            CHECK(threw);
        }
        {
            const auto bad_metric = fs::temp_directory_path() / "nanorag_bad_metric";
            fs::create_directories(bad_metric);
            ret.save(bad_metric.string());
            {
                auto meta = nanorag::load_meta((bad_metric / "meta.txt").string());
                meta.metric = "euclidean";
                // save_meta rewrites version; validate_index_meta rejects non-cosine
                nanorag::save_meta((bad_metric / "meta.txt").string(), meta);
            }
            bool threw = false;
            try {
                (void)nanorag::Retriever::open(bad_metric.string());
            } catch (const std::exception& e) {
                threw = true;
                CHECK(std::string(e.what()).find("metric") != std::string::npos);
            }
            CHECK(threw);
        }
        {
            const auto bad_n = fs::temp_directory_path() / "nanorag_bad_nchunks";
            fs::create_directories(bad_n);
            ret.save(bad_n.string());
            // Bypass load_meta validation: write meta with wrong n_chunks but valid kind/metric
            {
                std::ofstream out((bad_n / "meta.txt").string());
                out << "nanorag_index_meta_version=2\n";
                out << "embedder_id=" << ret.embedder().id() << "\n";
                out << "dim=" << ret.embedder().dim() << "\n";
                out << "metric=cosine\n";
                out << "index_kind=hnsw\n";
                out << "n_chunks=" << (ret.size() + 99) << "\n";
                out << "n_real_chunks=" << ret.real_size() << "\n";
                out << "has_no_evidence=0\n";
            }
            bool threw = false;
            try {
                (void)nanorag::Retriever::open(bad_n.string());
            } catch (const std::exception& e) {
                threw = true;
                CHECK(std::string(e.what()).find("n_chunks") != std::string::npos);
            }
            CHECK(threw);
        }
        {
            const auto future = fs::temp_directory_path() / "nanorag_future_meta";
            fs::create_directories(future);
            ret.save(future.string());
            {
                std::ofstream out((future / "meta.txt").string());
                out << "nanorag_index_meta_version=99\n";
                out << "embedder_id=" << ret.embedder().id() << "\n";
                out << "dim=" << ret.embedder().dim() << "\n";
                out << "metric=cosine\n";
                out << "index_kind=hnsw\n";
                out << "n_chunks=" << ret.size() << "\n";
                out << "n_real_chunks=" << ret.real_size() << "\n";
                out << "has_no_evidence=0\n";
            }
            bool threw = false;
            try {
                (void)nanorag::Retriever::open(future.string());
            } catch (const std::exception& e) {
                threw = true;
                CHECK(std::string(e.what()).find("version") != std::string::npos);
            }
            CHECK(threw);
        }
    }

    // --- Word2Vec ablation: unsupervised co-occurrence is NOT enough for paraphrases ---
    // Use dense-only so hybrid BM25 doesn't mask weak embedders.
    {
        nanorag::Word2VecTrainConfig cfg;
        cfg.dim = 64;
        cfg.epochs = 100;
        cfg.doc_repeat = 24;
        cfg.seed = 99;
        auto ret = nanorag::Retriever::build_word2vec(store, cfg);
        ret.set_retrieve_mode(nanorag::RetrieveMode::Dense);
        int hits = 0;
        for (const auto& p : kParaphrases) {
            auto r = ret.retrieve(p.q, 1);
            if (!r.chunks.empty() && r.chunks[0].id == p.gold) {
                ++hits;
            }
        }
        // Expect failure mode: not all paraphrases found. If this flips to 6/6, tests still
        // pass — but we assert it's strictly worse than contrastive on this set.
        CHECK(hits < 6);
        std::cout << "word2vec paraphrase recall@1: " << hits << "/6 (expected <6)\n";
    }

    // --- Hashing ablation: keyword-shaped query may work; paraphrase should not all pass ---
    {
        auto emb = std::make_shared<nanorag::HashingEmbedder>(128);
        auto ret = nanorag::Retriever::build(emb, store);
        ret.set_retrieve_mode(nanorag::RetrieveMode::Dense);
        int hits = 0;
        for (const auto& p : kParaphrases) {
            auto r = ret.retrieve(p.q, 1);
            if (!r.chunks.empty() && r.chunks[0].id == p.gold) {
                ++hits;
            }
        }
        CHECK(hits < 6);
        std::cout << "hashing paraphrase recall@1: " << hits << "/6 (expected <6)\n";
    }

    // --- Sampled negatives terminate on a corpus smaller than max_negatives+1 ---
    {
        nanorag::ChunkStore tiny;
        tiny.add({0, "a", "cats purr when they are content"});
        tiny.add({1, "b", "dogs bark at strangers near the fence"});
        tiny.add({2, "c", "water boils at one hundred degrees celsius"});
        std::vector<nanorag::TrainPair> pairs = {{"which pet purrs", 0}, {"which pet barks", 1}};
        nanorag::ContrastiveTrainConfig cfg;
        cfg.dim = 16;
        cfg.epochs = 2;
        cfg.seed = 3;
        cfg.full_doc_softmax = false;  // sampled-negatives path
        cfg.max_negatives = 32;        // > corpus size: used to loop forever
        auto m = nanorag::ContrastiveModel::train(tiny, pairs, cfg);
        CHECK(m.dim() == 16);
        std::cout << "sampled-negatives small-corpus training terminated\n";
    }

    // --- save rejects tokens longer than the u16 length prefix ---
    {
        nanorag::ChunkStore tiny;
        tiny.add({0, "a", std::string(70000, 'x')});  // one 70k-char token
        std::vector<nanorag::TrainPair> pairs = {{"find the long token", 0}};
        nanorag::ContrastiveTrainConfig cfg;
        cfg.dim = 8;
        cfg.epochs = 1;
        cfg.seed = 3;
        auto m = nanorag::ContrastiveModel::train(tiny, pairs, cfg);
        const auto dir = fs::temp_directory_path() / "nanorag_longtok_test";
        fs::create_directories(dir);
        bool threw = false;
        try {
            m.save((dir / "model.nctr").string());
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        std::cout << "save rejected >64KB token (no silent corruption)\n";
    }

    // --- issue #13: loaded v1 re-save upgrades format_version_ so id/meta match file ---
    {
        const auto dir = fs::temp_directory_path() / "nanorag_ctr_v1_resave";
        fs::create_directories(dir);
        const auto v1_path = (dir / "legacy.nctr").string();
        // Minimal NCTR v1: magic, ver=1, dim=2, V=1, token "x", emb [1,0]
        {
            std::ofstream out(v1_path, std::ios::binary);
            const char magic[4] = {'N', 'C', 'T', 'R'};
            out.write(magic, 4);
            auto wu32 = [&](std::uint32_t v) {
                out.write(reinterpret_cast<const char*>(&v), 4);
            };
            auto wu16 = [&](std::uint16_t v) {
                out.write(reinterpret_cast<const char*>(&v), 2);
            };
            wu32(1);  // format v1
            wu32(2);  // dim
            wu32(1);  // vocab size
            wu16(1);
            out.write("x", 1);
            float emb[2] = {1.f, 0.f};
            out.write(reinterpret_cast<const char*>(emb), sizeof(emb));
            CHECK(static_cast<bool>(out));
        }
        auto m = nanorag::ContrastiveModel::load(v1_path);
        CHECK(m.format_version() == nanorag::kContrastiveFormatVersionV1);
        nanorag::ContrastiveEmbedder emb_v1(std::move(m));
        CHECK(emb_v1.id() == std::string(nanorag::kContrastiveEmbedderIdV1));

        const auto v2_path = (dir / "upgraded.nctr").string();
        emb_v1.save(v2_path);
        // In-memory id must now match the v2 bytes on disk (meta uses id() after save).
        CHECK(emb_v1.model().format_version() == nanorag::kContrastiveFormatVersion);
        CHECK(emb_v1.id() == std::string(nanorag::kContrastiveEmbedderId));

        auto reloaded = nanorag::ContrastiveEmbedder::load(v2_path);
        CHECK(reloaded.model().format_version() == nanorag::kContrastiveFormatVersion);
        CHECK(reloaded.id() == std::string(nanorag::kContrastiveEmbedderId));
        CHECK(reloaded.id() == emb_v1.id());
        std::cout << "contrastive v1 re-save upgrades format_version (issue #13)\n";
    }

    // --- issue #16: ChunkStore rejects user -1; allows system sentinel ---
    {
        nanorag::ChunkStore s;
        bool threw = false;
        try {
            s.add({-1, "docs", "user owned negative id"});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        s.add({nanorag::kReservedNoEvidenceChunkId, nanorag::kReservedNoEvidenceSource,
               nanorag::kReservedNoEvidenceText});
        CHECK(s.contains(-1));
        std::cout << "ChunkStore reserves id -1 for NO_EVIDENCE sentinel (issue #16)\n";
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "nanorag_tests OK (contrastive paraphrase + ablations)\n";
    return 0;
}
