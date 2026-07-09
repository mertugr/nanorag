// nanorag CLI — local RAG over owned tinyann + nanollm + in-house embedders.
//
// Commands:
//   nanorag smoke
//   nanorag ingest --chunks data/demo/chunks.tsv --out index/demo [--dim N] [--embedder word2vec|hashing]
//   nanorag ask --index index/demo --query "..." [--k 3]
//   nanorag ask --index index/demo --query "..." --model m.nanollm --tokenizer t.nllmtok

#include "nanorag/chunk_store.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/word2vec.hpp"

#include "nanollm/generate.hpp"
#include "nanollm/model.hpp"
#include "nanollm/tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void usage(const char* argv0) {
    std::cerr
        << "nanorag — local RAG over tinyann + nanollm (owned stack)\n\n"
        << "Usage:\n"
        << "  " << argv0 << " smoke\n"
        << "  " << argv0
        << " ingest --chunks <file.tsv> --out <index_dir>\n"
        << "         [--dim N] [--embedder word2vec|hashing] [--epochs N]\n"
        << "  " << argv0 << " ask --index <index_dir> --query <text> [--k N]\n"
        << "      [--model <file.nanollm> --tokenizer <file.nllmtok>] [--max-tokens N]\n";
}

std::string require_arg(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

int cmd_smoke() {
    nanorag::ChunkStore store;
    store.add({0, "demo", "cats meow and are common house pets that like milk"});
    store.add({1, "demo", "dogs bark and are common house pets that like walks"});
    store.add({2, "demo", "tinyann is a vector similarity search library with hnsw indexes"});
    store.add({3, "demo", "nanollm is a llama style language model inference engine"});

    nanorag::Word2VecTrainConfig cfg;
    cfg.dim = 32;
    cfg.epochs = 60;
    cfg.doc_repeat = 20;
    cfg.seed = 42;
    auto retriever = nanorag::Retriever::build_word2vec(store, cfg);

    const std::string q1 = "which animal meows";
    auto r1 = retriever.retrieve(q1, 2);
    const std::string q2 = "vector search hnsw library";
    auto r2 = retriever.retrieve(q2, 2);

    std::cout << "nanorag smoke OK\n";
    std::cout << "embedder=" << retriever.embedder().id() << " dim=" << retriever.embedder().dim()
              << "\n";
    std::cout << "query1: " << q1 << "\n";
    for (const auto& h : r1.chunks) {
        std::cout << "  id=" << h.id << " score=" << h.score << " text=" << h.text << "\n";
    }
    std::cout << "query2: " << q2 << "\n";
    for (const auto& h : r2.chunks) {
        std::cout << "  id=" << h.id << " score=" << h.score << " text=" << h.text << "\n";
    }
    if (r1.chunks.empty() || r1.chunks[0].id != 0) {
        std::cerr << "smoke: expected top hit id=0 (cats) for query1\n";
        return 1;
    }
    if (r2.chunks.empty() || r2.chunks[0].id != 2) {
        std::cerr << "smoke: expected top hit id=2 (tinyann) for query2\n";
        return 1;
    }
    // Grounding: top chunk text should contain a distinctive token.
    if (r1.chunks[0].text.find("cat") == std::string::npos) {
        std::cerr << "smoke: grounding check failed (cats text)\n";
        return 1;
    }
    std::cout << "grounding checks OK\n";
    return 0;
}

int cmd_ingest(const std::string& chunks_path, const std::string& out_dir, std::size_t dim,
               const std::string& embedder_name, int epochs) {
    auto store = nanorag::ChunkStore::load(chunks_path);
    if (store.size() == 0) {
        throw std::runtime_error("ingest: empty chunk store");
    }
    fs::create_directories(out_dir);

    tinyann::HnswParams params;
    params.M = 16;
    params.ef_construction = 200;
    params.ef_search = 64;

    nanorag::Retriever retriever = [&]() {
        if (embedder_name == "hashing") {
            auto emb = std::make_shared<nanorag::HashingEmbedder>(dim);
            return nanorag::Retriever::build(emb, store, params);
        }
        if (embedder_name == "word2vec") {
            nanorag::Word2VecTrainConfig cfg;
            cfg.dim = dim;
            cfg.epochs = epochs;
            cfg.doc_repeat = 12;
            cfg.seed = 0xC0FFEEu;
            return nanorag::Retriever::build_word2vec(store, cfg, params);
        }
        throw std::runtime_error("ingest: unknown --embedder (use word2vec|hashing)");
    }();

    retriever.save(out_dir);
    std::cout << "ingested " << store.size() << " chunks → " << out_dir << "\n";
    std::cout << "  embedder=" << retriever.embedder().id() << " dim=" << retriever.embedder().dim()
              << "\n";
    std::cout << "  wrote chunks.tsv, vectors.hnsw.tann, meta.txt";
    if (embedder_name == "word2vec") {
        std::cout << ", embeddings.nw2v";
    }
    std::cout << "\n";
    return 0;
}

int cmd_ask(const std::string& index_dir, const std::string& query, std::size_t k,
            const std::string& model_path, const std::string& tok_path, int max_tokens) {
    auto retriever = nanorag::Retriever::open(index_dir);
    auto r = retriever.retrieve(query, k);

    std::cout << "=== retrieved (k=" << k << ", embedder=" << retriever.embedder().id()
              << ") ===\n";
    for (const auto& h : r.chunks) {
        std::cout << "[#" << h.id << "] score=" << h.score << " source=" << h.source << "\n"
                  << h.text << "\n\n";
    }

    if (model_path.empty()) {
        std::cout << "=== prompt (no model; pass --model/--tokenizer to generate) ===\n";
        std::cout << r.prompt << "\n";
        return 0;
    }
    if (tok_path.empty()) {
        throw std::runtime_error("ask: --tokenizer required with --model");
    }

    auto model = nanollm::LlamaModel::load(model_path);
    auto tokenizer = nanollm::Tokenizer::load(tok_path);
    nanollm::GenerateConfig cfg;
    cfg.max_new_tokens = max_tokens;
    cfg.add_bos = true;
    cfg.sampler.temperature = 0.f;

    std::cout << "=== generation ===\n";
    auto result = nanollm::generate_text(
        model, tokenizer, r.prompt, cfg, [](nanollm::index_t, const std::string& piece) {
            std::cout << piece << std::flush;
        });
    std::cout << "\n\n(generated_tokens=" << result.generated_tokens
              << " prompt_tokens=" << result.prompt_tokens << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 2;
        }
        const std::string cmd = argv[1];
        if (cmd == "smoke") {
            return cmd_smoke();
        }
        if (cmd == "ingest") {
            std::string chunks;
            std::string out;
            std::size_t dim = 64;
            std::string embedder = "word2vec";
            int epochs = 50;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--chunks") {
                    chunks = require_arg(i, argc, argv, "--chunks");
                } else if (a == "--out") {
                    out = require_arg(i, argc, argv, "--out");
                } else if (a == "--dim") {
                    dim = static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--dim")));
                } else if (a == "--embedder") {
                    embedder = require_arg(i, argc, argv, "--embedder");
                } else if (a == "--epochs") {
                    epochs = std::stoi(require_arg(i, argc, argv, "--epochs"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (chunks.empty() || out.empty()) {
                throw std::runtime_error("ingest requires --chunks and --out");
            }
            return cmd_ingest(chunks, out, dim, embedder, epochs);
        }
        if (cmd == "ask") {
            std::string index;
            std::string query;
            std::size_t k = 3;
            std::string model;
            std::string tok;
            int max_tokens = 64;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--index") {
                    index = require_arg(i, argc, argv, "--index");
                } else if (a == "--query" || a == "-q") {
                    query = require_arg(i, argc, argv, "--query");
                } else if (a == "--k") {
                    k = static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--k")));
                } else if (a == "--model") {
                    model = require_arg(i, argc, argv, "--model");
                } else if (a == "--tokenizer") {
                    tok = require_arg(i, argc, argv, "--tokenizer");
                } else if (a == "--max-tokens") {
                    max_tokens = std::stoi(require_arg(i, argc, argv, "--max-tokens"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (index.empty() || query.empty()) {
                throw std::runtime_error("ask requires --index and --query");
            }
            return cmd_ask(index, query, k, model, tok, max_tokens);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
