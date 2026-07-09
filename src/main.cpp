// nanorag CLI — owned-stack RAG (tinyann + nanollm + in-house embedders).
//
// Default embedder is contrastive-v1 (needs --pairs). Word2Vec/hashing remain available
// for ablation; they are NOT expected to pass paraphrase retrieval.

#include "nanorag/chunk_store.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/text_util.hpp"
#include "nanorag/word2vec.hpp"

#include "nanollm/generate.hpp"
#include "nanollm/model.hpp"
#include "nanollm/tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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
        << "  " << argv0 << " ingest --chunks <tsv> --out <dir>\n"
        << "         [--embedder contrastive|word2vec|hashing]\n"
        << "         [--pairs <train_pairs.tsv>]   # required for contrastive\n"
        << "         [--dim N] [--epochs N]\n"
        << "  " << argv0 << " ask --index <dir> --query <text> [--k N]\n"
        << "      [--model <file.nanollm> --tokenizer <file.nllmtok>] [--max-tokens N]\n"
        << "  " << argv0 << " eval-paraphrase --index <dir> --pairs <eval.tsv>\n"
        << "         [--max-jaccard 0.25]   # fail if query/gold overlap above this\n";
}

std::string require_arg(int& i, int argc, char** argv, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

struct EvalPair {
    std::string query;
    std::int64_t gold_id = 0;
};

std::vector<EvalPair> load_eval_pairs(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<EvalPair> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("bad eval line: " + line);
        }
        EvalPair p;
        p.query = line.substr(0, tab);
        p.gold_id = std::stoll(line.substr(tab + 1));
        out.push_back(std::move(p));
    }
    return out;
}

int cmd_smoke() {
    // Minimal neutral docs + paraphrase train pairs (not copies of the docs).
    nanorag::ChunkStore store;
    store.add({0, "a",
               "Felis catus is a small carnivorous mammal that shares dwellings with people and "
               "produces a low continuous vocalization when content."});
    store.add({1, "a",
               "Canis familiaris is a social canid known for explosive vocal bursts and object "
               "retrieval play."});
    store.add({2, "s",
               "Tinyann implements hierarchical navigable small world graphs for in-memory "
               "nearest neighbor search in C++."});
    store.add({3, "s",
               "Nanollm is a dependency-free decoder-only transformer runtime with quantized "
               "matmuls."});

    std::vector<nanorag::TrainPair> pairs = {
        {"Which small companion mammal makes a soft vibrating sound when relaxed?", 0},
        {"What domestic predator shares living space with humans?", 0},
        {"What animal is famous for short explosive sounds and fetch games?", 1},
        {"What pure C++ toolkit finds neighbors with graph-based approximate search?", 2},
        {"Which project provides hierarchical navigable small world style ANN?", 2},
        {"What engine runs decoder-only transformers without external ML frameworks?", 3},
    };

    // Enforce that smoke pairs are not keyword cheats.
    for (const auto& p : pairs) {
        const double j = nanorag::token_jaccard(p.query, store.get(p.pos_id).text);
        if (j > 0.35) {
            std::cerr << "smoke: train pair too lexically similar (jaccard=" << j << "): " << p.query
                      << "\n";
            return 1;
        }
    }

    nanorag::ContrastiveTrainConfig cfg;
    cfg.dim = 48;
    cfg.epochs = 150;
    cfg.seed = 42;
    auto ret = nanorag::Retriever::build_contrastive(store, pairs, cfg);

    const std::string q = "Which feline housemate vibrates softly when comfortable?";
    const double jq = nanorag::token_jaccard(q, store.get(0).text);
    auto r = ret.retrieve(q, 2);
    std::cout << "nanorag smoke (contrastive paraphrase)\n";
    std::cout << "embedder=" << ret.embedder().id() << " dim=" << ret.embedder().dim() << "\n";
    std::cout << "query: " << q << "\n";
    std::cout << "token_jaccard(query, gold)=" << jq << "\n";
    for (const auto& h : r.chunks) {
        std::cout << "  id=" << h.id << " score=" << h.score << "\n";
    }
    if (jq > 0.35) {
        std::cerr << "smoke: eval query not a paraphrase (jaccard too high)\n";
        return 1;
    }
    if (r.chunks.empty() || r.chunks[0].id != 0) {
        std::cerr << "smoke: expected gold id=0 for paraphrase query\n";
        return 1;
    }
    // Ablation: word2vec alone should struggle on this paraphrase (document weakness).
    nanorag::Word2VecTrainConfig wcfg;
    wcfg.dim = 48;
    wcfg.epochs = 80;
    wcfg.doc_repeat = 20;
    wcfg.seed = 42;
    auto w2v = nanorag::Retriever::build_word2vec(store, wcfg);
    auto rw = w2v.retrieve(q, 1);
    std::cout << "word2vec ablation top id=" << (rw.chunks.empty() ? -1 : rw.chunks[0].id)
              << " (not required to match gold; contrastive is)\n";
    std::cout << "smoke OK\n";
    return 0;
}

int cmd_ingest(const std::string& chunks_path, const std::string& out_dir, std::size_t dim,
               const std::string& embedder_name, const std::string& pairs_path, int epochs) {
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
            return nanorag::Retriever::build(std::make_shared<nanorag::HashingEmbedder>(dim), store,
                                             params);
        }
        if (embedder_name == "word2vec") {
            nanorag::Word2VecTrainConfig cfg;
            cfg.dim = dim;
            cfg.epochs = epochs;
            cfg.doc_repeat = 12;
            return nanorag::Retriever::build_word2vec(store, cfg, params);
        }
        if (embedder_name == "contrastive") {
            if (pairs_path.empty()) {
                throw std::runtime_error(
                    "ingest: contrastive embedder requires --pairs <train_pairs.tsv>");
            }
            auto pairs = nanorag::load_train_pairs(pairs_path);
            nanorag::ContrastiveTrainConfig cfg;
            cfg.dim = dim;
            cfg.epochs = epochs;
            return nanorag::Retriever::build_contrastive(store, pairs, cfg, params);
        }
        throw std::runtime_error("ingest: unknown --embedder");
    }();

    retriever.save(out_dir);
    std::cout << "ingested " << store.size() << " chunks → " << out_dir << "\n";
    std::cout << "  embedder=" << retriever.embedder().id() << " dim=" << retriever.embedder().dim()
              << "\n";
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
        std::cout << "=== prompt (pass --model/--tokenizer to generate) ===\n";
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
    cfg.sampler.temperature = 0.f;
    std::cout << "=== generation ===\n";
    auto result = nanollm::generate_text(
        model, tokenizer, r.prompt, cfg, [](nanollm::index_t, const std::string& piece) {
            std::cout << piece << std::flush;
        });
    std::cout << "\n\n(generated_tokens=" << result.generated_tokens << ")\n";
    return 0;
}

int cmd_eval_paraphrase(const std::string& index_dir, const std::string& pairs_path,
                        double max_jaccard) {
    auto retriever = nanorag::Retriever::open(index_dir);
    auto pairs = load_eval_pairs(pairs_path);
    int ok = 0;
    int n = 0;
    for (const auto& p : pairs) {
        if (!retriever.store().contains(p.gold_id)) {
            throw std::runtime_error("eval: missing gold id " + std::to_string(p.gold_id));
        }
        const auto& gold = retriever.store().get(p.gold_id);
        const double j = nanorag::token_jaccard(p.query, gold.text);
        if (j > max_jaccard) {
            std::cerr << "FAIL overlap: jaccard=" << j << " > " << max_jaccard << " for query: "
                      << p.query << "\n";
            return 1;
        }
        auto r = retriever.retrieve(p.query, 1);
        const bool hit = !r.chunks.empty() && r.chunks[0].id == p.gold_id;
        std::cout << (hit ? "OK " : "MISS ") << "gold=" << p.gold_id
                  << " top=" << (r.chunks.empty() ? -1 : r.chunks[0].id) << " jaccard=" << j
                  << " q=" << p.query << "\n";
        if (hit) {
            ++ok;
        }
        ++n;
    }
    const double acc = n ? static_cast<double>(ok) / n : 0.0;
    std::cout << "paraphrase recall@1: " << ok << "/" << n << " (" << acc << ")\n";
    std::cout << "embedder=" << retriever.embedder().id() << "\n";
    // Require perfect on the small held-out set for contrastive indexes.
    if (ok != n) {
        return 1;
    }
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
            std::string chunks, out, pairs;
            std::size_t dim = 64;
            std::string embedder = "contrastive";
            int epochs = 200;
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
                } else if (a == "--pairs") {
                    pairs = require_arg(i, argc, argv, "--pairs");
                } else if (a == "--epochs") {
                    epochs = std::stoi(require_arg(i, argc, argv, "--epochs"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (chunks.empty() || out.empty()) {
                throw std::runtime_error("ingest requires --chunks and --out");
            }
            return cmd_ingest(chunks, out, dim, embedder, pairs, epochs);
        }
        if (cmd == "ask") {
            std::string index, query, model, tok;
            std::size_t k = 3;
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
        if (cmd == "eval-paraphrase") {
            std::string index, pairs;
            double max_j = 0.25;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--index") {
                    index = require_arg(i, argc, argv, "--index");
                } else if (a == "--pairs") {
                    pairs = require_arg(i, argc, argv, "--pairs");
                } else if (a == "--max-jaccard") {
                    max_j = std::stod(require_arg(i, argc, argv, "--max-jaccard"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (index.empty() || pairs.empty()) {
                throw std::runtime_error("eval-paraphrase requires --index and --pairs");
            }
            return cmd_eval_paraphrase(index, pairs, max_j);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
