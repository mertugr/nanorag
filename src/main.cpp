// nanorag CLI — owned-stack RAG with grounded answers + citations.

#include "nanorag/chunk_store.hpp"
#include "nanorag/chunker.hpp"
#include "nanorag/contrastive.hpp"
#include "nanorag/embedder.hpp"
#include "nanorag/eval.hpp"
#include "nanorag/generate_chat.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/hybrid.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/prompt.hpp"
#include "nanorag/text_util.hpp"
#include "nanorag/word2vec.hpp"

#include "nanollm/generate.hpp"
#include "nanollm/meta.hpp"
#include "nanollm/model.hpp"
#include "nanollm/model_io.hpp"
#include "nanollm/runtime.hpp"
#include "nanollm/tokenizer.hpp"

#include <algorithm>
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
        << "nanorag — local grounded RAG (tinyann + nanollm + in-house embedders)\n\n"
        << "Usage:\n"
        << "  " << argv0 << " smoke\n"
        << "  " << argv0 << " chunk --input <file|dir> --out <chunks.tsv>\n"
        << "         [--strategy paragraph|sentence|window|markdown]\n"
        << "         [--max-chars N] [--overlap N] [--min-chars N] [--start-id N]\n"
        << "         [--source LABEL]\n"
        << "  " << argv0 << " ingest --chunks <tsv> --out <dir>\n"
        << "         [--embedder contrastive|word2vec|hashing]\n"
        << "         [--pairs <train_pairs.tsv>] [--dim N] [--epochs N]\n"
        << "  " << argv0 << " ask --index <dir> --query <text> [--k N]\n"
        << "         [--retrieve dense|sparse|hybrid]  (default hybrid)\n"
        << "         [--min-score F] [--mode extractive|generate]\n"
        << "         [--model <file.nanollm> --tokenizer <file.nllmtok>] [--max-tokens N]\n"
        << "         generate chat: [--meta <model.meta.txt>] [--chat-family F]\n"
        << "           [--system TEXT] [--use-model-system] [--allow-approx-chat]\n"
        << "           [--no-bos|--bos] [--temperature F] [--prompt-file FILE]\n"
        << "  " << argv0 << " eval-paraphrase --index <dir> --pairs <eval.tsv> [--max-jaccard F]\n"
        << "         [--retrieve dense|sparse|hybrid]\n"
        << "  " << argv0 << " eval-grounding --index <dir> --pairs <eval.tsv>\n"
        << "         [--ood-query <text>]... [--min-score F] [--retrieve dense|sparse|hybrid]\n"
        << "  " << argv0 << " eval-suite --data data/demo [--no-ablations]\n"
        << "         [--retrieve dense|sparse|hybrid]  (default hybrid; report includes A/B)\n";
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
        if (p.gold_id == nanorag::kNoEvidenceId) {
            throw std::runtime_error("eval pairs must not use NO_EVIDENCE id as gold");
        }
        out.push_back(std::move(p));
    }
    return out;
}

void print_grounded(const nanorag::GroundedAnswer& ga) {
    std::cout << "=== candidates (" << ga.candidates.size() << ") ===\n";
    for (const auto& h : ga.candidates) {
        std::cout << "  id=" << h.id << " score=" << h.score << " source=" << h.source << "\n";
    }
    std::cout << "=== used after answerability gate (" << ga.used.size() << ") ===\n";
    for (const auto& h : ga.used) {
        std::cout << "  [#" << h.id << "] score=" << h.score << "\n" << h.text << "\n\n";
    }
    std::cout << "=== answer (mode=" << ga.mode << ", refused=" << (ga.refused ? "yes" : "no")
              << ") ===\n";
    std::cout << ga.answer << "\n";
    std::cout << "=== grounding: " << (ga.check.ok ? "PASS" : "FAIL") << " (" << ga.check.reason
              << ") ===\n";
    if (!ga.check.cited_ids.empty()) {
        std::cout << "citations:";
        for (auto id : ga.check.cited_ids) {
            std::cout << " #" << id;
        }
        std::cout << "\n";
    }
    if (!ga.check.ok) {
        throw std::runtime_error("grounding check failed: " + ga.check.reason);
    }
}

int cmd_smoke() {
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

    std::vector<nanorag::TrainPair> pairs = {
        {"Which small companion mammal makes a soft vibrating sound when relaxed?", 0},
        {"What domestic predator shares living space with humans?", 0},
        {"Which feline housemate vibrates softly when comfortable?", 0},
        {"What animal is famous for short explosive sounds and fetch games?", 1},
        {"What pure C++ toolkit finds neighbors with graph-based approximate search?", 2},
        {"Which project provides hierarchical navigable small world style ANN?", 2},
    };

    nanorag::ContrastiveTrainConfig cfg;
    cfg.dim = 96;
    cfg.epochs = 380;
    cfg.lr = 0.08f;
    cfg.momentum = 0.9f;
    cfg.seed = 42;
    cfg.temperature = 0.05f;
    auto gcfg = nanorag::default_grounding_config();

    // In-domain paraphrase without NO_EVIDENCE (stable across platforms).
    auto ret_in = nanorag::Retriever::build_contrastive(store, pairs, cfg, {},
                                                        /*inject_no_evidence=*/false);

    auto blank = ret_in.ask_grounded("   ", 5, gcfg);
    if (!blank.refused || blank.answer != nanorag::kDontKnowAnswer) {
        std::cerr << "smoke: blank query must refuse\n";
        return 1;
    }

    // Lexical support path (content tokens present in gold) + trained paraphrase.
    auto in_domain =
        ret_in.ask_grounded("Which mammal shares dwellings with people?", 5, gcfg);
    if (in_domain.refused || !in_domain.check.ok) {
        std::cerr << "smoke: expected grounded in-domain answer refused=" << in_domain.refused
                  << " reason=" << in_domain.check.reason << "\n";
        return 1;
    }
    auto cites = nanorag::extract_citation_ids(in_domain.answer);
    if (std::find(cites.begin(), cites.end(), 0) == cites.end()) {
        std::cerr << "smoke: expected citation [#0] answer=" << in_domain.answer << "\n";
        return 1;
    }

    // Near-miss / OOD with NO_EVIDENCE injection (production-like ingest).
    auto ret_ood = nanorag::Retriever::build_contrastive(store, pairs, cfg, {},
                                                         /*inject_no_evidence=*/true);
    auto alcohol = ret_ood.ask_grounded("What is the boiling point of alcohol?", 5, gcfg);
    if (!alcohol.refused || alcohol.answer != nanorag::kDontKnowAnswer || !alcohol.check.ok) {
        std::cerr << "smoke: alcohol near-miss must refuse\n";
        return 1;
    }

    auto ood = ret_ood.ask_grounded("Who invented the chocolate pizza telescope?", 5, gcfg);
    if (!ood.refused || ood.answer != nanorag::kDontKnowAnswer || !ood.check.ok) {
        std::cerr << "smoke: expected I don't know for OOD query\n";
        return 1;
    }

    std::cout << "nanorag smoke OK (blank/alcohol refuse + in-domain cite)\n";
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
                throw std::runtime_error("ingest: contrastive requires --pairs");
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
    std::cout << "ingested real_chunks=" << retriever.real_size()
              << " store_rows=" << retriever.size() << " → " << out_dir << "\n";
    std::cout << "  embedder=" << retriever.embedder().id() << " dim=" << retriever.embedder().dim()
              << "\n";
    return 0;
}

int cmd_ask(const std::string& index_dir, const std::string& query, std::size_t k, float min_score,
            const std::string& mode, const std::string& model_path, const std::string& tok_path,
            int max_tokens, nanorag::RetrieveMode retrieve_mode,
            const nanorag::GenerateChatOptions& gen_opts) {
    nanorag::HybridConfig hcfg;
    hcfg.mode = retrieve_mode;
    auto retriever = nanorag::Retriever::open(index_dir, hcfg);
    auto gcfg = nanorag::default_grounding_config();
    gcfg.min_score = min_score;

    const std::size_t fetch = std::max(k, gcfg.max_context_chunks * 3);
    auto retrieved = retriever.retrieve(query, fetch);
    std::cout << "retrieve_mode=" << nanorag::retrieve_mode_name(retrieve_mode) << "\n";

    nanorag::GroundedAnswer ga;
    if (mode == "extractive" || model_path.empty()) {
        ga = nanorag::answer_extractive_for_query(query, retrieved.chunks, retriever.embedder(),
                                                  gcfg);
    } else if (mode == "generate") {
        if (tok_path.empty() && gen_opts.prompt_override.empty()) {
            // tokenizer required unless only testing packaging (still need model+tok for generate)
        }
        if (tok_path.empty()) {
            throw std::runtime_error("ask --mode generate requires --tokenizer");
        }
        auto used = nanorag::filter_relevant(retrieved.chunks, query, &retriever.embedder(), gcfg);
        if (used.empty()) {
            ga = nanorag::answer_extractive_for_query(query, retrieved.chunks, retriever.embedder(),
                                                     gcfg);
        } else {
            auto file = nanollm::load_model(model_path);
            const auto model_fmt = file.format_version;
            nanollm::LlamaModel model = [&]() {
                if (file.weight_format == nanollm::WeightFormat::Int8PerColumn) {
                    return nanollm::LlamaModel(std::move(file.config), std::move(file.qweights));
                }
                if (file.weight_format == nanollm::WeightFormat::Int4Grouped) {
                    return nanollm::LlamaModel(std::move(file.config), std::move(file.q4weights));
                }
                return nanollm::LlamaModel(std::move(file.config), std::move(file.weights));
            }();
            auto tokenizer = nanollm::Tokenizer::load(tok_path);
            if (tokenizer.vocab_size() != model.config().vocab_size) {
                throw std::runtime_error("tokenizer vocab_size does not match model vocab_size");
            }
            auto meta = nanorag::load_meta_for_generate(model_path, gen_opts.meta_path);
            // Fail-closed when a meta sidecar exists (same policy as nanollm CLI).
            {
                const auto meta_path = gen_opts.meta_path.empty()
                                           ? nanollm::default_meta_txt_path(model_path)
                                           : gen_opts.meta_path;
                std::ifstream meta_in(meta_path);
                if (meta_in) {
                    nanollm::check_artifact_consistency(model.config(), model_fmt, tokenizer,
                                                        tokenizer.format_version(), meta);
                }
            }

            nanorag::GenerateChatOptions opts = gen_opts;
            opts.max_new_tokens = max_tokens;
            std::string family;
            const std::string prompt =
                nanorag::build_generate_prompt(query, used, opts, meta, &family);
            auto runtime = nanorag::make_generate_runtime(meta, tokenizer, opts);
            auto cfg = nanorag::make_generate_config(runtime, opts);

            std::cout << "generate chat_family=" << family << " add_bos=" << (cfg.add_bos ? 1 : 0)
                      << " stop_ids=" << cfg.stop_token_ids.size()
                      << " temperature=" << cfg.sampler.temperature << "\n";

            auto result = nanollm::generate_text(model, tokenizer, prompt, cfg);
            const std::string gen = nanollm::decode_generated_text(tokenizer, result);
            ga = nanorag::finalize_with_model_text(query, retrieved.chunks, gen,
                                                   &retriever.embedder(), gcfg);
            // Keep chat-formatted prompt on the answer for debugging / eval.
            ga.prompt = prompt;
        }
    } else {
        throw std::runtime_error("ask: unknown --mode (extractive|generate)");
    }

    if (ga.candidates.size() > k) {
        ga.candidates.resize(k);
    }
    print_grounded(ga);
    return 0;
}

int cmd_eval_paraphrase(const std::string& index_dir, const std::string& pairs_path,
                        double max_jaccard, nanorag::RetrieveMode retrieve_mode) {
    nanorag::HybridConfig hcfg;
    hcfg.mode = retrieve_mode;
    auto retriever = nanorag::Retriever::open(index_dir, hcfg);
    auto pairs = load_eval_pairs(pairs_path);
    int ok = 0;
    int n = 0;
    for (const auto& p : pairs) {
        const auto& gold = retriever.store().get(p.gold_id);
        const double j = nanorag::token_jaccard(p.query, gold.text);
        if (j > max_jaccard) {
            std::cerr << "FAIL overlap: jaccard=" << j << " q=" << p.query << "\n";
            return 1;
        }
        // Skip NO_EVIDENCE when scoring retrieval quality among real docs.
        const auto top = retriever.top_real_id(p.query, /*k=*/8);
        const bool hit = top == p.gold_id;
        std::cout << (hit ? "OK " : "MISS ") << "gold=" << p.gold_id << " top_real=" << top
                  << " jaccard=" << j << "\n";
        if (hit) {
            ++ok;
        }
        ++n;
    }
    std::cout << "paraphrase recall@1 (real docs only): " << ok << "/" << n << "\n";
    return ok == n ? 0 : 1;
}

int cmd_eval_grounding(const std::string& index_dir, const std::string& pairs_path,
                       const std::vector<std::string>& ood_queries, float min_score,
                       nanorag::RetrieveMode retrieve_mode) {
    nanorag::HybridConfig hcfg;
    hcfg.mode = retrieve_mode;
    auto retriever = nanorag::Retriever::open(index_dir, hcfg);
    auto gcfg = nanorag::default_grounding_config();
    gcfg.min_score = min_score;

    int n = 0;
    int ok = 0;

    if (!pairs_path.empty()) {
        for (const auto& p : load_eval_pairs(pairs_path)) {
            auto ga = retriever.ask_grounded(p.query, 5, gcfg);
            const auto cites = nanorag::extract_citation_ids(ga.answer);
            const bool has_gold_cite =
                std::find(cites.begin(), cites.end(), p.gold_id) != cites.end();
            const bool pass = !ga.refused && ga.check.ok && has_gold_cite;
            std::cout << (pass ? "OK  " : "FAIL") << " in-domain gold=#" << p.gold_id
                      << " refused=" << ga.refused << " cites=";
            for (auto id : cites) {
                std::cout << "#" << id << ",";
            }
            std::cout << " support=" << ga.check.content_support << " q=" << p.query << "\n";
            if (!pass) {
                std::cout << "  answer: " << ga.answer << "\n"
                          << "  reason: " << ga.check.reason << "\n";
            }
            ++n;
            if (pass) {
                ++ok;
            }
        }
    }

    std::vector<std::string> oods = ood_queries;
    if (oods.empty()) {
        oods = {
            "What is the boiling point of alcohol?",
            "What is the boiling point of ethanol?",
            "What is the melting point of iron?",
            "What is the freezing point of mercury?",
            "How many legs does a spider have?",
            "What is the capital of Germany?",
            "Who is the president of France?",
            "What is the chemical formula for methane?",
            "Who invented the chocolate pizza telescope in medieval France?",
            "What is the stock price of completely fictional company Zyblerqux?",
        };
    }
    for (const auto& q : oods) {
        auto ga = retriever.ask_grounded(q, 5, gcfg);
        const bool pass =
            ga.refused && ga.check.ok && ga.answer == nanorag::kDontKnowAnswer && ga.used.empty();
        std::cout << (pass ? "OK  " : "FAIL") << " ood refuse q=" << q << "\n";
        if (!pass) {
            std::cout << "  answer: " << ga.answer << " used=" << ga.used.size() << "\n";
        }
        ++n;
        if (pass) {
            ++ok;
        }
    }

    // Blank query
    {
        auto ga = retriever.ask_grounded("", 5, gcfg);
        const bool pass = ga.refused && ga.check.ok && ga.answer == nanorag::kDontKnowAnswer;
        std::cout << (pass ? "OK  " : "FAIL") << " blank query refuse\n";
        ++n;
        if (pass) {
            ++ok;
        }
    }

    // Validator negatives
    {
        std::vector<nanorag::RetrievedChunk> used = {
            {7, 0.9f, "t", "The moon is Earth's only major natural satellite."}};
        auto bad_cite = nanorag::validate_grounding("The moon is cheese [#99]", used, gcfg);
        auto no_cite = nanorag::validate_grounding(
            "The moon is Earth's only major natural satellite.", used, gcfg);
        auto good = nanorag::validate_grounding(
            "The moon is Earth's only major natural satellite. [#7]", used, gcfg);
        auto empty_ok = nanorag::validate_grounding(nanorag::kDontKnowAnswer, {}, gcfg);
        auto empty_bad = nanorag::validate_grounding("The moon is cheese.", {}, gcfg);
        const bool pass = !bad_cite.ok && !no_cite.ok && good.ok && empty_ok.ok && !empty_bad.ok;
        std::cout << (pass ? "OK  " : "FAIL") << " validator negatives\n";
        ++n;
        if (pass) {
            ++ok;
        }
    }

    std::cout << "grounding eval: " << ok << "/" << n << "\n";
    return ok == n ? 0 : 1;
}


int cmd_chunk(const std::string& input_path, const std::string& out_path,
              nanorag::ChunkerConfig cfg) {
    auto chunks = nanorag::chunk_path(input_path, cfg);
    if (chunks.empty()) {
        throw std::runtime_error("chunk: no chunks produced from " + input_path);
    }
    // Ensure parent directory exists.
    {
        fs::path outp(out_path);
        if (outp.has_parent_path() && !outp.parent_path().empty()) {
            fs::create_directories(outp.parent_path());
        }
    }
    nanorag::write_chunks_tsv(chunks, out_path);
    std::cout << "chunk: wrote " << chunks.size() << " chunks → " << out_path
              << " strategy=" << nanorag::chunk_strategy_name(cfg.strategy)
              << " max_chars=" << cfg.max_chars << " overlap=" << cfg.overlap_chars << "\n";
    // Preview first few
    const std::size_t preview = std::min<std::size_t>(3, chunks.size());
    for (std::size_t i = 0; i < preview; ++i) {
        const auto& c = chunks[i];
        std::string t = c.text;
        if (t.size() > 100) {
            t = t.substr(0, 97) + "...";
        }
        std::cout << "  [" << c.id << "] source=" << c.source << " chars=" << c.text.size()
                  << " " << t << "\n";
    }
    return 0;
}

int cmd_eval_suite(const std::string& data_root, bool ablations,
                   nanorag::RetrieveMode retrieve_mode) {
    auto paths = nanorag::eval::default_demo_paths(data_root);
    nanorag::ContrastiveTrainConfig cfg;
    cfg.dim = 128;
    cfg.epochs = 420;
    cfg.lr = 0.08f;
    cfg.momentum = 0.9f;
    cfg.temperature = 0.05f;
    cfg.seed = 7;
    auto primary = nanorag::eval::build_contrastive_from_paths(paths, cfg, true);
    primary.set_retrieve_mode(retrieve_mode);
    auto report = nanorag::eval::run_full_eval(std::move(primary), paths,
                                               nanorag::default_grounding_config(), ablations);
    std::cout << nanorag::eval::format_report(report);
    const std::string err = nanorag::eval::check_gates(report);
    if (!err.empty()) {
        std::cerr << err;
        return 1;
    }
    std::cout << "eval-suite: all quality gates PASS\n";
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
        if (cmd == "chunk") {
            std::string input, out;
            nanorag::ChunkerConfig cfg;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--input" || a == "-i") {
                    input = require_arg(i, argc, argv, "--input");
                } else if (a == "--out" || a == "-o") {
                    out = require_arg(i, argc, argv, "--out");
                } else if (a == "--strategy") {
                    cfg.strategy =
                        nanorag::parse_chunk_strategy(require_arg(i, argc, argv, "--strategy"));
                } else if (a == "--max-chars") {
                    cfg.max_chars =
                        static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--max-chars")));
                } else if (a == "--overlap") {
                    cfg.overlap_chars =
                        static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--overlap")));
                } else if (a == "--min-chars") {
                    cfg.min_chars =
                        static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--min-chars")));
                } else if (a == "--start-id") {
                    cfg.start_id = std::stoll(require_arg(i, argc, argv, "--start-id"));
                } else if (a == "--source") {
                    cfg.source = require_arg(i, argc, argv, "--source");
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (input.empty() || out.empty()) {
                throw std::runtime_error("chunk requires --input and --out");
            }
            return cmd_chunk(input, out, cfg);
        }
        if (cmd == "ingest") {
            std::string chunks, out, pairs;
            std::size_t dim = 128;
            std::string embedder = "contrastive";
            int epochs = 420;
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
            std::string index, query, model, tok, mode = "extractive";
            std::size_t k = 5;
            float min_score = nanorag::default_grounding_config().min_score;
            int max_tokens = 64;
            nanorag::RetrieveMode retrieve_mode = nanorag::RetrieveMode::Hybrid;
            nanorag::GenerateChatOptions gen_opts;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--index") {
                    index = require_arg(i, argc, argv, "--index");
                } else if (a == "--query" || a == "-q") {
                    query = require_arg(i, argc, argv, "--query");
                } else if (a == "--k") {
                    k = static_cast<std::size_t>(std::stoull(require_arg(i, argc, argv, "--k")));
                } else if (a == "--min-score") {
                    min_score = std::stof(require_arg(i, argc, argv, "--min-score"));
                } else if (a == "--mode") {
                    mode = require_arg(i, argc, argv, "--mode");
                } else if (a == "--retrieve") {
                    retrieve_mode = nanorag::parse_retrieve_mode(require_arg(i, argc, argv, "--retrieve"));
                } else if (a == "--model") {
                    model = require_arg(i, argc, argv, "--model");
                    if (mode == "extractive") {
                        mode = "generate";
                    }
                } else if (a == "--tokenizer") {
                    tok = require_arg(i, argc, argv, "--tokenizer");
                } else if (a == "--max-tokens") {
                    max_tokens = std::stoi(require_arg(i, argc, argv, "--max-tokens"));
                } else if (a == "--meta") {
                    gen_opts.meta_path = require_arg(i, argc, argv, "--meta");
                } else if (a == "--chat-family") {
                    gen_opts.chat_family = require_arg(i, argc, argv, "--chat-family");
                } else if (a == "--system") {
                    gen_opts.system_extra = require_arg(i, argc, argv, "--system");
                } else if (a == "--use-model-system") {
                    gen_opts.use_model_default_system = true;
                } else if (a == "--allow-approx-chat") {
                    gen_opts.allow_approx_chat = true;
                } else if (a == "--no-bos") {
                    gen_opts.force_no_bos = true;
                } else if (a == "--bos") {
                    gen_opts.force_bos = true;
                } else if (a == "--temperature") {
                    gen_opts.temperature = std::stof(require_arg(i, argc, argv, "--temperature"));
                } else if (a == "--prompt-file") {
                    const std::string pf = require_arg(i, argc, argv, "--prompt-file");
                    std::ifstream in(pf);
                    if (!in) {
                        throw std::runtime_error("cannot open --prompt-file " + pf);
                    }
                    std::ostringstream ss;
                    ss << in.rdbuf();
                    gen_opts.prompt_override = ss.str();
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (index.empty()) {
                throw std::runtime_error("ask requires --index");
            }
            if (query.empty() && mode != "extractive") {
                // allow empty for refuse path testing only via explicit -q ""
            }
            // Require --query flag was provided: empty string is a valid refuse test if passed.
            bool has_query = false;
            for (int i = 2; i < argc; ++i) {
                if (std::string(argv[i]) == "--query" || std::string(argv[i]) == "-q") {
                    has_query = true;
                }
            }
            if (!has_query) {
                throw std::runtime_error("ask requires --query");
            }
            return cmd_ask(index, query, k, min_score, mode, model, tok, max_tokens, retrieve_mode,
                           gen_opts);
        }
        if (cmd == "eval-paraphrase") {
            std::string index, pairs;
            double max_j = 0.30;
            nanorag::RetrieveMode retrieve_mode = nanorag::RetrieveMode::Hybrid;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--index") {
                    index = require_arg(i, argc, argv, "--index");
                } else if (a == "--pairs") {
                    pairs = require_arg(i, argc, argv, "--pairs");
                } else if (a == "--max-jaccard") {
                    max_j = std::stod(require_arg(i, argc, argv, "--max-jaccard"));
                } else if (a == "--retrieve") {
                    retrieve_mode = nanorag::parse_retrieve_mode(require_arg(i, argc, argv, "--retrieve"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (index.empty() || pairs.empty()) {
                throw std::runtime_error("eval-paraphrase requires --index and --pairs");
            }
            return cmd_eval_paraphrase(index, pairs, max_j, retrieve_mode);
        }
        if (cmd == "eval-suite") {
            std::string data = "data/demo";
            bool ablations = true;
            nanorag::RetrieveMode retrieve_mode = nanorag::RetrieveMode::Hybrid;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--data") {
                    data = require_arg(i, argc, argv, "--data");
                } else if (a == "--no-ablations") {
                    ablations = false;
                } else if (a == "--retrieve") {
                    retrieve_mode = nanorag::parse_retrieve_mode(require_arg(i, argc, argv, "--retrieve"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            return cmd_eval_suite(data, ablations, retrieve_mode);
        }
        if (cmd == "eval-grounding") {
            std::string index, pairs;
            float min_score = nanorag::default_grounding_config().min_score;
            std::vector<std::string> oods;
            nanorag::RetrieveMode retrieve_mode = nanorag::RetrieveMode::Hybrid;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--index") {
                    index = require_arg(i, argc, argv, "--index");
                } else if (a == "--pairs") {
                    pairs = require_arg(i, argc, argv, "--pairs");
                } else if (a == "--min-score") {
                    min_score = std::stof(require_arg(i, argc, argv, "--min-score"));
                } else if (a == "--ood-query") {
                    oods.push_back(require_arg(i, argc, argv, "--ood-query"));
                } else if (a == "--retrieve") {
                    retrieve_mode = nanorag::parse_retrieve_mode(require_arg(i, argc, argv, "--retrieve"));
                } else {
                    throw std::runtime_error("unknown arg: " + a);
                }
            }
            if (index.empty()) {
                throw std::runtime_error("eval-grounding requires --index");
            }
            return cmd_eval_grounding(index, pairs, oods, min_score, retrieve_mode);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
