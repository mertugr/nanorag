#pragma once

// Load-once RAG service for HTTPS /ask: index + optional nanollm MultiSeqSession.

#include "nanorag/generate_chat.hpp"
#include "nanorag/grounding.hpp"
#include "nanorag/hybrid.hpp"
#include "nanorag/json_util.hpp"
#include "nanorag/pipeline.hpp"
#include "nanorag/prompt.hpp"

#include "nanollm/generate.hpp"
#include "nanollm/meta.hpp"
#include "nanollm/model.hpp"
#include "nanollm/model_io.hpp"
#include "nanollm/runtime.hpp"
#include "nanollm/sampler.hpp"
#include "nanollm/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {

struct ServeConfig {
    std::string index_dir;
    RetrieveMode retrieve_mode = RetrieveMode::Hybrid;
    float min_score = default_grounding_config().min_score;
    std::size_t default_k = 5;
    /// extractive | generate
    std::string mode = "extractive";
    std::string model_path;
    std::string tokenizer_path;
    GenerateChatOptions gen_opts;
    int max_tokens = 64;
    /// MultiSeqSession slot count (serving).
    nanollm::index_t n_seqs = 1;
};

struct AskJsonResult {
    int http_status = 200;
    std::string body;  // JSON
};

inline std::string chunk_to_json(const RetrievedChunk& c) {
    std::ostringstream oss;
    oss << "{"
        << "\"id\":" << json::number(c.id) << ","
        << "\"score\":" << json::number(c.score) << ","
        << "\"source\":" << json::str(c.source) << ","
        << "\"text\":" << json::str(c.text) << ","
        << "\"score_is_cosine\":" << json::boolean(c.score_is_cosine)
        << "}";
    return oss.str();
}

inline std::string chunks_to_json_array(const std::vector<RetrievedChunk>& chunks) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << chunk_to_json(chunks[i]);
    }
    oss << "]";
    return oss.str();
}

inline std::string ids_to_json_array(const std::vector<std::int64_t>& ids) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << json::number(ids[i]);
    }
    oss << "]";
    return oss.str();
}

/// Build API JSON from a grounded answer. Citations are structured objects, not free text.
inline std::string grounded_to_json(const std::string& query, std::size_t k,
                                    const GroundedAnswer& ga) {
    // Citations: structured objects for each cited id (used → candidates → stub).
    std::vector<RetrievedChunk> citations;
    if (!ga.check.cited_ids.empty()) {
        for (auto id : ga.check.cited_ids) {
            const RetrievedChunk* hit = nullptr;
            for (const auto& u : ga.used) {
                if (u.id == id) {
                    hit = &u;
                    break;
                }
            }
            if (!hit) {
                for (const auto& c : ga.candidates) {
                    if (c.id == id) {
                        hit = &c;
                        break;
                    }
                }
            }
            if (hit) {
                citations.push_back(*hit);
            } else {
                RetrievedChunk stub;
                stub.id = id;
                citations.push_back(stub);
            }
        }
    } else {
        citations = ga.used;
    }

    std::ostringstream oss;
    oss << "{"
        << "\"query\":" << json::str(query) << ","
        << "\"k\":" << json::number(k) << ","
        << "\"answer\":" << json::str(ga.answer) << ","
        << "\"refused\":" << json::boolean(ga.refused) << ","
        << "\"mode\":" << json::str(ga.mode) << ","
        << "\"grounding\":{"
        << "\"ok\":" << json::boolean(ga.check.ok) << ","
        << "\"reason\":" << json::str(ga.check.reason) << ","
        << "\"content_support\":" << json::number(ga.check.content_support) << ","
        << "\"cited_ids\":" << ids_to_json_array(ga.check.cited_ids) << ","
        << "\"illegal_ids\":" << ids_to_json_array(ga.check.illegal_ids)
        << "},"
        << "\"citations\":" << chunks_to_json_array(citations) << ","
        << "\"used\":" << chunks_to_json_array(ga.used) << ","
        << "\"candidates\":" << chunks_to_json_array(ga.candidates)
        << "}";
    return oss.str();
}

inline std::string error_json(const std::string& message, const std::string& query = "",
                              std::size_t k = 0) {
    std::ostringstream oss;
    oss << "{"
        << "\"error\":" << json::str(message) << ","
        << "\"query\":" << json::str(query) << ","
        << "\"k\":" << json::number(k)
        << "}";
    return oss.str();
}

/// Autoregressive generate via nanollm::MultiSeqSession (shared weights, per-seq KV).
inline nanollm::GenerateResult generate_with_multiseq(
    nanollm::MultiSeqSession& session, const nanollm::Tokenizer& tokenizer,
    const std::string& prompt, const nanollm::GenerateConfig& config,
    nanollm::index_t seq_id = 0) {
    if (prompt.empty()) {
        throw std::runtime_error("generate_with_multiseq: empty prompt");
    }
    const auto prompt_tokens = tokenizer.encode(prompt, config.add_bos, /*add_eos=*/false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("generate_with_multiseq: empty tokenized prompt");
    }

    auto is_stop = [&](nanollm::index_t token) {
        if (!config.stop_on_eos) {
            return false;
        }
        if (!config.stop_token_ids.empty()) {
            return std::find(config.stop_token_ids.begin(), config.stop_token_ids.end(), token) !=
                   config.stop_token_ids.end();
        }
        return token == tokenizer.eos_id();
    };

    auto row0 = [](nanollm::Tensor logits) -> nanollm::Tensor {
        if (logits.ndim() == 1) {
            return logits;
        }
        if (logits.ndim() == 2) {
            // decode_batch returns [B, vocab] — take first sequence row as [vocab].
            return logits.slice(0, 0, 1).contiguous().view(nanollm::Shape({logits.size(1)}));
        }
        throw std::runtime_error("generate_with_multiseq: unexpected logits rank");
    };

    nanollm::Sampler sampler(config.sampler);
    nanollm::GenerateResult result;
    result.token_ids = prompt_tokens;
    result.prompt_tokens = static_cast<nanollm::index_t>(prompt_tokens.size());

    nanollm::Tensor last_logits = row0(session.prefill(seq_id, prompt_tokens, /*reset=*/true));

    for (nanollm::index_t step = 0; step < config.max_new_tokens; ++step) {
        const nanollm::index_t next = sampler.sample(last_logits);
        result.token_ids.push_back(next);
        ++result.generated_tokens;
        if (is_stop(next)) {
            break;
        }
        last_logits = row0(session.decode_batch({seq_id}, {next}));
    }
    return result;
}

class RagService {
public:
    explicit RagService(ServeConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.index_dir.empty()) {
            throw std::invalid_argument("RagService: index_dir required");
        }
        nanorag::HybridConfig hcfg;
        hcfg.mode = cfg_.retrieve_mode;
        retriever_ = std::make_unique<Retriever>(Retriever::open(cfg_.index_dir, hcfg));
        gcfg_ = default_grounding_config();
        gcfg_.min_score = cfg_.min_score;

        if (cfg_.mode == "generate") {
            if (cfg_.model_path.empty() || cfg_.tokenizer_path.empty()) {
                throw std::invalid_argument(
                    "RagService: generate mode requires model_path and tokenizer_path");
            }
            load_llm_once();
        } else if (cfg_.mode != "extractive") {
            throw std::invalid_argument("RagService: mode must be extractive|generate");
        }
    }

    const ServeConfig& config() const { return cfg_; }
    const Retriever& retriever() const { return *retriever_; }
    bool has_llm() const { return model_ != nullptr; }

    /// Handle one /ask request body. Index is never reloaded.
    AskJsonResult ask_json(const std::string& body) {
        std::lock_guard<std::mutex> lock(mu_);
        json::AskRequest req;
        try {
            req = json::parse_ask_request(body);
        } catch (const std::exception& e) {
            return {400, error_json(std::string("bad request: ") + e.what())};
        }
        if (!req.has_k) {
            req.k = cfg_.default_k;
        }
        try {
            GroundedAnswer ga = run_ask(req.query, req.k);
            // Cap candidates to requested k for response size.
            if (ga.candidates.size() > req.k) {
                ga.candidates.resize(req.k);
            }
            return {200, grounded_to_json(req.query, req.k, ga)};
        } catch (const std::exception& e) {
            return {500, error_json(std::string("ask failed: ") + e.what(), req.query, req.k)};
        }
    }

    GroundedAnswer run_ask(const std::string& query, std::size_t k) {
        const std::size_t fetch = std::max(k, gcfg_.max_context_chunks * 3);
        auto retrieved = retriever_->retrieve(query, fetch);

        if (cfg_.mode == "extractive" || !model_) {
            return answer_extractive_for_query(query, retrieved.chunks, retriever_->embedder(),
                                               gcfg_);
        }

        // generate mode with load-once MultiSeqSession
        auto used = filter_relevant(retrieved.chunks, query, &retriever_->embedder(), gcfg_);
        if (used.empty()) {
            return answer_extractive_for_query(query, retrieved.chunks, retriever_->embedder(),
                                               gcfg_);
        }

        GenerateChatOptions opts = cfg_.gen_opts;
        opts.max_new_tokens = cfg_.max_tokens;
        std::string family;
        const std::string prompt = build_generate_prompt(query, used, opts, meta_, &family);
        auto runtime = make_generate_runtime(meta_, *tokenizer_, opts);
        auto gencfg = make_generate_config(runtime, opts);

        // Sequential slot 0 for single-threaded serve; MultiSeqSession ready for multi-slot.
        const nanollm::index_t seq_id = 0;
        auto result = generate_with_multiseq(*multi_seq_, *tokenizer_, prompt, gencfg, seq_id);
        multi_seq_->reset(seq_id);
        const std::string gen = nanollm::decode_generated_text(*tokenizer_, result);
        auto ga = finalize_with_model_text(query, retrieved.chunks, gen, &retriever_->embedder(),
                                           gcfg_);
        ga.prompt = prompt;
        return ga;
    }

private:
    void load_llm_once() {
        auto file = nanollm::load_model(cfg_.model_path);
        const auto model_fmt = file.format_version;
        if (file.weight_format == nanollm::WeightFormat::Int8PerColumn) {
            model_ = std::make_unique<nanollm::LlamaModel>(std::move(file.config),
                                                           std::move(file.qweights));
        } else if (file.weight_format == nanollm::WeightFormat::Int4Grouped) {
            model_ = std::make_unique<nanollm::LlamaModel>(std::move(file.config),
                                                           std::move(file.q4weights));
        } else {
            model_ = std::make_unique<nanollm::LlamaModel>(std::move(file.config),
                                                           std::move(file.weights));
        }
        tokenizer_ = std::make_unique<nanollm::Tokenizer>(
            nanollm::Tokenizer::load(cfg_.tokenizer_path));
        if (tokenizer_->vocab_size() != model_->config().vocab_size) {
            throw std::runtime_error("tokenizer vocab_size does not match model vocab_size");
        }
        meta_ = load_meta_for_generate(cfg_.model_path, cfg_.gen_opts.meta_path);
        {
            const auto meta_path = cfg_.gen_opts.meta_path.empty()
                                       ? nanollm::default_meta_txt_path(cfg_.model_path)
                                       : cfg_.gen_opts.meta_path;
            std::ifstream meta_in(meta_path);
            if (meta_in) {
                nanollm::check_artifact_consistency(model_->config(), model_fmt, *tokenizer_,
                                                    tokenizer_->format_version(), meta_);
            }
        }
        const nanollm::index_t n = cfg_.n_seqs > 0 ? cfg_.n_seqs : 1;
        multi_seq_ = std::make_unique<nanollm::MultiSeqSession>(*model_, n);
    }

    ServeConfig cfg_;
    GroundingConfig gcfg_;
    std::unique_ptr<Retriever> retriever_;
    std::unique_ptr<nanollm::LlamaModel> model_;
    std::unique_ptr<nanollm::Tokenizer> tokenizer_;
    std::unique_ptr<nanollm::MultiSeqSession> multi_seq_;
    nanollm::ModelMeta meta_;
    std::mutex mu_;
};

}  // namespace nanorag
