#pragma once

// Chat-aware prompt packaging for nanorag --mode generate.
// Uses nanollm meta + chat families (raw|chatml|llama3|tinyllama) like the nanollm CLI.
// Exact HF jinja templates still require tools/hf_render_chat.py + --prompt-file (or --allow-approx-chat).

#include "nanorag/grounding.hpp"
#include "nanorag/prompt.hpp"

#include "nanollm/chat.hpp"
#include "nanollm/meta.hpp"
#include "nanollm/runtime.hpp"
#include "nanollm/tokenizer.hpp"
#include "nanollm/types.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {

struct GenerateChatOptions {
    /// Path to model.meta.txt (empty → default_meta_txt_path(model)).
    std::string meta_path;
    /// Override meta chat_family (empty → meta / chatml default).
    std::string chat_family;
    /// Extra system text (appended after grounded rules). Empty keeps rules only.
    std::string system_extra;
    /// If true, append model meta.default_system after grounded rules.
    bool use_model_default_system = false;
    /// Allow approximate C++ chat families when meta has HF jinja / hf_jinja family.
    bool allow_approx_chat = false;
    /// CLI BOS overrides (tri-state via pair of flags in main).
    bool force_no_bos = false;
    bool force_bos = false;
    float temperature = 0.f;
    int max_new_tokens = 64;
    /// Pre-rendered prompt (e.g. from hf_render_chat.py). Non-empty skips chat formatting.
    std::string prompt_override;
};

/// User-turn body: context passages + question (no system rules).
inline std::string build_grounded_user_content(const std::string& question,
                                               const std::vector<RetrievedChunk>& used) {
    std::ostringstream oss;
    oss << "Context:\n";
    if (used.empty()) {
        oss << "(no relevant passages — you must answer: I don't know)\n";
    } else {
        for (const auto& h : used) {
            oss << "[#" << h.id << " source=" << h.source << " score=" << h.score << "]\n";
            oss << h.text << "\n\n";
        }
    }
    oss << "Question: " << question << "\n\n"
        << "Answer with citations like [#id]. If the context does not contain the answer, "
           "reply exactly: I don't know";
    return oss.str();
}

/// System message for grounded generate (rules + optional extras).
inline std::string build_grounded_system_content(const GenerateChatOptions& opt,
                                                 const nanollm::ModelMeta* meta) {
    std::ostringstream oss;
    oss << grounded_system_rules();
    if (opt.use_model_default_system && meta != nullptr && !meta->default_system.empty()) {
        oss << "\nModel default system (secondary):\n" << meta->default_system << "\n";
    }
    if (!opt.system_extra.empty()) {
        oss << "\n" << opt.system_extra << "\n";
    }
    return oss.str();
}

/// Resolve chat family string from options + meta (before fail-closed checks).
inline std::string resolve_chat_family(const GenerateChatOptions& opt,
                                       const nanollm::ModelMeta& meta) {
    if (!opt.chat_family.empty()) {
        return opt.chat_family;
    }
    if (!meta.chat_family.empty()) {
        return meta.chat_family;
    }
    return "chatml";
}

/// Fail-closed policy aligned with nanollm chat CLI (C-1).
inline void enforce_chat_policy(const std::string& family, const nanollm::ModelMeta& meta,
                                bool allow_approx) {
    if (family == "hf_jinja" && !allow_approx) {
        throw std::runtime_error(
            "this model needs the full HF jinja chat_template. "
            "Render with: python3 tools/hf_render_chat.py --meta <model.meta.json> "
            "--user '...' --out p.txt then nanorag ask --mode generate --prompt-file p.txt ... "
            "Or pass --allow-approx-chat --chat-family chatml|llama3|tinyllama.");
    }
    if (meta.has_chat_template && !allow_approx && family != "raw" && family != "chatml" &&
        family != "llama3" && family != "tinyllama") {
        throw std::runtime_error(
            "model meta has_chat_template=1 and chat_family='" + family +
            "' is not a supported C++ approx family. "
            "Use tools/hf_render_chat.py or --allow-approx-chat --chat-family chatml.");
    }
}

/// Build the final string prompt for generate_text.
/// If opt.prompt_override is set, returns it as-is (jinja / external render path).
inline std::string build_generate_prompt(const std::string& question,
                                         const std::vector<RetrievedChunk>& used,
                                         const GenerateChatOptions& opt,
                                         const nanollm::ModelMeta& meta,
                                         std::string* family_out = nullptr) {
    if (!opt.prompt_override.empty()) {
        if (family_out) {
            *family_out = "override";
        }
        return opt.prompt_override;
    }

    std::string family = resolve_chat_family(opt, meta);
    enforce_chat_policy(family, meta, opt.allow_approx_chat);
    if (family == "hf_jinja") {
        // Only reachable with allow_approx_chat.
        family = "chatml";
    }

    const std::string system = build_grounded_system_content(opt, &meta);
    const std::string user = build_grounded_user_content(question, used);

    std::vector<nanollm::ChatMessage> messages;
    if (!system.empty()) {
        messages.push_back({"system", system});
    }
    messages.push_back({"user", user});

    if (family_out) {
        *family_out = family;
    }

    if (family == "raw") {
        // Preserve legacy single-blob layout for raw family.
        return build_grounded_prompt(question, used);
    }
    return nanollm::format_chat(family, messages);
}

/// Load meta: explicit path, else sidecar next to model, else empty defaults.
inline nanollm::ModelMeta load_meta_for_generate(const std::string& model_path,
                                                 const std::string& meta_path_opt) {
    std::string path = meta_path_opt;
    if (path.empty()) {
        path = nanollm::default_meta_txt_path(model_path);
    }
    std::ifstream in(path);
    if (!in) {
        // No sidecar: safe defaults (raw chat, add_bos true, stop = eos only via runtime).
        nanollm::ModelMeta m;
        m.chat_family = "raw";
        m.add_bos = true;
        return m;
    }
    return nanollm::load_meta_txt(path);
}

/// Apply meta + CLI into GenerateConfig (add_bos, stop ids, temperature, max tokens).
inline nanollm::RuntimeOptions make_generate_runtime(const nanollm::ModelMeta& meta,
                                                     const nanollm::Tokenizer& tokenizer,
                                                     const GenerateChatOptions& opt) {
    return nanollm::resolve_runtime(meta, tokenizer, opt.force_no_bos, opt.force_bos,
                                    opt.chat_family, /*cli_extra_stops=*/{},
                                    opt.allow_approx_chat);
}

inline nanollm::GenerateConfig make_generate_config(const nanollm::RuntimeOptions& runtime,
                                                    const GenerateChatOptions& opt) {
    nanollm::GenerateConfig cfg;
    cfg.max_new_tokens = opt.max_new_tokens;
    cfg.sampler.temperature = opt.temperature;
    nanollm::apply_runtime_to_generate(runtime, cfg);
    return cfg;
}

}  // namespace nanorag
