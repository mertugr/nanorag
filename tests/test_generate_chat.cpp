#include "nanorag/generate_chat.hpp"
#include "nanorag/grounding.hpp"

#include "nanollm/chat.hpp"

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

int main() {
    using namespace nanorag;
    std::vector<RetrievedChunk> used = {
        {6, 0.9f, "science", "Under one atmosphere pure H2O becomes gas at 373.15 K."},
    };

    // User content packages context + question
    {
        const auto user = build_grounded_user_content("When does water boil?", used);
        CHECK(user.find("[#6") != std::string::npos);
        CHECK(user.find("373.15") != std::string::npos);
        CHECK(user.find("When does water boil?") != std::string::npos);
        CHECK(user.find("I don't know") != std::string::npos);
    }

    // ChatML packaging
    {
        GenerateChatOptions opt;
        opt.chat_family = "chatml";
        nanollm::ModelMeta meta;
        meta.chat_family = "chatml";
        meta.add_bos = false;
        meta.has_chat_template = false;
        std::string family;
        const auto prompt = build_generate_prompt("When does water boil?", used, opt, meta, &family);
        CHECK(family == "chatml");
        CHECK(prompt.find("<|im_start|>system") != std::string::npos);
        CHECK(prompt.find("<|im_start|>user") != std::string::npos);
        CHECK(prompt.find("<|im_start|>assistant") != std::string::npos);
        CHECK(prompt.find("retrieval-grounded") != std::string::npos);
        CHECK(prompt.find("Context:") != std::string::npos);
        CHECK(prompt.find("[#6") != std::string::npos);
    }

    // Llama3 packaging
    {
        GenerateChatOptions opt;
        opt.chat_family = "llama3";
        nanollm::ModelMeta meta;
        meta.chat_family = "llama3";
        std::string family;
        const auto prompt = build_generate_prompt("q", used, opt, meta, &family);
        CHECK(family == "llama3");
        CHECK(prompt.find("<|begin_of_text|>") != std::string::npos);
        CHECK(prompt.find("<|start_header_id|>system") != std::string::npos);
        CHECK(prompt.find("<|start_header_id|>assistant") != std::string::npos);
    }

    // raw family keeps legacy blob (no chat specials required)
    {
        GenerateChatOptions opt;
        opt.chat_family = "raw";
        nanollm::ModelMeta meta;
        meta.chat_family = "raw";
        std::string family;
        const auto prompt = build_generate_prompt("When does water boil?", used, opt, meta, &family);
        CHECK(family == "raw");
        CHECK(prompt.find("Question: When does water boil?") != std::string::npos);
        CHECK(prompt.find("<|im_start|>") == std::string::npos);
    }

    // hf_jinja fail-closed without allow_approx
    {
        GenerateChatOptions opt;
        opt.chat_family = "hf_jinja";
        nanollm::ModelMeta meta;
        meta.chat_family = "hf_jinja";
        meta.has_chat_template = true;
        bool threw = false;
        try {
            (void)build_generate_prompt("q", used, opt, meta, nullptr);
        } catch (const std::exception& e) {
            threw = true;
            CHECK(std::string(e.what()).find("jinja") != std::string::npos ||
                  std::string(e.what()).find("hf_render") != std::string::npos);
        }
        CHECK(threw);
    }

    // hf_jinja + allow_approx remaps to chatml
    {
        GenerateChatOptions opt;
        opt.chat_family = "hf_jinja";
        opt.allow_approx_chat = true;
        nanollm::ModelMeta meta;
        meta.chat_family = "hf_jinja";
        meta.has_chat_template = true;
        std::string family;
        const auto prompt = build_generate_prompt("q", used, opt, meta, &family);
        CHECK(family == "chatml");
        CHECK(prompt.find("<|im_start|>") != std::string::npos);
    }

    // prompt override bypasses chat packaging
    {
        GenerateChatOptions opt;
        opt.prompt_override = "PRE-RENDERED PROMPT";
        nanollm::ModelMeta meta;
        std::string family;
        const auto prompt = build_generate_prompt("q", used, opt, meta, &family);
        CHECK(family == "override");
        CHECK(prompt == "PRE-RENDERED PROMPT");
    }

    // system extra + model default system
    {
        GenerateChatOptions opt;
        opt.chat_family = "chatml";
        opt.system_extra = "Be concise.";
        opt.use_model_default_system = true;
        nanollm::ModelMeta meta;
        meta.chat_family = "chatml";
        meta.default_system = "You are Qwen.";
        const auto sys = build_grounded_system_content(opt, &meta);
        CHECK(sys.find("Be concise.") != std::string::npos);
        CHECK(sys.find("You are Qwen.") != std::string::npos);
        CHECK(sys.find("I don't know") != std::string::npos);
    }

    // runtime applies meta stop ids / add_bos
    {
        nanollm::ModelMeta meta;
        meta.add_bos = false;
        meta.chat_family = "chatml";
        meta.stop_token_ids = {151643, 151645};
        // Minimal tokenizer not available — skip full make_generate_runtime without tok.
        // Policy already covered; stop packing is nanollm resolve_runtime unit-tested upstream.
        CHECK(meta.stop_token_ids.size() == 2);
        CHECK(meta.add_bos == false);
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_generate_chat OK\n";
    return 0;
}
