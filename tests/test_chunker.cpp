#include "nanorag/chunker.hpp"
#include "nanorag/chunk_store.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
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
    using namespace nanorag;

    // sanitize
    {
        CHECK(sanitize_chunk_text("  a\tb\nc  ") == "a b c");
        CHECK(sanitize_chunk_text("x\n\ny") == "x y");
    }

    // window
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Window;
        cfg.max_chars = 20;
        cfg.overlap_chars = 5;
        cfg.min_chars = 5;
        const std::string text(100, 'a');  // 100 a's
        auto chunks = chunk_text(text, cfg);
        CHECK(chunks.size() >= 4);
        for (const auto& c : chunks) {
            CHECK(c.text.size() <= 20);
            CHECK(c.text.find('\t') == std::string::npos);
            CHECK(c.text.find('\n') == std::string::npos);
        }
        // sequential ids
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            CHECK(chunks[i].id == static_cast<std::int64_t>(i));
        }
    }

    // paragraph
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Paragraph;
        cfg.max_chars = 200;
        cfg.min_chars = 5;
        const std::string text = "First paragraph about cats.\n\nSecond paragraph about dogs.\n\n";
        auto chunks = chunk_text(text, cfg, "animals");
        CHECK(chunks.size() == 2);
        CHECK(chunks[0].text.find("cats") != std::string::npos);
        CHECK(chunks[1].text.find("dogs") != std::string::npos);
        CHECK(chunks[0].source == "animals");
    }

    // sentence packing
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Sentence;
        cfg.max_chars = 40;
        cfg.overlap_chars = 0;
        cfg.min_chars = 5;
        const std::string text =
            "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four.";
        auto chunks = chunk_text(text, cfg);
        CHECK(chunks.size() >= 2);
        for (const auto& c : chunks) {
            CHECK(c.text.size() <= 50);  // soft pack bound
        }
    }

    // markdown sections
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Markdown;
        cfg.max_chars = 500;
        cfg.min_chars = 5;
        const std::string md =
            "# Intro\nHello world intro text.\n\n## Details\nMore detail about the system.\n";
        auto chunks = chunk_text(md, cfg, "doc.md");
        CHECK(chunks.size() >= 2);
        bool saw_intro = false, saw_details = false;
        for (const auto& c : chunks) {
            if (c.source.find("Intro") != std::string::npos) {
                saw_intro = true;
            }
            if (c.source.find("Details") != std::string::npos) {
                saw_details = true;
            }
            CHECK(c.text.find('\t') == std::string::npos);
        }
        CHECK(saw_intro);
        CHECK(saw_details);
    }

    // round-trip TSV via ChunkStore
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Paragraph;
        cfg.max_chars = 100;
        auto chunks = chunk_text("Hello world paragraph.\n\nSecond one is here.", cfg, "t");
        const auto dir = fs::temp_directory_path() / "nanorag_chunker_test";
        fs::create_directories(dir);
        const auto path = (dir / "chunks.tsv").string();
        write_chunks_tsv(chunks, path);
        auto store = ChunkStore::load(path);
        CHECK(store.size() == chunks.size());
        CHECK(store.get(chunks[0].id).text == chunks[0].text);
    }

    // file path
    {
        const auto dir = fs::temp_directory_path() / "nanorag_chunker_files";
        fs::create_directories(dir);
        const auto f = dir / "sample.txt";
        {
            std::ofstream out(f);
            out << "Sentence one is short. Sentence two continues the story about retrieval.\n\n"
                << "Another paragraph lives here with enough text to survive min_chars.\n";
        }
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Sentence;
        cfg.max_chars = 80;
        cfg.min_chars = 10;
        auto chunks = chunk_path(f.string(), cfg);
        CHECK(!chunks.empty());
        CHECK(chunks[0].source == "sample.txt");
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_chunker OK\n";
    return 0;
}
