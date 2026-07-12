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

static bool valid_utf8(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = 0;
        if (c < 0x80) {
            len = 1;
        } else if ((c >> 5) == 0x6) {
            len = 2;
        } else if ((c >> 4) == 0xE) {
            len = 3;
        } else if ((c >> 3) == 0x1E) {
            len = 4;
        } else {
            return false;
        }
        if (i + len > s.size()) {
            return false;
        }
        for (std::size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) {
                return false;
            }
        }
        i += len;
    }
    return true;
}

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

    // window: word-break snap must not skip text (long unbroken token)
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Window;
        cfg.max_chars = 800;
        cfg.overlap_chars = 100;
        cfg.min_chars = 5;
        std::string text;
        for (int i = 0; i < 50; ++i) {
            text += "word ";  // 250 chars of breakable words
        }
        unsigned x = 12345;  // non-repeating 700-char unbroken token
        for (int i = 0; i < 700; ++i) {
            x = x * 1103515245u + 12345u;
            text += static_cast<char>('a' + (x >> 16) % 26);
        }
        text += " closing words after the long token to end the document cleanly";
        const std::string clean = sanitize_chunk_text(text);
        auto chunks = chunk_text(text, cfg);
        CHECK(!chunks.empty());
        // Every 25-char span of the input must land in some chunk (no silent gaps).
        bool all_covered = true;
        for (std::size_t p = 0; p + 25 <= clean.size(); p += 10) {
            const std::string probe = clean.substr(p, 25);
            bool found = false;
            for (const auto& c : chunks) {
                if (c.text.find(probe) != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                all_covered = false;
                break;
            }
        }
        CHECK(all_covered);
    }

    // window boundaries must not split multi-byte UTF-8 codepoints
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Window;
        cfg.max_chars = 21;  // odd byte count → would land mid-codepoint
        cfg.overlap_chars = 4;
        cfg.min_chars = 1;
        cfg.prefer_word_break = false;
        std::string text;
        for (int i = 0; i < 60; ++i) {
            text += "\xc3\xa7\xc3\xb6\xc3\xbc";  // çöü — 2-byte codepoints
        }
        auto chunks = chunk_text(text, cfg);
        CHECK(!chunks.empty());
        for (const auto& c : chunks) {
            CHECK(valid_utf8(c.text));
        }
    }

    // sentence-pack overlap seed must not split multi-byte codepoints
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Sentence;
        cfg.max_chars = 40;
        cfg.overlap_chars = 7;
        cfg.min_chars = 1;
        const std::string text =
            "Çölde güneş öğlen çok sıcaktır. Gece rüzgârı serin eser. "
            "Üzüm bağları öğleden sonra gölgelenir. Şölen akşamı görkemliydi.";
        auto chunks = chunk_text(text, cfg);
        CHECK(!chunks.empty());
        for (const auto& c : chunks) {
            CHECK(valid_utf8(c.text));
        }
    }

    // degenerate window narrower than one codepoint: whole codepoint, valid UTF-8
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Window;
        cfg.max_chars = 1;  // narrower than a 2-byte codepoint
        cfg.overlap_chars = 0;
        cfg.min_chars = 1;
        cfg.prefer_word_break = false;
        std::string text;
        for (int i = 0; i < 12; ++i) {
            text += "\xc3\xa7";  // ç
        }
        auto chunks = chunk_text(text, cfg);
        CHECK(!chunks.empty());
        for (const auto& c : chunks) {
            CHECK(valid_utf8(c.text));
        }
    }

    // middle window pieces are kept even when shorter than min_chars
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Window;
        cfg.max_chars = 20;
        cfg.overlap_chars = 0;
        cfg.min_chars = 10;
        std::string text = std::string(20, 'A') + " qumquat " + std::string(30, 'B') +
                           " tail ending is long enough";
        auto chunks = chunk_text(text, cfg);
        bool has_qumquat = false;
        for (const auto& c : chunks) {
            if (c.text.find("qumquat") != std::string::npos) {
                has_qumquat = true;
            }
        }
        CHECK(has_qumquat);
    }

    // short pending sentence before a long unit survives packing
    {
        ChunkerConfig cfg;
        cfg.strategy = ChunkStrategy::Sentence;
        cfg.max_chars = 30;
        cfg.overlap_chars = 0;
        cfg.min_chars = 12;
        const std::string text =
            "A first proper sentence here. Tiny bit. "
            "This single sentence is deliberately much longer than the thirty "
            "character packing limit so it gets windowed.";
        auto chunks = chunk_text(text, cfg);
        bool has_tiny = false;
        for (const auto& c : chunks) {
            if (c.text.find("Tiny bit") != std::string::npos) {
                has_tiny = true;
            }
        }
        CHECK(has_tiny);
    }

    if (g_fails) {
        std::cerr << g_fails << " failure(s)\n";
        return 1;
    }
    std::cout << "test_chunker OK\n";
    return 0;
}
