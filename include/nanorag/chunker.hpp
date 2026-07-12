#pragma once

// Text → nanorag chunks.tsv helpers.
//
// Strategies:
//   - paragraph : split on blank lines, then window long paragraphs
//   - sentence  : split on sentence boundaries, pack up to max_chars with overlap
//   - window    : fixed character windows with overlap
//   - markdown  : split on ATx headings, then sentence/window inside sections
//
// Output is always ChunkStore-compatible: id \t source \t text
// (tabs/newlines in text are normalized to spaces).

#include "nanorag/chunk_store.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {

enum class ChunkStrategy {
    Paragraph,
    Sentence,
    Window,
    Markdown,
};

struct ChunkerConfig {
    ChunkStrategy strategy = ChunkStrategy::Paragraph;
    /// Soft max characters per chunk (after whitespace normalize).
    std::size_t max_chars = 800;
    /// Overlap with previous chunk when packing/windowing (characters).
    std::size_t overlap_chars = 100;
    /// First chunk id to assign (subsequent ids are sequential).
    std::int64_t start_id = 0;
    /// Default source label when not taken from path/heading.
    std::string source = "doc";
    /// Prefer breaking windows at word boundaries when possible.
    bool prefer_word_break = true;
    /// Drop chunks shorter than this after packing.
    std::size_t min_chars = 20;
};

inline const char* chunk_strategy_name(ChunkStrategy s) {
    switch (s) {
        case ChunkStrategy::Paragraph:
            return "paragraph";
        case ChunkStrategy::Sentence:
            return "sentence";
        case ChunkStrategy::Window:
            return "window";
        case ChunkStrategy::Markdown:
            return "markdown";
    }
    return "unknown";
}

inline ChunkStrategy parse_chunk_strategy(const std::string& s) {
    if (s == "paragraph" || s == "para") {
        return ChunkStrategy::Paragraph;
    }
    if (s == "sentence" || s == "sent") {
        return ChunkStrategy::Sentence;
    }
    if (s == "window" || s == "fixed") {
        return ChunkStrategy::Window;
    }
    if (s == "markdown" || s == "md") {
        return ChunkStrategy::Markdown;
    }
    throw std::invalid_argument("unknown chunk strategy '" + s +
                                "' (paragraph|sentence|window|markdown)");
}

/// Replace tabs/newlines with spaces and collapse runs of whitespace (ChunkStore TSV safe).
inline std::string sanitize_chunk_text(std::string text) {
    for (char& c : text) {
        if (c == '\t' || c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    std::string out;
    out.reserve(text.size());
    bool sp = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!sp) {
                out.push_back(' ');
                sp = true;
            }
        } else {
            out.push_back(c);
            sp = false;
        }
    }
    while (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

namespace chunker_detail {

inline bool is_sentence_end(char c) {
    return c == '.' || c == '!' || c == '?';
}

/// Byte offset of the UTF-8 sequence start at or before pos (skips continuation bytes),
/// so substr boundaries never split a multi-byte codepoint.
inline std::size_t utf8_floor(const std::string& s, std::size_t pos) {
    while (pos > 0 && pos < s.size() && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return pos;
}

/// Split into sentences (keeps trailing punctuation on the sentence).
inline std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (std::size_t i = 0; i < text.size(); ++i) {
        cur.push_back(text[i]);
        if (is_sentence_end(text[i])) {
            // include following quotes/parens
            while (i + 1 < text.size() &&
                   (text[i + 1] == '"' || text[i + 1] == '\'' || text[i + 1] == ')' ||
                    text[i + 1] == ']')) {
                ++i;
                cur.push_back(text[i]);
            }
            auto s = sanitize_chunk_text(cur);
            if (!s.empty()) {
                out.push_back(std::move(s));
            }
            cur.clear();
        }
    }
    auto tail = sanitize_chunk_text(cur);
    if (!tail.empty()) {
        out.push_back(std::move(tail));
    }
    return out;
}

inline std::vector<std::string> split_paragraphs(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        auto s = sanitize_chunk_text(cur);
        if (!s.empty()) {
            out.push_back(std::move(s));
        }
        cur.clear();
    };
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            // blank line → paragraph break
            std::size_t j = i;
            int newlines = 0;
            while (j < text.size() && (text[j] == '\n' || text[j] == '\r')) {
                if (text[j] == '\n') {
                    ++newlines;
                }
                ++j;
            }
            if (newlines >= 2) {
                flush();
                i = j - 1;
                continue;
            }
            cur.push_back(' ');
            continue;
        }
        if (text[i] == '\r') {
            continue;
        }
        cur.push_back(text[i]);
    }
    flush();
    return out;
}

/// Fixed windows with optional word-boundary snap and overlap.
inline std::vector<std::string> window_text(const std::string& text, const ChunkerConfig& cfg) {
    const auto clean = sanitize_chunk_text(text);
    std::vector<std::string> out;
    if (clean.empty()) {
        return out;
    }
    if (cfg.max_chars == 0) {
        throw std::invalid_argument("ChunkerConfig.max_chars must be > 0");
    }
    std::size_t i = 0;
    while (i < clean.size()) {
        std::size_t end = std::min(i + cfg.max_chars, clean.size());
        if (end < clean.size()) {
            if (cfg.prefer_word_break) {
                std::size_t back = end;
                while (back > i && clean[back] != ' ') {
                    --back;
                }
                if (back > i + cfg.max_chars / 4) {
                    end = back;
                }
            }
            end = utf8_floor(clean, end);
            if (end <= i) {
                // Degenerate (window narrower than one codepoint): take the whole
                // codepoint rather than emit invalid bytes (may exceed max_chars by
                // up to 3 bytes).
                end = i + 1;
                while (end < clean.size() &&
                       (static_cast<unsigned char>(clean[end]) & 0xC0) == 0x80) {
                    ++end;
                }
            }
        }
        auto piece = sanitize_chunk_text(clean.substr(i, end - i));
        // min_chars prunes only the trailing fragment — middle windows are never
        // re-emitted, so dropping them would lose text.
        const bool is_tail = end >= clean.size();
        if (!piece.empty() && (!is_tail || piece.size() >= cfg.min_chars || out.empty())) {
            out.push_back(std::move(piece));
        }
        if (end >= clean.size()) {
            break;
        }
        // Advance from the actual end — a word-break snap may pull `end` well short of
        // the nominal window, and stepping from the nominal window would silently skip
        // the text in between. Keep overlap_chars of context.
        std::size_t next = end > cfg.overlap_chars ? end - cfg.overlap_chars : 0;
        next = utf8_floor(clean, next);
        if (next <= i) {
            next = end;
        }
        i = next;
    }
    return out;
}

/// Pack units into chunks ≤ max_chars with character overlap of prior chunk tail.
inline std::vector<std::string> pack_units(const std::vector<std::string>& units,
                                           const ChunkerConfig& cfg) {
    std::vector<std::string> out;
    if (units.empty()) {
        return out;
    }
    std::string cur;
    auto flush = [&]() {
        auto s = sanitize_chunk_text(cur);
        if (!s.empty() && (s.size() >= cfg.min_chars || out.empty())) {
            out.push_back(std::move(s));
        }
        cur.clear();
    };
    for (const auto& u : units) {
        if (u.size() > cfg.max_chars) {
            // Long unit: flush current, then window the unit. A pending run shorter
            // than min_chars would be dropped by flush — prepend it to the unit so
            // its text survives.
            if (!cur.empty() && cur.size() < cfg.min_chars) {
                std::string combined = cur + " " + u;
                cur.clear();
                for (auto& w : window_text(combined, cfg)) {
                    out.push_back(std::move(w));
                }
                continue;
            }
            flush();
            for (auto& w : window_text(u, cfg)) {
                out.push_back(std::move(w));
            }
            continue;
        }
        if (cur.empty()) {
            cur = u;
            continue;
        }
        if (cur.size() + 1 + u.size() <= cfg.max_chars) {
            cur.push_back(' ');
            cur += u;
        } else {
            flush();
            // Overlap: seed next chunk with tail of previous output.
            if (cfg.overlap_chars > 0 && !out.empty()) {
                const auto& prev = out.back();
                if (prev.size() > cfg.overlap_chars) {
                    const std::size_t seed_start =
                        utf8_floor(prev, prev.size() - cfg.overlap_chars);
                    cur = sanitize_chunk_text(prev.substr(seed_start));
                    if (!cur.empty()) {
                        cur.push_back(' ');
                    }
                }
            }
            cur += u;
            if (cur.size() > cfg.max_chars) {
                // Overlap seed + unit too long: drop seed.
                cur = u;
            }
        }
    }
    flush();
    return out;
}

struct MdSection {
    std::string heading;  // without leading # marks, may be empty for preamble
    std::string body;
};

inline std::vector<MdSection> split_markdown_sections(const std::string& text) {
    std::vector<MdSection> sections;
    MdSection cur;
    std::istringstream in(text);
    std::string line;
    auto flush = [&]() {
        auto body = sanitize_chunk_text(cur.body);
        if (!body.empty() || !cur.heading.empty()) {
            cur.body = std::move(body);
            sections.push_back(std::move(cur));
        }
        cur = MdSection{};
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::size_t hashes = 0;
        while (hashes < line.size() && line[hashes] == '#') {
            ++hashes;
        }
        if (hashes >= 1 && hashes <= 6 && hashes < line.size() && line[hashes] == ' ') {
            flush();
            cur.heading = sanitize_chunk_text(line.substr(hashes + 1));
            continue;
        }
        cur.body += line;
        cur.body.push_back('\n');
    }
    flush();
    if (sections.empty()) {
        sections.push_back({"", sanitize_chunk_text(text)});
    }
    return sections;
}

inline std::string join_source(const std::string& base, const std::string& part) {
    if (part.empty()) {
        return base;
    }
    if (base.empty()) {
        return part;
    }
    return base + "#" + part;
}

}  // namespace chunker_detail

/// Reject reserved / negative user chunk ids early (ChunkStore also enforces).
inline void validate_chunker_start_id(std::int64_t start_id) {
    if (start_id < 0) {
        throw std::invalid_argument(
            "chunker: start_id must be >= 0 (negative ids are reserved for the system "
            "NO_EVIDENCE sentinel at -1)");
    }
}

/// Chunk a single text blob into Chunk records.
inline std::vector<Chunk> chunk_text(const std::string& text, const ChunkerConfig& cfg,
                                     const std::string& source_override = {}) {
    if (cfg.max_chars == 0) {
        throw std::invalid_argument("chunk_text: max_chars must be > 0");
    }
    validate_chunker_start_id(cfg.start_id);
    const std::string src_base = source_override.empty() ? cfg.source : source_override;
    std::vector<std::string> pieces;

    switch (cfg.strategy) {
        case ChunkStrategy::Window:
            pieces = chunker_detail::window_text(text, cfg);
            break;
        case ChunkStrategy::Sentence: {
            auto sents = chunker_detail::split_sentences(text);
            pieces = chunker_detail::pack_units(sents, cfg);
            break;
        }
        case ChunkStrategy::Paragraph: {
            auto paras = chunker_detail::split_paragraphs(text);
            // Each paragraph becomes one or more chunks (window if too long).
            for (const auto& p : paras) {
                if (p.size() <= cfg.max_chars) {
                    if (p.size() >= cfg.min_chars || pieces.empty()) {
                        pieces.push_back(p);
                    }
                } else {
                    for (auto& w : chunker_detail::window_text(p, cfg)) {
                        pieces.push_back(std::move(w));
                    }
                }
            }
            break;
        }
        case ChunkStrategy::Markdown: {
            std::vector<Chunk> out;
            std::int64_t id = cfg.start_id;
            auto sections = chunker_detail::split_markdown_sections(text);
            for (const auto& sec : sections) {
                const std::string sec_src = chunker_detail::join_source(src_base, sec.heading);
                std::string body = sec.body;
                if (!sec.heading.empty()) {
                    // Keep heading context in the chunk text for retrieval.
                    body = sec.heading + ". " + body;
                }
                ChunkerConfig sub = cfg;
                sub.strategy = ChunkStrategy::Sentence;
                sub.source = sec_src;
                sub.start_id = id;
                auto part = chunk_text(body, sub, sec_src);
                for (auto& c : part) {
                    c.id = id++;
                    c.source = sec_src;
                    out.push_back(std::move(c));
                }
            }
            return out;
        }
    }

    std::vector<Chunk> out;
    out.reserve(pieces.size());
    std::int64_t id = cfg.start_id;
    for (auto& p : pieces) {
        if (p.empty()) {
            continue;
        }
        if (p.size() < cfg.min_chars && !out.empty()) {
            // merge tiny leftover into previous when possible
            if (out.back().text.size() + 1 + p.size() <= cfg.max_chars * 2) {
                out.back().text = sanitize_chunk_text(out.back().text + " " + p);
                continue;
            }
        }
        Chunk c;
        c.id = id++;
        c.source = src_base;
        c.text = std::move(p);
        out.push_back(std::move(c));
    }
    return out;
}

/// Read entire file as UTF-8 text (no encoding conversion).
inline std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("read_text_file: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline bool is_text_extension(const std::filesystem::path& p) {
    auto ext = p.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    static const char* ok[] = {".txt", ".md", ".markdown", ".rst", ".text", ".log", ""};
    for (const char* e : ok) {
        if (ext == e) {
            return true;
        }
    }
    return false;
}

/// Chunk one file; source defaults to filename (or relative path).
inline std::vector<Chunk> chunk_file(const std::string& path, ChunkerConfig cfg,
                                     const std::string& source_override = {}) {
    const auto text = read_text_file(path);
    if (source_override.empty() && cfg.source == "doc") {
        cfg.source = std::filesystem::path(path).filename().string();
    }
    // Auto-pick markdown strategy for .md if user left default paragraph on md files?
    // Keep explicit strategy; caller can pass markdown.
    return chunk_text(text, cfg, source_override.empty() ? cfg.source : source_override);
}

/// Chunk a file or recursively all text files under a directory.
inline std::vector<Chunk> chunk_path(const std::string& path, ChunkerConfig cfg) {
    namespace fs = std::filesystem;
    validate_chunker_start_id(cfg.start_id);
    const fs::path p(path);
    if (!fs::exists(p)) {
        throw std::runtime_error("chunk_path: path does not exist: " + path);
    }
    std::vector<Chunk> all;
    std::int64_t next_id = cfg.start_id;

    auto append_file = [&](const fs::path& fp, const std::string& source) {
        ChunkerConfig local = cfg;
        local.start_id = next_id;
        local.source = source;
        // Prefer markdown strategy for .md when strategy is paragraph (convenience).
        auto ext = fp.extension().string();
        for (char& c : ext) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if ((ext == ".md" || ext == ".markdown") && local.strategy == ChunkStrategy::Paragraph) {
            local.strategy = ChunkStrategy::Markdown;
        }
        auto parts = chunk_file(fp.string(), local, source);
        for (auto& c : parts) {
            next_id = c.id + 1;
            all.push_back(std::move(c));
        }
    };

    if (fs::is_regular_file(p)) {
        append_file(p, p.filename().string());
        return all;
    }
    if (!fs::is_directory(p)) {
        throw std::runtime_error("chunk_path: not a file or directory: " + path);
    }
    std::vector<fs::path> files;
    for (auto it = fs::recursive_directory_iterator(p); it != fs::recursive_directory_iterator();
         ++it) {
        if (!it->is_regular_file()) {
            continue;
        }
        if (!is_text_extension(it->path())) {
            continue;
        }
        files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& fp : files) {
        std::string rel = fs::relative(fp, p).generic_string();
        append_file(fp, rel);
    }
    return all;
}

inline ChunkStore chunks_to_store(const std::vector<Chunk>& chunks) {
    ChunkStore store;
    for (const auto& c : chunks) {
        store.add(c);
    }
    return store;
}

inline void write_chunks_tsv(const std::vector<Chunk>& chunks, const std::string& path) {
    chunks_to_store(chunks).save(path);
}

}  // namespace nanorag
