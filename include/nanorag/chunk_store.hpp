#pragma once

// Simple id → chunk text store. Phase 0 format: one record per line:
//   <int64 id>\t<source>\t<text...>
// Tab-separated; text may contain spaces but not tabs/newlines.
//
// Negative ids are reserved for system use. The only allowed negative chunk is
// the NO_EVIDENCE sentinel at id -1 (source "system", fixed refuse text).

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nanorag {

/// Must match kNoEvidenceId / kNoEvidenceText in grounding.hpp (kept here so
/// ChunkStore does not pull the full grounding dependency).
inline constexpr std::int64_t kReservedNoEvidenceChunkId = -1;
inline constexpr const char* kReservedNoEvidenceSource = "system";
inline constexpr const char* kReservedNoEvidenceText =
    "NO_EVIDENCE: the corpus contains no answer to this question.";

struct Chunk {
    std::int64_t id = 0;
    std::string source;
    std::string text;
};

inline bool is_system_no_evidence_chunk(const Chunk& c) {
    return c.id == kReservedNoEvidenceChunkId && c.source == kReservedNoEvidenceSource &&
           c.text == kReservedNoEvidenceText;
}

class ChunkStore {
public:
    void clear() { by_id_.clear(); order_.clear(); }

    void add(Chunk c) {
        // Reserve all negative ids; only the official NO_EVIDENCE sentinel may use -1.
        if (c.id < 0) {
            if (!is_system_no_evidence_chunk(c)) {
                throw std::invalid_argument(
                    "ChunkStore: negative chunk ids are reserved for the system NO_EVIDENCE "
                    "sentinel (id=-1, source=system); rejected id " +
                    std::to_string(c.id));
            }
        }
        if (by_id_.count(c.id)) {
            throw std::invalid_argument("ChunkStore: duplicate id " + std::to_string(c.id));
        }
        order_.push_back(c.id);
        by_id_.emplace(c.id, std::move(c));
    }

    bool contains(std::int64_t id) const { return by_id_.count(id) != 0; }

    const Chunk& get(std::int64_t id) const {
        auto it = by_id_.find(id);
        if (it == by_id_.end()) {
            throw std::out_of_range("ChunkStore: missing id " + std::to_string(id));
        }
        return it->second;
    }

    std::size_t size() const { return by_id_.size(); }

    const std::vector<std::int64_t>& ids() const { return order_; }

    std::vector<Chunk> all() const {
        std::vector<Chunk> out;
        out.reserve(order_.size());
        for (auto id : order_) {
            out.push_back(by_id_.at(id));
        }
        return out;
    }

    void save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("ChunkStore::save: cannot open " + path);
        }
        out << "# nanorag chunk store v1 — id\\tsource\\ttext\n";
        for (auto id : order_) {
            const Chunk& c = by_id_.at(id);
            if (c.source.find('\t') != std::string::npos || c.source.find('\n') != std::string::npos ||
                c.text.find('\t') != std::string::npos || c.text.find('\n') != std::string::npos) {
                throw std::runtime_error("ChunkStore::save: tab/newline not allowed in source/text");
            }
            out << c.id << '\t' << c.source << '\t' << c.text << '\n';
        }
    }

    static ChunkStore load(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("ChunkStore::load: cannot open " + path);
        }
        ChunkStore store;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            const auto t1 = line.find('\t');
            if (t1 == std::string::npos) {
                throw std::runtime_error("ChunkStore::load: bad line (no tab): " + line);
            }
            const auto t2 = line.find('\t', t1 + 1);
            if (t2 == std::string::npos) {
                throw std::runtime_error("ChunkStore::load: bad line (need id, source, text): " + line);
            }
            Chunk c;
            c.id = std::stoll(line.substr(0, t1));
            c.source = line.substr(t1 + 1, t2 - t1 - 1);
            c.text = line.substr(t2 + 1);
            store.add(std::move(c));
        }
        return store;
    }

private:
    std::unordered_map<std::int64_t, Chunk> by_id_;
    std::vector<std::int64_t> order_;
};

}  // namespace nanorag
