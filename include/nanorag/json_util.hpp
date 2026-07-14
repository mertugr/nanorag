#pragma once

// Minimal JSON helpers for nanorag HTTPS /ask (no external JSON library).

#include <cctype>
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {
namespace json {

inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

inline std::string str(const std::string& s) { return "\"" + escape(s) + "\""; }

inline std::string boolean(bool v) { return v ? "true" : "false"; }

inline std::string number(double v) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);
    oss << v;
    std::string s = oss.str();
    // trim trailing zeros after decimal
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') {
            s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
    }
    if (s.empty() || s == "-") {
        return "0";
    }
    return s;
}

inline std::string number(std::int64_t v) { return std::to_string(v); }

inline std::string number(std::size_t v) { return std::to_string(v); }

inline std::string number(int v) { return std::to_string(v); }

inline std::string number(float v) { return number(static_cast<double>(v)); }

// --- tiny pull parser for {"query":"...","k":N} style objects ---

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

inline std::string parse_string(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') {
        throw std::runtime_error("json: expected string");
    }
    ++i;
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (i >= s.size()) {
                throw std::runtime_error("json: bad escape");
            }
            char e = s[i++];
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(e);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    if (i + 4 > s.size()) {
                        throw std::runtime_error("json: bad unicode escape");
                    }
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') {
                            code |= static_cast<unsigned>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            code |= static_cast<unsigned>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            code |= static_cast<unsigned>(h - 'A' + 10);
                        } else {
                            throw std::runtime_error("json: bad unicode escape");
                        }
                    }
                    if (code < 0x80) {
                        out.push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    throw std::runtime_error("json: unsupported escape");
            }
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("json: unterminated string");
}

inline double parse_number(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    std::size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
        ++i;
    }
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                            s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
        // Allow scientific notation digits after e/E
        if ((s[i] == '+' || s[i] == '-') && i > start && s[i - 1] != 'e' && s[i - 1] != 'E') {
            break;
        }
        ++i;
    }
    if (start == i) {
        throw std::runtime_error("json: expected number");
    }
    return std::stod(s.substr(start, i - start));
}

struct AskRequest {
    std::string query;
    std::size_t k = 5;
    bool has_k = false;
};

/// Parse POST body: requires "query" string; optional "k" number (default 5).
inline AskRequest parse_ask_request(const std::string& body) {
    AskRequest req;
    std::size_t i = 0;
    skip_ws(body, i);
    if (i >= body.size() || body[i] != '{') {
        throw std::runtime_error("json: expected object");
    }
    ++i;
    bool saw_query = false;
    skip_ws(body, i);
    if (i < body.size() && body[i] == '}') {
        throw std::runtime_error("json: missing required field \"query\"");
    }
    while (i < body.size()) {
        skip_ws(body, i);
        if (i < body.size() && body[i] == '}') {
            break;
        }
        const std::string key = parse_string(body, i);
        skip_ws(body, i);
        if (i >= body.size() || body[i] != ':') {
            throw std::runtime_error("json: expected ':'");
        }
        ++i;
        skip_ws(body, i);
        if (key == "query") {
            req.query = parse_string(body, i);
            saw_query = true;
        } else if (key == "k") {
            double kv = parse_number(body, i);
            if (kv < 1 || kv > 1000) {
                throw std::runtime_error("json: k must be in [1, 1000]");
            }
            req.k = static_cast<std::size_t>(kv);
            req.has_k = true;
        } else {
            // skip value: string | number | bool | null | object/array (shallow fail)
            if (i < body.size() && body[i] == '"') {
                (void)parse_string(body, i);
            } else if (i < body.size() &&
                       (body[i] == '-' || std::isdigit(static_cast<unsigned char>(body[i])))) {
                (void)parse_number(body, i);
            } else if (body.compare(i, 4, "true") == 0) {
                i += 4;
            } else if (body.compare(i, 5, "false") == 0) {
                i += 5;
            } else if (body.compare(i, 4, "null") == 0) {
                i += 4;
            } else {
                throw std::runtime_error("json: unsupported value for key " + key);
            }
        }
        skip_ws(body, i);
        if (i < body.size() && body[i] == ',') {
            ++i;
            continue;
        }
        if (i < body.size() && body[i] == '}') {
            break;
        }
        throw std::runtime_error("json: expected ',' or '}'");
    }
    if (!saw_query) {
        throw std::runtime_error("json: missing required field \"query\"");
    }
    return req;
}

}  // namespace json
}  // namespace nanorag
