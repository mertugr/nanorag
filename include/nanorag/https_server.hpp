#pragma once

// Minimal HTTPS (TLS) request server for nanorag serve — OpenSSL + BSD sockets.
// Single-threaded accept loop; each request is handled fully then closed.

#include "nanorag/json_util.hpp"
#include "nanorag/rag_service.hpp"

#include <openssl/err.h>
#include <openssl/ssl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanorag {

struct HttpsServerConfig {
    std::string host = "127.0.0.1";
    int port = 8443;
    std::string cert_path;
    std::string key_path;
    /// If true, plain HTTP (no TLS). Default false (HTTPS).
    bool plain_http = false;
};

inline std::string http_response(int status, const std::string& status_text,
                                 const std::string& content_type, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type\r\n"
        << "\r\n"
        << body;
    return oss.str();
}

inline std::string http_json(int status, const std::string& body) {
    const char* text = "OK";
    if (status == 400) {
        text = "Bad Request";
    } else if (status == 404) {
        text = "Not Found";
    } else if (status == 405) {
        text = "Method Not Allowed";
    } else if (status == 500) {
        text = "Internal Server Error";
    }
    return http_response(status, text, "application/json; charset=utf-8", body);
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

inline bool parse_http_request(const std::string& raw, HttpRequest& out) {
    const auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return false;
    }
    const std::string headers = raw.substr(0, header_end);
    std::istringstream hs(headers);
    std::string request_line;
    if (!std::getline(hs, request_line)) {
        return false;
    }
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }
    {
        std::istringstream rl(request_line);
        std::string version;
        if (!(rl >> out.method >> out.path >> version)) {
            return false;
        }
    }
    std::size_t content_length = 0;
    std::string line;
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, colon);
        for (char& c : key) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        std::string val = line.substr(colon + 1);
        while (!val.empty() && val.front() == ' ') {
            val.erase(val.begin());
        }
        if (key == "content-length") {
            content_length = static_cast<std::size_t>(std::stoull(val));
        }
    }
    out.body = raw.substr(header_end + 4);
    if (out.body.size() < content_length) {
        return false;  // need more data
    }
    if (out.body.size() > content_length) {
        out.body.resize(content_length);
    }
    return true;
}

class HttpsServer {
public:
    HttpsServer(RagService& service, HttpsServerConfig cfg)
        : service_(service), cfg_(std::move(cfg)) {}

    ~HttpsServer() { stop(); }

    void stop() {
        running_ = false;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (ssl_ctx_) {
            SSL_CTX_free(ssl_ctx_);
            ssl_ctx_ = nullptr;
        }
    }

    /// Blocking serve loop until stop() or fatal error.
    void run() {
        init_openssl_once();
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
        }
        int yes = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(cfg_.port));
        if (::inet_pton(AF_INET, cfg_.host.c_str(), &addr.sin_addr) != 1) {
            throw std::runtime_error("invalid --host address: " + cfg_.host);
        }
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error(std::string("bind: ") + std::strerror(errno));
        }
        if (::listen(listen_fd_, 16) < 0) {
            throw std::runtime_error(std::string("listen: ") + std::strerror(errno));
        }

        if (!cfg_.plain_http) {
            ssl_ctx_ = SSL_CTX_new(TLS_server_method());
            if (!ssl_ctx_) {
                throw std::runtime_error("SSL_CTX_new failed");
            }
            if (SSL_CTX_use_certificate_file(ssl_ctx_, cfg_.cert_path.c_str(), SSL_FILETYPE_PEM) !=
                1) {
                throw std::runtime_error("failed to load cert: " + cfg_.cert_path);
            }
            if (SSL_CTX_use_PrivateKey_file(ssl_ctx_, cfg_.key_path.c_str(), SSL_FILETYPE_PEM) != 1) {
                throw std::runtime_error("failed to load key: " + cfg_.key_path);
            }
            if (SSL_CTX_check_private_key(ssl_ctx_) != 1) {
                throw std::runtime_error("private key does not match certificate");
            }
        }

        running_ = true;
        const char* scheme = cfg_.plain_http ? "http" : "https";
        std::cerr << "nanorag serve listening on " << scheme << "://" << cfg_.host << ":"
                  << cfg_.port << "\n"
                  << "  POST /ask  JSON {\"query\":\"...\",\"k\":N}\n"
                  << "  GET  /health\n"
                  << "  index loaded once; MultiSeqSession="
                  << (service_.has_llm() ? "ready" : "n/a (extractive)") << "\n"
                  << std::flush;

        while (running_) {
            sockaddr_in client{};
            socklen_t clen = sizeof(client);
            int cfd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &clen);
            if (cfd < 0) {
                if (!running_) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                std::cerr << "accept error: " << std::strerror(errno) << "\n";
                continue;
            }
            try {
                handle_client(cfd);
            } catch (const std::exception& e) {
                std::cerr << "client error: " << e.what() << "\n";
            }
            ::close(cfd);
        }
    }

private:
    static void init_openssl_once() {
        static std::once_flag once;
        std::call_once(once, [] {
            SSL_library_init();
            SSL_load_error_strings();
            OpenSSL_add_ssl_algorithms();
        });
    }

    void handle_client(int cfd) {
        if (cfg_.plain_http) {
            std::string raw = read_all_fd(cfd);
            HttpRequest req;
            if (!parse_http_request(raw, req)) {
                // try read more once
                raw += read_all_fd(cfd);
                if (!parse_http_request(raw, req)) {
                    write_all_fd(cfd, http_json(400, error_json("incomplete HTTP request")));
                    return;
                }
            }
            const std::string resp = dispatch(req);
            write_all_fd(cfd, resp);
            return;
        }

        SSL* ssl = SSL_new(ssl_ctx_);
        if (!ssl) {
            throw std::runtime_error("SSL_new failed");
        }
        SSL_set_fd(ssl, cfd);
        if (SSL_accept(ssl) <= 0) {
            SSL_free(ssl);
            throw std::runtime_error("SSL_accept failed");
        }
        try {
            std::string raw = read_all_ssl(ssl);
            HttpRequest req;
            if (!parse_http_request(raw, req)) {
                raw += read_all_ssl(ssl);
                if (!parse_http_request(raw, req)) {
                    write_all_ssl(ssl, http_json(400, error_json("incomplete HTTP request")));
                    SSL_shutdown(ssl);
                    SSL_free(ssl);
                    return;
                }
            }
            const std::string resp = dispatch(req);
            write_all_ssl(ssl, resp);
        } catch (...) {
            SSL_shutdown(ssl);
            SSL_free(ssl);
            throw;
        }
        SSL_shutdown(ssl);
        SSL_free(ssl);
    }

    std::string dispatch(const HttpRequest& req) {
        std::cerr << req.method << " " << req.path << " body_bytes=" << req.body.size() << "\n"
                  << std::flush;

        if (req.method == "OPTIONS") {
            return http_response(204, "No Content", "text/plain", "");
        }

        // normalize path (strip query string)
        std::string path = req.path;
        const auto qpos = path.find('?');
        if (qpos != std::string::npos) {
            path = path.substr(0, qpos);
        }

        if (path == "/health" || path == "/") {
            if (req.method != "GET" && req.method != "HEAD") {
                return http_json(405, error_json("use GET for /health"));
            }
            std::ostringstream oss;
            oss << "{"
                << "\"status\":" << json::str("ok") << ","
                << "\"service\":" << json::str("nanorag") << ","
                << "\"mode\":" << json::str(service_.config().mode) << ","
                << "\"index\":" << json::str(service_.config().index_dir) << ","
                << "\"chunks\":" << json::number(service_.retriever().size()) << ","
                << "\"real_chunks\":" << json::number(service_.retriever().real_size()) << ","
                << "\"llm\":" << json::boolean(service_.has_llm()) << ","
                << "\"retrieve\":"
                << json::str(retrieve_mode_name(service_.config().retrieve_mode))
                << "}";
            return http_json(200, oss.str());
        }

        if (path == "/ask") {
            if (req.method != "POST") {
                return http_json(405, error_json("use POST /ask with JSON body {query,k}"));
            }
            auto result = service_.ask_json(req.body);
            return http_json(result.http_status, result.body);
        }

        return http_json(404, error_json("not found: " + path));
    }

    static std::string read_all_fd(int fd) {
        std::string out;
        char buf[8192];
        // Non-blocking-ish: single read with a reasonable size; caller may re-read.
        // Set a short RCVTIMEO so we don't hang forever on partial bodies.
        timeval tv{};
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        for (;;) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n > 0) {
                out.append(buf, static_cast<std::size_t>(n));
                // If we already have a full request, stop.
                HttpRequest tmp;
                if (parse_http_request(out, tmp)) {
                    break;
                }
                if (out.size() > 8 * 1024 * 1024) {
                    break;
                }
                continue;
            }
            break;
        }
        return out;
    }

    static void write_all_fd(int fd, const std::string& data) {
        std::size_t off = 0;
        while (off < data.size()) {
            const ssize_t n =
                ::send(fd, data.data() + off, data.size() - off, 0);
            if (n <= 0) {
                break;
            }
            off += static_cast<std::size_t>(n);
        }
    }

    static std::string read_all_ssl(SSL* ssl) {
        std::string out;
        char buf[8192];
        for (;;) {
            const int n = SSL_read(ssl, buf, sizeof(buf));
            if (n > 0) {
                out.append(buf, static_cast<std::size_t>(n));
                HttpRequest tmp;
                if (parse_http_request(out, tmp)) {
                    break;
                }
                if (out.size() > 8 * 1024 * 1024) {
                    break;
                }
                continue;
            }
            break;
        }
        return out;
    }

    static void write_all_ssl(SSL* ssl, const std::string& data) {
        std::size_t off = 0;
        while (off < data.size()) {
            const int n = SSL_write(ssl, data.data() + off,
                                   static_cast<int>(std::min(data.size() - off, std::size_t(1 << 20))));
            if (n <= 0) {
                break;
            }
            off += static_cast<std::size_t>(n);
        }
    }

    RagService& service_;
    HttpsServerConfig cfg_;
    SSL_CTX* ssl_ctx_ = nullptr;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
};

}  // namespace nanorag
