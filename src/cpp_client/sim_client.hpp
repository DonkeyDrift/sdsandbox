// DonkeySim TCP JSON 客户端:帧定界与分叉版 gym_donkeycar/core/client.py 对齐——
// 按 '{'...'}' 提取对象(协议假设字符串值内无花括号),'\n' 为可选终止符,
// TCP 分段/粘包任意组合不丢消息。禁 Nagle + 8MB 接收缓冲(防遥测反压卡死 sim)。
#pragma once

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

#include <nlohmann/json.hpp>

namespace dk {

class SimClient {
public:
    SimClient(const std::string& host, uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("socket() failed");
        int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        int rcvbuf = 8 * 1024 * 1024;
        setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
            throw std::runtime_error("bad host: " + host);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(fd_);
            throw std::runtime_error("connect " + host + ":" + std::to_string(port) +
                                     " failed (sim running?)");
        }
    }

    ~SimClient() { if (fd_ >= 0) ::close(fd_); }
    SimClient(const SimClient&) = delete;
    SimClient& operator=(const SimClient&) = delete;

    void send_raw(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::send(fd_, s.data() + off, s.size() - off, 0);
            if (n <= 0) throw std::runtime_error("send failed (disconnect?)");
            off += static_cast<size_t>(n);
        }
    }

    void send_json(const nlohmann::json& j) { send_raw(j.dump() + "\n"); }

    // 下一条完整 JSON 对象文本(可能含非 telemetry 消息);断连抛异常。
    std::string next_json_text() {
        for (;;) {
            if (!queue_.empty()) {
                std::string out = queue_.front();
                queue_.pop_front();
                return out;
            }
            recv_more();
        }
    }

    nlohmann::json next_json() {
        return nlohmann::json::parse(next_json_text(), nullptr, false);
    }

    // 阻塞等到下一条 telemetry 帧(期间忽略 car_loaded 等)。
    nlohmann::json next_telemetry() {
        for (;;) {
            nlohmann::json j = next_json();
            if (j.is_discarded()) continue;
            if (j.value("msg_type", "") == "telemetry") return j;
        }
    }

    int fd() const { return fd_; }

private:
    // 从 socket 收一批字节并按 '}' 边界(剥掉其后可选 '\n'/"\r\n")切分对象。
    void recv_more() {
        char buf[65536];
        ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0) throw std::runtime_error("recv failed (sim closed connection?)");
        pending_.append(buf, static_cast<size_t>(n));

        size_t i = 0;
        const size_t len = pending_.size();
        while (i < len) {
            size_t open = pending_.find('{', i);
            if (open == std::string::npos) { i = len; break; }  // 无对象起点,丢弃残余
            size_t close = pending_.find('}', open);
            if (close == std::string::npos) break;              // 半个对象,等下一块
            queue_.push_back(pending_.substr(open, close - open + 1));
            i = close + 1;
            if (i < len && pending_[i] == '\r' && i + 1 < len && pending_[i + 1] == '\n')
                i += 2;
            else if (i < len && pending_[i] == '\n')
                i += 1;
        }
        pending_.erase(0, i);
    }

    int fd_ = -1;
    std::string pending_;            // 未切分的残余字节
    std::deque<std::string> queue_;  // 已切出的完整对象
};

}  // namespace dk
