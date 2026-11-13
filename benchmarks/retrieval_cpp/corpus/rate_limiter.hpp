#pragma once
#include <algorithm>
#include <chrono>
namespace retrieval_fixture {
class TokenBucket {
    using Clock = std::chrono::steady_clock;
    double tokens_, capacity_, rate_;
    Clock::time_point previous_ = Clock::now();
public:
    TokenBucket(double capacity, double rate) : tokens_(capacity), capacity_(capacity), rate_(rate) {}
    bool allow(double cost = 1.0) {
        auto now = Clock::now();
        tokens_ = std::min(capacity_, tokens_ + std::chrono::duration<double>(now - previous_).count() * rate_);
        previous_ = now;
        if (tokens_ < cost) return false;
        tokens_ -= cost;
        return true;
    }
};
}
