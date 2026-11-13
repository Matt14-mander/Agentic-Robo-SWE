#pragma once
#include <chrono>
namespace retrieval_fixture {
class Deadline {
    using Clock = std::chrono::steady_clock;
    Clock::time_point expires_;
public:
    explicit Deadline(std::chrono::milliseconds budget) : expires_(Clock::now() + budget) {}
    bool expired() const { return Clock::now() >= expires_; }
    std::chrono::milliseconds remaining() const {
        auto now = Clock::now();
        return now >= expires_ ? std::chrono::milliseconds(0)
            : std::chrono::duration_cast<std::chrono::milliseconds>(expires_ - now);
    }
};
}
