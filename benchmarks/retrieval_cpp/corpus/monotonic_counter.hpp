#pragma once
#include <cstdint>
namespace retrieval_fixture {
class CounterUnwrapper {
    std::uint32_t previous_ = 0;
    std::uint64_t extended_ = 0;
    bool initialized_ = false;
public:
    std::uint64_t observe(std::uint32_t reading) {
        if (!initialized_) { previous_ = reading; extended_ = reading; initialized_ = true; }
        else { extended_ += static_cast<std::uint32_t>(reading - previous_); previous_ = reading; }
        return extended_;
    }
};
}
