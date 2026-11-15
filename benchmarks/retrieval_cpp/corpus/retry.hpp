#pragma once
#include <algorithm>
#include <cstdint>
namespace retrieval_fixture {
inline std::uint64_t retry_delay_ms(unsigned attempt, std::uint64_t initial,
                                    std::uint64_t maximum) {
    std::uint64_t delay = std::min(initial, maximum);
    for (unsigned i = 0; i < attempt && delay < maximum; ++i) {
        if (delay > maximum / 2) return maximum;
        delay *= 2;
    }
    return delay;
}
}
