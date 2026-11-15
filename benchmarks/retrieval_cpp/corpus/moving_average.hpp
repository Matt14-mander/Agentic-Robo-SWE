#pragma once
#include <array>
#include <cstddef>
namespace retrieval_fixture {
template<std::size_t N> class WindowMean {
    static_assert(N > 0);
    std::array<double, N> samples_{};
    std::size_t cursor_ = 0, count_ = 0;
    double sum_ = 0.0;
public:
    double add(double sample) {
        sum_ -= samples_[cursor_];
        samples_[cursor_] = sample;
        sum_ += sample;
        cursor_ = (cursor_ + 1) % N;
        if (count_ < N) ++count_;
        return sum_ / count_;
    }
};
}
