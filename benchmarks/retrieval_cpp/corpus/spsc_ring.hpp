#pragma once
#include <array>
#include <atomic>
#include <cstddef>
namespace retrieval_fixture {
template<class T, std::size_t N> class SingleProducerRing {
    static_assert(N > 1);
    std::array<T, N> slots_{};
    std::atomic<std::size_t> head_{0}, tail_{0};
public:
    bool push(const T& value) {
        auto head = head_.load(std::memory_order_relaxed);
        auto next = (head + 1) % N;
        if (next == tail_.load(std::memory_order_acquire)) return false;
        slots_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }
    bool pop(T& value) {
        auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;
        value = slots_[tail];
        tail_.store((tail + 1) % N, std::memory_order_release);
        return true;
    }
};
}
