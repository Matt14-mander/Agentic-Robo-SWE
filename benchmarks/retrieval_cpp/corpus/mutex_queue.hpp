#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>
namespace retrieval_fixture {
template<class T> class BlockingQueue {
    std::queue<T> items_;
    std::mutex mutex_;
    std::condition_variable ready_;
public:
    void push(T value) {
        { std::lock_guard<std::mutex> lock(mutex_); items_.push(std::move(value)); }
        ready_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return !items_.empty(); });
        T value = std::move(items_.front());
        items_.pop();
        return value;
    }
};
}
