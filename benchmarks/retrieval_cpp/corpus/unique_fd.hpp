#pragma once
#include <utility>
namespace retrieval_fixture {
class UniqueHandle {
    int value_;
    void (*close_)(int);
public:
    UniqueHandle(int value, void (*close_fn)(int)) : value_(value), close_(close_fn) {}
    ~UniqueHandle() { if (value_ >= 0) close_(value_); }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    UniqueHandle(UniqueHandle&& other) noexcept
        : value_(std::exchange(other.value_, -1)), close_(other.close_) {}
    int release() { return std::exchange(value_, -1); }
    int get() const { return value_; }
};
}
