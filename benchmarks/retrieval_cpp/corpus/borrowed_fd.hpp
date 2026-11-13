#pragma once
namespace retrieval_fixture {
class BorrowedHandle {
    int value_;
public:
    explicit BorrowedHandle(int value) : value_(value) {}
    int get() const { return value_; }
};
}
