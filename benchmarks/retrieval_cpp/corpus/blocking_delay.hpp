#pragma once
#include <chrono>
#include <thread>
namespace retrieval_fixture {
inline void wait_fixed_delay(unsigned milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}
}
