#pragma once
#include <cstddef>
namespace retrieval_fixture {
inline unsigned sum_checksum(const unsigned char* bytes, std::size_t size) {
    unsigned sum = 0;
    for (std::size_t i = 0; i < size; ++i) sum = (sum + bytes[i]) & 255;
    return sum;
}
}
