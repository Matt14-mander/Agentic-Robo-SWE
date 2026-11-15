#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace retrieval_fixture {
inline std::vector<unsigned char> decode_frame(const unsigned char* bytes, std::size_t size) {
    if (size < 2) throw std::invalid_argument("missing header");
    const std::size_t payload_size = (std::size_t(bytes[0]) << 8) | bytes[1];
    if (payload_size > size - 2) throw std::invalid_argument("incomplete payload");
    return {bytes + 2, bytes + 2 + payload_size};
}
}
