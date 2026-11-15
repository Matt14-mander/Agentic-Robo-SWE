#pragma once
#include <cstdint>
namespace retrieval_fixture {
inline std::uint32_t read_network_u32(const unsigned char* bytes) {
    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16)
         | (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}
inline void write_network_u32(unsigned char* out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(value >> (24 - i * 8));
}
}
