#pragma once
#include <array>
namespace retrieval_fixture {
using Point = std::array<double, 3>;
inline Point transform_point(const std::array<double, 9>& rotation,
                             const Point& translation, const Point& p) {
    Point out{};
    for (int row = 0; row < 3; ++row) {
        out[row] = translation[row];
        for (int col = 0; col < 3; ++col) out[row] += rotation[row * 3 + col] * p[col];
    }
    return out;
}
}
