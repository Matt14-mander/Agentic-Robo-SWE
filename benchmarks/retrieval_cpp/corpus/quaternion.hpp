#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace retrieval_fixture {
inline std::array<double, 4> unit_quaternion(std::array<double, 4> q) {
    double squared = 0.0;
    for (double value : q) squared += value * value;
    if (!(squared > 1e-24) || !std::isfinite(squared))
        throw std::invalid_argument("rotation has no finite magnitude");
    const double scale = 1.0 / std::sqrt(squared);
    for (double& value : q) value *= scale;
    return q;
}
}
