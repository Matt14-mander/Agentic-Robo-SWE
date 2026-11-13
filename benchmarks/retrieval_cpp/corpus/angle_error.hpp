#pragma once
#include <cmath>
namespace retrieval_fixture {
inline double shortest_rotation(double target, double measured) {
    const double pi = std::acos(-1.0);
    double delta = std::remainder(target - measured, 2.0 * pi);
    return delta;
}
}
