#pragma once
#include <stdexcept>
namespace retrieval_fixture {
inline double interpolate(double x, double x0, double x1, double y0, double y1) {
    if (x1 == x0) throw std::invalid_argument("coincident samples");
    const double fraction = (x - x0) / (x1 - x0);
    return y0 + fraction * (y1 - y0);
}
}
