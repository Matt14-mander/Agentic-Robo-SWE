#pragma once
#include <array>
template<class Scalar> std::array<Scalar, 2> robot_step(const std::array<Scalar, 2>& x, const Scalar& u) {
  return {x[0] + u, x[1] - u};
}
