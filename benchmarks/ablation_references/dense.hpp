#pragma once
#include <array>
template<class Scalar> std::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x) {
  return {x[0]*x[0] + Scalar(0.5)*x[1], x[0]*x[1] + x[1]*x[1]};
}
