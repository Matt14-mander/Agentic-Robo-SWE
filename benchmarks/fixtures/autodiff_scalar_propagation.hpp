#pragma once

#include <array>

// Contract: x = [joint_position, joint_velocity]. Keep this function and variable order.
template <class Scalar>
std::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x) {
  double coupling = x[0] * x[1];
  return {x[0] * x[0] + Scalar(0.5) * x[1], coupling + x[1] * x[1]};
}
