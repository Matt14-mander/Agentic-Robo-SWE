#pragma once

#include <array>

// Contract: x = [joint_position, joint_velocity]. Keep this function and variable order.
template <class Scalar>
std::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x) {
  // Bug: the implementation treats x as [joint_velocity, joint_position].
  return {x[1] * x[1] + Scalar(0.5) * x[0], x[1] * x[0] + x[0] * x[0]};
}
