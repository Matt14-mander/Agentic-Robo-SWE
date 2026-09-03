#pragma once
#include <array>

// Contract: x = [joint_position, joint_velocity].
// y = [q*q, q*v + v*v]; J[0,1] is structurally zero.
template <class Scalar>
std::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x) {
  return {x[0] * x[0] + Scalar(0.01) * x[1], x[0] * x[1] + x[1] * x[1]};
}
