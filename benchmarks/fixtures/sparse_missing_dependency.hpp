#pragma once
#include <array>

// Contract: x = [joint_position, joint_velocity].
// y = [q*q, q*v + v*v]; J[1,0] must remain in the structural pattern even at v=0.
template <class Scalar>
std::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x) {
  return {x[0] * x[0], x[1] * x[1]};
}
