#pragma once

#include <array>

// Contract: x = [joint_position, joint_velocity]. Keep this function and variable order.
inline std::array<double, 2> robot_codegen_model(const std::array<double, 2>& x) {
  return {x[0] * x[0] + 0.5 * x[1], x[0] * x[1] + x[1] * x[1]};
}
