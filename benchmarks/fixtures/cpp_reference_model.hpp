#pragma once

#include <array>

template <typename Scalar>
std::array<Scalar, 2> robot_step(
    const std::array<Scalar, 2>& state,
    const Scalar& control) {
  // BUG: the second state equation uses the wrong control direction.
  return {state[0] + control, state[1] + control};
}

