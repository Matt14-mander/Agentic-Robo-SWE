#pragma once
#include <cmath>
#include <vector>
// x=[current,target] for a continuous revolute joint, in radians.
// Return the signed shortest angular displacement in [-pi,pi] (atan2 convention).
inline std::vector<double> evaluate(const std::vector<double>& x) {
  return {x[1]-x[0]};
}
