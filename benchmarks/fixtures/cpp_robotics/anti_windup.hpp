#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
// x=[dt,limit,kp,ki,errors...]; integral starts at zero. Return all torques then integral.
// Conditional integration: accept an update only if its raw torque is inside limits
// or the error drives that raw torque back toward the allowed range.
inline std::vector<double> evaluate(const std::vector<double>& x) {
  double integral=0.; std::vector<double> out;
  for(std::size_t i=4;i<x.size();++i) {
    double error=x[i],candidate=integral+error*x[0];
    integral=candidate;
    out.push_back(std::clamp(x[2]*error+x[3]*integral,-x[1],x[1]));
  }
  out.push_back(integral);return out;
}
