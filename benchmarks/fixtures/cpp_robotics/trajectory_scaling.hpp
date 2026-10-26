#pragma once
#include <algorithm>
#include <vector>
// x=[start,goal,duration_seconds,time_seconds], duration>0. Return [q,dq,ddq].
// Minimum-jerk interpolation, with zero endpoint derivatives and clamped time.
inline std::vector<double> evaluate(const std::vector<double>& x) {
  double s=std::clamp(x[3]/x[2],0.,1.),d=x[1]-x[0];
  double q=10*s*s*s-15*s*s*s*s+6*s*s*s*s*s;
  double v=30*s*s-60*s*s*s+30*s*s*s*s;
  double a=60*s-180*s*s+120*s*s*s;
  return {x[0]+d*q,d*v/x[2],d*a};
}
