#pragma once
#include <vector>
// x=[position,velocity,constant_acceleration,frequency_hz]; frequency is positive integer.
// Return exact constant-acceleration position/velocity after one sample (SI units).
inline std::vector<double> evaluate(const std::vector<double>& x) {
  double dt=1/static_cast<int>(x[3]);
  return {x[0]+x[1]*dt+.5*x[2]*dt*dt,x[1]+x[2]*dt};
}
