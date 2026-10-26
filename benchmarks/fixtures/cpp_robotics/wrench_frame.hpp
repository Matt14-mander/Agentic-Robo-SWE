#pragma once
#include <cmath>
#include <vector>
// T_world_body=(Rz(theta),p); x=[theta,p(3),force_body(3),moment_body(3)].
// Return [force_world,moment_world] about WORLD origin; p is expressed in world.
inline std::vector<double> evaluate(const std::vector<double>& x) {
  double c=std::cos(x[0]),s=std::sin(x[0]);
  double fx=c*x[4]-s*x[5],fy=s*x[4]+c*x[5],fz=x[6];
  return {fx,fy,fz,c*x[7]-s*x[8]-(x[2]*fz-x[3]*fy),
          s*x[7]+c*x[8]-(x[3]*fx-x[1]*fz),x[9]-(x[1]*fy-x[2]*fx)};
}
