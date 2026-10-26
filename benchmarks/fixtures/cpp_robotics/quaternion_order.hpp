#pragma once
#include <cmath>
#include <vector>
// x=[qx,qy,qz,qw,vx,vy,vz], nonzero quaternion, active body-to-world rotation.
inline std::vector<double> evaluate(const std::vector<double>& x) {
  double norm=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
  double w=x[0]/norm, a=x[1]/norm, b=x[2]/norm, c=x[3]/norm;
  double vx=x[4],vy=x[5],vz=x[6];
  return {(1-2*(b*b+c*c))*vx+2*(a*b-c*w)*vy+2*(a*c+b*w)*vz,
          2*(a*b+c*w)*vx+(1-2*(a*a+c*c))*vy+2*(b*c-a*w)*vz,
          2*(a*c-b*w)*vx+2*(b*c+a*w)*vy+(1-2*(a*a+b*b))*vz};
}
