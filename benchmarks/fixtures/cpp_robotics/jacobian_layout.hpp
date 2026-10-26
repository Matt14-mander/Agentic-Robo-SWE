#pragma once
#include <vector>
// x=[J00,J01,J10,J11,J20,J21,v0,v1]; J is ROW MAJOR, output=J*v.
inline std::vector<double> evaluate(const std::vector<double>& x) {
  std::vector<double> out(3,0.);
  for(int r=0;r<3;++r) for(int c=0;c<2;++c) out[r]+=x[c*3+r]*x[6+c];
  return out;
}
