#pragma once
// Public calibration solutions, NOT a hidden evaluation set. Never supplied as an Agent fix.
#include <algorithm>
#include <cmath>
#include <vector>
#if ROBOTICS_CASE == 0
template<class Scalar> Scalar residual(Scalar q, Scalar v) {
  using std::sin;return sin(q)+q*v;
}
#else
inline std::vector<double> evaluate(const std::vector<double>& x) {
#if ROBOTICS_CASE == 1
  std::vector<double> out(3,0.);
  for(int r=0;r<3;++r) for(int c=0;c<2;++c) out[r]+=x[r*2+c]*x[6+c];
  return out;
#elif ROBOTICS_CASE == 2
  double norm=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
  double w=x[3]/norm,a=x[0]/norm,b=x[1]/norm,c=x[2]/norm;
  double vx=x[4],vy=x[5],vz=x[6];
  return {(1-2*(b*b+c*c))*vx+2*(a*b-c*w)*vy+2*(a*c+b*w)*vz,
          2*(a*b+c*w)*vx+(1-2*(a*a+c*c))*vy+2*(b*c-a*w)*vz,
          2*(a*c-b*w)*vx+2*(b*c+a*w)*vy+(1-2*(a*a+b*b))*vz};
#elif ROBOTICS_CASE == 3
  double c=std::cos(x[0]),s=std::sin(x[0]);
  double fx=c*x[4]-s*x[5],fy=s*x[4]+c*x[5],fz=x[6];
  return {fx,fy,fz,c*x[7]-s*x[8]+(x[2]*fz-x[3]*fy),
          s*x[7]+c*x[8]+(x[3]*fx-x[1]*fz),x[9]+(x[1]*fy-x[2]*fx)};
#elif ROBOTICS_CASE == 4
  double dt=1./x[3];
  return {x[0]+x[1]*dt+.5*x[2]*dt*dt,x[1]+x[2]*dt};
#elif ROBOTICS_CASE == 5
  double s=std::clamp(x[3]/x[2],0.,1.),d=x[1]-x[0];
  return {x[0]+d*(10*s*s*s-15*s*s*s*s+6*s*s*s*s*s),
          d*(30*s*s-60*s*s*s+30*s*s*s*s)/x[2],
          d*(60*s-180*s*s+120*s*s*s)/(x[2]*x[2])};
#elif ROBOTICS_CASE == 6
  double integral=0.;std::vector<double> out;
  for(std::size_t i=4;i<x.size();++i) {
    double e=x[i],candidate=integral+e*x[0],raw=x[2]*e+x[3]*candidate;
    if(std::abs(raw)<=x[1] || (raw>x[1] && e<0) || (raw<-x[1] && e>0)) integral=candidate;
    out.push_back(std::clamp(x[2]*e+x[3]*integral,-x[1],x[1]));
  }
  out.push_back(integral);return out;
#elif ROBOTICS_CASE == 7
  return {std::atan2(std::sin(x[1]-x[0]),std::cos(x[1]-x[0]))};
#endif
}
#endif
