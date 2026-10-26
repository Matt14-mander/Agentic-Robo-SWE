#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Small forward-mode scalar tests C++ type propagation, not a CppAD substitute.
namespace probe {
struct Dual {
  double value, derivative;
  Dual(double v, double d=0.) : value(v), derivative(d) {}
  explicit operator double() const { return value; }
};
inline Dual operator+(Dual a, Dual b) {return {a.value+b.value,a.derivative+b.derivative};}
inline Dual operator*(Dual a, Dual b) {
  return {a.value*b.value,a.derivative*b.value+a.value*b.derivative};
}
inline Dual sin(Dual a) {return {std::sin(a.value),std::cos(a.value)*a.derivative};}
}
#include WORKSPACE_HEADER

int main(int argc, char** argv) {
  if(argc!=3) return 2;
  try {
    std::ifstream input(argv[1]); std::ofstream output(argv[2]);
    if(!input || !output) throw std::runtime_error("cannot open observation files");
    int count; if(!(input>>count) || count<1 || count>1000) return 2;
    output<<std::setprecision(17);
    for(int row=0;row<count;++row) {
      int size; if(!(input>>size) || size<1 || size>100) return 2;
      std::vector<double> x(size);
      for(auto& value:x) if(!(input>>value)) return 2;
      std::vector<double> actual;
#if ROBOTICS_CASE == 0
      auto q=residual(probe::Dual(x[0],1.),probe::Dual(x[1],0.));
      auto v=residual(probe::Dual(x[0],0.),probe::Dual(x[1],1.));
      actual={residual(x[0],x[1]),q.derivative,v.derivative};
#else
      actual=evaluate(x);
#endif
      output<<actual.size(); for(double value:actual) output<<' '<<value; output<<'\n';
    }
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
