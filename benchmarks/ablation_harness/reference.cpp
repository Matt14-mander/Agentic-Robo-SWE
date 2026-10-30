#include <array>
#include <fstream>
#include <iomanip>
#include "candidate.hpp"
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream input(argv[1]); std::ofstream output(argv[2]);
  if (!input || !output) return 3;
  output << std::setprecision(17);
  for (int i=0; i<24; ++i) {
    std::array<double,2> x; double u;
    if (!(input >> x[0] >> x[1] >> u)) return 4;
    auto y = robot_step(x, u);
    output << y[0] << ' ' << y[1] << '\n';
  }
  output.flush();
  return output ? 0 : 5;
}
