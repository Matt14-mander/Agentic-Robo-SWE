#include <array>
#include <cmath>
#include <iostream>

#include WORKSPACE_HEADER

int main() {
  const std::array<double, 2> state{1.25, -0.5};
  const double control = 0.75;
  const auto output = robot_step(state, control);
  const std::array<double, 2> expected{2.0, -1.25};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(output[index]) || std::abs(output[index] - expected[index]) > 1e-12) {
      std::cerr << "output[" << index << "]=" << output[index]
                << ", expected=" << expected[index] << '\n';
      return 1;
    }
  }
  std::cout << "output equivalence passed\n";
  return 0;
}

