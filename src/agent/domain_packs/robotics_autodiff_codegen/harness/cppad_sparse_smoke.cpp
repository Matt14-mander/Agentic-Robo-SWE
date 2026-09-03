// Cross-platform CppAD-only probe; this does NOT replace the Linux CodeGen gate.
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  using AD = CppAD::AD<double>;
  std::vector<AD> x(2);
  CppAD::Independent(x);
  std::vector<AD> y{x[0] * x[0], x[0] * x[1] + x[1] * x[1]};
  CppAD::ADFun<double> f(x, y);
  const std::vector<bool> identity{true, false, false, true};
  const auto pattern = f.ForSparseJac(2, identity);
  if (pattern != std::vector<bool>{true, false, true, true}) return 1;
  for (const std::vector<double>& input : {
           std::vector<double>{0, 0}, std::vector<double>{1.2, -0.4},
           std::vector<double>{-2, 4}}) {
    const auto dense = f.Jacobian(input);
    const auto sparse = f.SparseJacobian(input);
    const std::vector<double> expected{2 * input[0], 0, input[1], input[0] + 2 * input[1]};
    for (std::size_t i = 0; i < 4; ++i) {
      if (!std::isfinite(sparse[i]) || std::abs(sparse[i] - expected[i]) > 1e-12 ||
          std::abs(sparse[i] - dense[i]) > 1e-12) return 2;
    }
  }
  std::cout << "CppAD sparse/dense verified: structural nnz=3 including zero-valued origin\n";
  return 0;
}
