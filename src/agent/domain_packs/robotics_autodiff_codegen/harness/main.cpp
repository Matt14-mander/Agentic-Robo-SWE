#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cppad/cg.hpp>
#include WORKSPACE_HEADER

#ifndef OUTPUT_TOLERANCE
#define OUTPUT_TOLERANCE 1e-10
#endif
#ifndef JACOBIAN_RELATIVE_TOLERANCE
#define JACOBIAN_RELATIVE_TOLERANCE 1e-6
#endif

namespace {
constexpr std::size_t kInputSize = 2;
constexpr std::size_t kOutputSize = 2;

template <class Scalar>
std::array<Scalar, kOutputSize> expected_model(const std::array<Scalar, kInputSize>& x) {
  return {x[0] * x[0] + Scalar(0.5) * x[1], x[0] * x[1] + x[1] * x[1]};
}

double max_output_error(const std::vector<double>& actual,
                        const std::array<double, kOutputSize>& expected) {
  if (actual.size() != expected.size()) return INFINITY;
  double error = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i])) return INFINITY;
    error = std::max(error, std::abs(actual[i] - expected[i]));
  }
  return error;
}

std::vector<double> finite_difference(const std::array<double, kInputSize>& x) {
  constexpr double step = 1e-6;
  std::vector<double> jacobian(kOutputSize * kInputSize);
  for (std::size_t column = 0; column < kInputSize; ++column) {
    auto plus = x;
    auto minus = x;
    plus[column] += step;
    minus[column] -= step;
    const auto y_plus = robot_codegen_model(plus);
    const auto y_minus = robot_codegen_model(minus);
    for (std::size_t row = 0; row < kOutputSize; ++row) {
      jacobian[row * kInputSize + column] = (y_plus[row] - y_minus[row]) / (2.0 * step);
    }
  }
  return jacobian;
}

double relative_error(const std::vector<double>& actual, const std::vector<double>& expected) {
  if (actual.size() != expected.size()) return INFINITY;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return INFINITY;
    const double difference = actual[i] - expected[i];
    squared_error += difference * difference;
    squared_reference += expected[i] * expected[i];
  }
  return std::sqrt(squared_error) / std::max(1.0, std::sqrt(squared_reference));
}

CppAD::ADFun<double> record_cppad() {
  using AD = CppAD::AD<double>;
  std::vector<AD> independent(kInputSize);
  CppAD::Independent(independent);
  const std::array<AD, kInputSize> input{independent[0], independent[1]};
  const auto output = robot_codegen_model(input);
  std::vector<AD> dependent(output.begin(), output.end());
  return CppAD::ADFun<double>(independent, dependent);
}

std::unique_ptr<CppAD::cg::DynamicLib<double>> compile_codegen() {
  using CG = CppAD::cg::CG<double>;
  using ADCG = CppAD::AD<CG>;
  std::vector<ADCG> independent(kInputSize);
  CppAD::Independent(independent);
  const std::array<ADCG, kInputSize> input{independent[0], independent[1]};
  const auto output = robot_codegen_model(input);
  std::vector<ADCG> dependent(output.begin(), output.end());
  CppAD::ADFun<CG> function(independent, dependent);
  CppAD::cg::ModelCSourceGen<double> source(function, "robot_model");
  source.setCreateForwardZero(true);
  source.setCreateJacobian(true);
  CppAD::cg::ModelLibraryCSourceGen<double> library(source);
  CppAD::cg::DynamicModelLibraryProcessor<double> processor(library, "robot_model_library");
  CppAD::cg::GccCompiler<double> compiler;
  return processor.createDynamicLibrary(compiler);
}
}  // namespace

int main() {
  try {
    auto cppad = record_cppad();
    auto library = compile_codegen();
    auto generated = library->model("robot_model");
    const std::array<std::array<double, kInputSize>, 3> seeds{{
        {{0.25, -0.75}}, {{1.2, 0.4}}, {{-0.9, 1.1}},
    }};
    double worst_output = 0.0;
    double worst_jacobian = 0.0;
    for (const auto& input : seeds) {
      const auto expected = expected_model(input);
      const auto original_array = robot_codegen_model(input);
      const std::vector<double> original(original_array.begin(), original_array.end());
      const std::vector<double> x(input.begin(), input.end());
      const auto ad_output = cppad.Forward(0, x);
      const auto codegen_output = generated->ForwardZero(x);
      worst_output = std::max({worst_output, max_output_error(original, expected),
                               max_output_error(ad_output, expected),
                               max_output_error(codegen_output, expected)});
      const auto finite_difference_jacobian = finite_difference(input);
      worst_jacobian = std::max({worst_jacobian,
                                 relative_error(cppad.Jacobian(x), finite_difference_jacobian),
                                 relative_error(generated->Jacobian(x), finite_difference_jacobian)});
    }
    if (worst_output > OUTPUT_TOLERANCE) {
      std::cerr << "output equivalence failed: max_abs_error=" << worst_output << '\n';
      return 1;
    }
    if (worst_jacobian > JACOBIAN_RELATIVE_TOLERANCE) {
      std::cerr << "Jacobian gate failed: max_relative_error=" << worst_jacobian << '\n';
      return 1;
    }
    std::cout << "CppAD/CodeGen output and Jacobian verified; max_output_error=" << worst_output
              << "; max_jacobian_relative_error=" << worst_jacobian;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CodeGen runtime error: " << error.what() << '\n';
    return 2;
  }
}
