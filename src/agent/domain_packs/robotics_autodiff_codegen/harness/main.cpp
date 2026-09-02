#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cppad/cg.hpp>
#include WORKSPACE_HEADER

#ifndef OUTPUT_ATOL
#define OUTPUT_ATOL 1e-10
#endif
#ifndef OUTPUT_RTOL
#define OUTPUT_RTOL 1e-9
#endif
#ifndef JACOBIAN_ATOL
#define JACOBIAN_ATOL 1e-8
#endif
#ifndef JACOBIAN_RTOL
#define JACOBIAN_RTOL 1e-6
#endif

namespace {
constexpr std::size_t kInputSize = 2;
constexpr std::size_t kOutputSize = 2;
constexpr std::uint64_t kMultiplier = 2685821657736338717ULL;

struct Sample {
  std::string kind;
  std::array<double, kInputSize> input;
};

struct WorstFailure {
  bool present = false;
  std::string quantity;
  std::string comparison;
  std::size_t sample_index = 0;
  std::string sample_kind;
  std::array<double, kInputSize> input{};
  std::size_t row = 0;
  std::size_t column = 0;
  double expected = 0.0;
  double actual = 0.0;
  double absolute_error = 0.0;
  double normalized_error = 0.0;
};

struct Metrics {
  std::size_t failures = 0;
  std::size_t non_finite = 0;
  std::size_t dimension_failures = 0;
  double max_output_absolute_error = 0.0;
  double max_output_normalized_error = 0.0;
  double max_jacobian_absolute_error = 0.0;
  double max_jacobian_normalized_error = 0.0;
  double max_jacobian_frobenius_relative_error = 0.0;
  WorstFailure worst;
};

template <class Scalar>
std::array<Scalar, kOutputSize> expected_model(const std::array<Scalar, kInputSize>& x) {
  return {x[0] * x[0] + Scalar(0.5) * x[1], x[0] * x[1] + x[1] * x[1]};
}

std::vector<double> expected_jacobian(const std::array<double, kInputSize>& x) {
  return {2.0 * x[0], 0.5, x[1], x[0] + 2.0 * x[1]};
}

double next_unit(std::uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  const std::uint64_t result = state * kMultiplier;
  return static_cast<double>(result >> 11) * (1.0 / 9007199254740992.0);
}

std::vector<Sample> make_samples(std::uint64_t seed, std::size_t random_samples) {
  std::vector<Sample> samples{
      {"regression", {0.25, -0.75}}, {"regression", {1.2, 0.4}},
      {"regression", {-0.9, 1.1}}, {"boundary", {0.0, 0.0}},
      {"boundary", {2.0, 4.0}}, {"boundary", {-2.0, -4.0}},
      {"boundary", {2.0, -4.0}}, {"boundary", {-2.0, 4.0}},
  };
  std::uint64_t state = seed == 0 ? 1 : seed;
  for (std::size_t index = 0; index < random_samples; ++index) {
    const double x0 = -2.0 + 4.0 * next_unit(state);
    const double x1 = -4.0 + 8.0 * next_unit(state);
    samples.push_back({"random", {x0, x1}});
  }
  return samples;
}

std::vector<double> finite_difference(const std::array<double, kInputSize>& x) {
  std::vector<double> jacobian(kOutputSize * kInputSize);
  const double scale = std::cbrt(std::numeric_limits<double>::epsilon());
  for (std::size_t column = 0; column < kInputSize; ++column) {
    const double step = scale * std::max(1.0, std::abs(x[column]));
    auto plus = x;
    auto minus = x;
    plus[column] += step;
    minus[column] -= step;
    const auto y_plus = robot_codegen_model(plus);
    const auto y_minus = robot_codegen_model(minus);
    for (std::size_t row = 0; row < kOutputSize; ++row) {
      jacobian[row * kInputSize + column] =
          (y_plus[row] - y_minus[row]) / (2.0 * step);
    }
  }
  return jacobian;
}

double frobenius_relative_error(const std::vector<double>& actual,
                                const std::vector<double>& expected) {
  if (actual.size() != expected.size()) return std::numeric_limits<double>::infinity();
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      return std::numeric_limits<double>::infinity();
    }
    const double difference = actual[index] - expected[index];
    squared_error += difference * difference;
    squared_reference += expected[index] * expected[index];
  }
  return std::sqrt(squared_error) / std::max(1.0, std::sqrt(squared_reference));
}

void compare_values(const std::string& comparison, const Sample& sample,
                    std::size_t sample_index, const std::vector<double>& actual,
                    const std::vector<double>& expected, std::size_t columns,
                    double atol, double rtol, bool jacobian, Metrics& metrics) {
  if (actual.size() != expected.size()) {
    ++metrics.failures;
    ++metrics.dimension_failures;
    return;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double actual_value = actual[index];
    const double expected_value = expected[index];
    const bool finite = std::isfinite(actual_value) && std::isfinite(expected_value);
    const double absolute_error = finite
                                      ? std::abs(actual_value - expected_value)
                                      : std::numeric_limits<double>::infinity();
    const double allowance = atol + rtol * std::abs(expected_value);
    const double normalized_error = finite
                                        ? absolute_error / allowance
                                        : std::numeric_limits<double>::infinity();
    if (!finite) ++metrics.non_finite;
    if (jacobian) {
      metrics.max_jacobian_absolute_error =
          std::max(metrics.max_jacobian_absolute_error, absolute_error);
      metrics.max_jacobian_normalized_error =
          std::max(metrics.max_jacobian_normalized_error, normalized_error);
    } else {
      metrics.max_output_absolute_error =
          std::max(metrics.max_output_absolute_error, absolute_error);
      metrics.max_output_normalized_error =
          std::max(metrics.max_output_normalized_error, normalized_error);
    }
    if (normalized_error <= 1.0) continue;
    ++metrics.failures;
    if (!metrics.worst.present || normalized_error > metrics.worst.normalized_error) {
      metrics.worst = {true, jacobian ? "jacobian" : "output", comparison, sample_index,
                       sample.kind, sample.input, jacobian ? index / columns : index,
                       jacobian ? index % columns : 0, expected_value, actual_value,
                       absolute_error, normalized_error};
    }
  }
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

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

std::string make_report(bool passed, std::uint64_t seed, std::size_t random_samples,
                        const std::vector<Sample>& samples, const Metrics& metrics) {
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"passed\":" << (passed ? "true" : "false")
         << ",\"seed\":" << seed << ",\"random_samples\":" << random_samples
         << ",\"sample_count\":" << samples.size()
         << ",\"thresholds\":{\"output_atol\":" << OUTPUT_ATOL
         << ",\"output_rtol\":" << OUTPUT_RTOL
         << ",\"jacobian_atol\":" << JACOBIAN_ATOL
         << ",\"jacobian_rtol\":" << JACOBIAN_RTOL << "}"
         << ",\"metrics\":{\"failures\":" << metrics.failures
         << ",\"non_finite\":" << metrics.non_finite
         << ",\"dimension_failures\":" << metrics.dimension_failures
         << ",\"max_output_absolute_error\":"
         << json_number(metrics.max_output_absolute_error)
         << ",\"max_output_normalized_error\":"
         << json_number(metrics.max_output_normalized_error)
         << ",\"max_jacobian_absolute_error\":"
         << json_number(metrics.max_jacobian_absolute_error)
         << ",\"max_jacobian_normalized_error\":"
         << json_number(metrics.max_jacobian_normalized_error)
         << ",\"max_jacobian_frobenius_relative_error\":"
         << json_number(metrics.max_jacobian_frobenius_relative_error) << "}"
         << ",\"worst_failure\":";
  if (!metrics.worst.present) {
    output << "null";
  } else {
    const auto& failure = metrics.worst;
    output << "{\"quantity\":\"" << failure.quantity << "\",\"comparison\":\""
           << failure.comparison << "\",\"sample_index\":"
           << failure.sample_index << ",\"sample_kind\":\"" << failure.sample_kind
           << "\",\"input\":[" << failure.input[0] << ',' << failure.input[1]
           << "],\"row\":" << failure.row << ",\"column\":" << failure.column
           << ",\"expected\":" << json_number(failure.expected)
           << ",\"actual\":" << json_number(failure.actual)
           << ",\"absolute_error\":" << json_number(failure.absolute_error)
           << ",\"normalized_error\":" << json_number(failure.normalized_error) << '}';
  }
  output << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) output << ',';
    output << "{\"index\":" << index << ",\"kind\":\"" << samples[index].kind
           << "\",\"input\":[" << samples[index].input[0] << ','
           << samples[index].input[1] << "]}";
  }
  output << "]}";
  return output.str();
}

void emit_report(const std::string& report, const std::string& report_path) {
  std::ofstream file(report_path);
  if (!file) throw std::runtime_error("cannot open dense report path");
  file << report << '\n';
  file.close();
  std::cout << report << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  std::uint64_t seed = 20260902;
  std::size_t random_samples = 32;
  std::string report_path = "dense-validation.json";
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string option = argv[index];
    if (option == "--seed") seed = std::stoull(argv[index + 1]);
    else if (option == "--random-samples") random_samples = std::stoull(argv[index + 1]);
    else if (option == "--report") report_path = argv[index + 1];
    else {
      std::cerr << "unknown option: " << option << '\n';
      return 2;
    }
  }
  try {
    auto cppad = record_cppad();
    auto library = compile_codegen();
    auto generated = library->model("robot_model");
    const auto samples = make_samples(seed, random_samples);
    Metrics metrics;
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      const auto& sample = samples[sample_index];
      const auto expected_array = expected_model(sample.input);
      const std::vector<double> expected(expected_array.begin(), expected_array.end());
      const auto original_array = robot_codegen_model(sample.input);
      const std::vector<double> original(original_array.begin(), original_array.end());
      const std::vector<double> input(sample.input.begin(), sample.input.end());
      const auto ad_output = cppad.Forward(0, input);
      const auto codegen_output = generated->ForwardZero(input);
      compare_values("original_vs_reference", sample, sample_index, original, expected,
                     kOutputSize, OUTPUT_ATOL, OUTPUT_RTOL, false, metrics);
      compare_values("cppad_vs_reference", sample, sample_index, ad_output, expected,
                     kOutputSize, OUTPUT_ATOL, OUTPUT_RTOL, false, metrics);
      compare_values("codegen_vs_reference", sample, sample_index, codegen_output, expected,
                     kOutputSize, OUTPUT_ATOL, OUTPUT_RTOL, false, metrics);
      compare_values("codegen_vs_cppad_output", sample, sample_index, codegen_output, ad_output,
                     kOutputSize, OUTPUT_ATOL, OUTPUT_RTOL, false, metrics);

      const auto analytic = expected_jacobian(sample.input);
      const auto finite_difference_jacobian = finite_difference(sample.input);
      const auto ad_jacobian = cppad.Jacobian(input);
      const auto codegen_jacobian = generated->Jacobian(input);
      const std::array<std::pair<const char*, const std::vector<double>*>, 6> comparisons{{
          {"finite_difference_vs_reference", &finite_difference_jacobian},
          {"cppad_vs_finite_difference", &ad_jacobian},
          {"codegen_vs_finite_difference", &codegen_jacobian},
          {"cppad_vs_reference", &ad_jacobian},
          {"codegen_vs_reference", &codegen_jacobian},
          {"codegen_vs_cppad", &codegen_jacobian},
      }};
      for (const auto& comparison : comparisons) {
        const std::string name = comparison.first;
        const auto& reference = name == "codegen_vs_cppad"
                                    ? ad_jacobian
                                    : (name.find("vs_finite_difference") != std::string::npos
                                           ? finite_difference_jacobian
                                           : analytic);
        compare_values(name, sample, sample_index, *comparison.second, reference,
                       kInputSize, JACOBIAN_ATOL, JACOBIAN_RTOL, true, metrics);
        const double frobenius_error =
            frobenius_relative_error(*comparison.second, reference);
        metrics.max_jacobian_frobenius_relative_error =
            std::max(metrics.max_jacobian_frobenius_relative_error,
                     frobenius_error);
        if (frobenius_error > JACOBIAN_RTOL) ++metrics.failures;
      }
    }
    const bool passed = metrics.failures == 0 && metrics.non_finite == 0 &&
                        metrics.dimension_failures == 0 &&
                        metrics.max_jacobian_frobenius_relative_error <= JACOBIAN_RTOL;
    emit_report(make_report(passed, seed, random_samples, samples, metrics), report_path);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "CodeGen runtime error: " << error.what() << '\n';
    return 2;
  }
}
