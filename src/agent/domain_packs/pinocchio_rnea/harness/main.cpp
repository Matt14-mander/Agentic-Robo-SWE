// Official fixed-base RNEA probe. Candidate code is only the Scalar-generic wrapper.
#include <pinocchio/codegen/cppadcg.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include WORKSPACE_HEADER
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

using Vector = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;

template<class Base>
CppAD::ADFun<Base> record(const pinocchio::Model& original) {
  using AD = CppAD::AD<Base>;
  auto model = original.cast<AD>();
  pinocchio::DataTpl<AD> data(model);
  std::vector<AD> independent(6);
  // Record away from zero; the official samples also test the zero state.
  for (int i = 0; i < 6; ++i) independent[i] = 0.1 * (i + 1);
  CppAD::Independent(independent);
  Eigen::Matrix<AD, Eigen::Dynamic, 1> x(6);
  for (int i = 0; i < 6; ++i) x[i] = independent[i];
  const auto output = candidate_rnea<AD>(model, data, x);
  if (output.size() != 2) throw std::runtime_error("candidate output dimension must be 2");
  std::vector<AD> dependent{output[0], output[1]};
  return CppAD::ADFun<Base>(independent, dependent);
}

template<class Values>
std::string array_json(const Values& values) {
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (std::size_t i = 0; i < static_cast<std::size_t>(values.size()); ++i) {
    if (i) out << ',';
    if (std::isfinite(values[i])) out << values[i]; else out << "null";
  }
  out << ']';
  return out.str();
}

Vector native(const pinocchio::Model& model, pinocchio::Data& data, const Vector& x) {
  return pinocchio::rnea(model, data, x.segment(0, 2), x.segment(2, 2), x.segment(4, 2));
}

std::vector<double> analytic(const pinocchio::Model& model, pinocchio::Data& data, const Vector& x) {
  data.dtau_dq.setZero(); data.dtau_dv.setZero(); data.M.setZero();
  pinocchio::computeRNEADerivatives(model, data, x.segment(0, 2), x.segment(2, 2), x.segment(4, 2));
  // Pinocchio only fills the upper triangle of d(tau)/d(a) = M.
  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  std::vector<double> jac(12);
  for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c) {
    jac[r * 6 + c] = data.dtau_dq(r, c);
    jac[r * 6 + c + 2] = data.dtau_dv(r, c);
    jac[r * 6 + c + 4] = data.M(r, c);
  }
  return jac;
}

std::vector<double> finite_difference(const pinocchio::Model& model, pinocchio::Data& data, const Vector& x) {
  std::vector<double> jac(12);
  for (int c = 0; c < 6; ++c) {
    const double h = 1e-5 * std::max(1.0, std::abs(x[c]));
    Vector plus = x, minus = x; plus[c] += h; minus[c] -= h;
    const Vector yp = native(model, data, plus), ym = native(model, data, minus);
    for (int r = 0; r < 2; ++r) jac[r * 6 + c] = (yp[r] - ym[r]) / (2 * h);
  }
  return jac;
}

struct LoopResult {
  Vector state;
  double checksum = 0;
  std::vector<double> checkpoints;
};

LoopResult control_loop(const pinocchio::Model& model,
                        CppAD::cg::GenericModel<double>& generated,
                        bool use_codegen, int steps, bool keep_checkpoints) {
  pinocchio::Data dynamics_data(model), plant_data(model);
  Vector q(2), v(2);
  q << 0.35, -0.55;
  v << 0.15, -0.1;
  LoopResult result;
  for (int step = 0; step < steps; ++step) {
    const double time = step * 0.001;
    Vector q_desired(2), v_desired(2), a_feedforward(2);
    q_desired << 0.6 * std::sin(1.2 * time), -0.45 * std::cos(0.8 * time);
    v_desired << 0.72 * std::cos(1.2 * time), 0.36 * std::sin(0.8 * time);
    a_feedforward << -0.864 * std::sin(1.2 * time), 0.288 * std::cos(0.8 * time);
    const Vector acceleration = a_feedforward + 35.0 * (q_desired - q) + 8.0 * (v_desired - v);
    Vector x(6);
    x << q, v, acceleration;
    Vector torque(2);
    std::vector<double> jacobian;
    if (use_codegen) {
      std::vector<double> input(x.data(), x.data() + x.size());
      const auto output = generated.ForwardZero(input);
      jacobian = generated.Jacobian(input);
      torque << output[0], output[1];
    } else {
      jacobian = analytic(model, dynamics_data, x);
      torque = dynamics_data.tau;
    }
    const Vector realized = pinocchio::aba(model, plant_data, q, v, torque);
    v.noalias() += 0.001 * realized;
    q.noalias() += 0.001 * v;
    result.checksum += torque.sum() + realized.sum()
        + std::accumulate(jacobian.begin(), jacobian.end(), 0.0);
    if (keep_checkpoints && (step % 64 == 63 || step == steps - 1)) {
      result.checkpoints.insert(result.checkpoints.end(), q.data(), q.data() + q.size());
      result.checkpoints.insert(result.checkpoints.end(), v.data(), v.data() + v.size());
      result.checkpoints.insert(result.checkpoints.end(), torque.data(), torque.data() + torque.size());
    }
  }
  result.state.resize(4);
  result.state << q, v;
  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error("usage: validator samples.txt report.json [--reuse-library]");
    if (argc == 4 && std::string(argv[3]) != "--reuse-library")
      throw std::runtime_error("usage: validator samples.txt report.json [--reuse-library]");
    const bool reuse_library = argc == 4;
    pinocchio::Model model;
    pinocchio::urdf::buildModel(ROBOT_URDF, model);
    model.gravity.linear() << 0, 0, -9.81;
    if (model.nq != 2 || model.nv != 2) throw std::runtime_error("only fixed-base nq=nv=2 supported");
    pinocchio::Data ref_data(model), source_data(model);
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("sample file missing");
    std::vector<Vector> points;
    for (int row = 0; row < 24; ++row) {
      Vector x(6);
      for (int col = 0; col < 6; ++col)
        if (!(input >> x[col])) throw std::runtime_error("incomplete samples");
      points.push_back(x);
    }
    auto ad = record<double>(model);
    const auto compile_start = Clock::now();
    std::unique_ptr<CppAD::cg::DynamicLib<double>> library;
    if (reuse_library) {
      library.reset(new CppAD::cg::LinuxDynamicLib<double>("rnea_library.so"));
    } else {
      auto cg_tape = record<CppAD::cg::CG<double>>(model);
      CppAD::cg::ModelCSourceGen<double> source(cg_tape, "rnea_model");
      source.setCreateForwardZero(true); source.setCreateJacobian(true);
      CppAD::cg::ModelLibraryCSourceGen<double> library_source(source);
      CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_source, "rnea_library");
      CppAD::cg::GccCompiler<double> compiler;
      compiler.setCompileFlags({"-O3", "-DNDEBUG"});
      library = processor.createDynamicLibrary(compiler);
    }
    auto generated = library->model("rnea_model");
    const double setup_seconds = std::chrono::duration<double>(Clock::now() - compile_start).count();
    const double compile_seconds = reuse_library ? 0.0 : setup_seconds;
    const double library_load_seconds = reuse_library ? setup_seconds : 0.0;
    std::ostringstream report;
    report << std::setprecision(17) << "{\"schema_version\":1,\"nq\":2,\"nv\":2,\"urdf_sha256\":\""
           << URDF_SHA256 << "\",\"samples\":[";
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto& x = points[i];
      std::vector<double> values(x.data(), x.data() + x.size());
      if (i) report << ',';
      report << "{\"input\":" << array_json(x)
        << ",\"pinocchio_output\":" << array_json(native(model, ref_data, x))
        << ",\"source_output\":" << array_json(candidate_rnea<double>(model, source_data, x))
        << ",\"cppad_output\":" << array_json(ad.Forward(0, values))
        << ",\"codegen_output\":" << array_json(generated->ForwardZero(values))
        << ",\"analytic_jacobian\":" << array_json(analytic(model, ref_data, x))
        << ",\"analytic_output\":" << array_json(ref_data.tau)
        << ",\"finite_difference_jacobian\":" << array_json(finite_difference(model, ref_data, x))
        << ",\"cppad_jacobian\":" << array_json(ad.Jacobian(values))
        << ",\"codegen_jacobian\":" << array_json(generated->Jacobian(values)) << '}';
    }
    // Identical workload: output + full 2x6 Jacobian. Prebuild inputs, reuse models/data.
    std::vector<std::vector<double>> flat;
    for (const auto& x : points) flat.emplace_back(x.data(), x.data() + x.size());
    volatile double checksum = 0;
    std::array<std::function<void(std::size_t)>, 4> operations{{
      // computeRNEADerivatives already computes tau: avoid charging the analytic
      // baseline a redundant RNEA evaluation just to retrieve the output.
      [&](std::size_t i) { auto j = analytic(model, ref_data, points[i]); checksum += ref_data.tau.sum() + std::accumulate(j.begin(), j.end(), 0.0); },
      [&](std::size_t i) { auto y = ad.Forward(0, flat[i]); auto j = ad.Jacobian(flat[i]); checksum += y[0] + y[1] + std::accumulate(j.begin(), j.end(), 0.0); },
      [&](std::size_t i) { auto y = generated->ForwardZero(flat[i]); auto j = generated->Jacobian(flat[i]); checksum += y[0] + y[1] + std::accumulate(j.begin(), j.end(), 0.0); },
      [&](std::size_t i) { auto y = native(model, ref_data, points[i]); auto j = finite_difference(model, ref_data, points[i]); checksum += y.sum() + std::accumulate(j.begin(), j.end(), 0.0); }
    }};
    for (int k = 0; k < 32; ++k) for (auto& op : operations) op(k % points.size());
    std::array<std::vector<double>, 4> timings;
    for (int round = 0; round < 50; ++round) for (int slot = 0; slot < 4; ++slot) {
      const int backend = (round + slot) % 4; // Rotate to reduce systematic thermal/order bias.
      const auto start = Clock::now();
      for (int batch = 0; batch < 16; ++batch) operations[backend]((round * 16 + batch) % points.size());
      timings[backend].push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count() / 16);
    }
    const std::array<const char*, 4> names{{"pinocchio", "cppad", "codegen", "finite_difference"}};
    report << "],\"timing_spec\":{\"warmup\":32,\"repeats\":50,\"batch_size\":16},\"timings_ns\":{";
    for (int i = 0; i < 4; ++i) {
      if (i) report << ',';
      report << '\"' << names[i] << "\":" << array_json(timings[i]);
    }
    // A small closed-loop workload: both backends share controller, ABA plant and integrator.
    // This measures Amdahl-diluted benefit, not an MPC/Crocoddyl solver.
    for (int warmup = 0; warmup < 4; ++warmup) {
      checksum += control_loop(model, *generated, false, 512, false).checksum;
      checksum += control_loop(model, *generated, true, 512, false).checksum;
    }
    std::array<std::vector<double>, 2> loop_timings;
    for (int repeat = 0; repeat < 30; ++repeat) for (int slot = 0; slot < 2; ++slot) {
      const int backend = (repeat + slot) % 2;
      const auto start = Clock::now();
      const auto loop = control_loop(model, *generated, backend == 1, 512, false);
      loop_timings[backend].push_back(
          std::chrono::duration<double, std::nano>(Clock::now() - start).count() / 512);
      checksum += loop.checksum;
    }
    const auto analytic_loop = control_loop(model, *generated, false, 512, true);
    const auto codegen_loop = control_loop(model, *generated, true, 512, true);
    report << "},\"codegen_compile_seconds\":" << compile_seconds
      << ",\"library_load_seconds\":" << library_load_seconds
      << ",\"library_reused\":" << (reuse_library ? "true" : "false")
      << ",\"control_loop\":{\"spec\":{\"warmup\":4,\"repeats\":30,"
         "\"steps\":512,\"dt\":0.001,\"deadline_ns\":1000000},\"timings_ns\":{"
      << "\"pinocchio\":" << array_json(loop_timings[0])
      << ",\"codegen\":" << array_json(loop_timings[1]) << "},\"pinocchio\":{"
      << "\"final_state\":" << array_json(analytic_loop.state)
      << ",\"checkpoints\":" << array_json(analytic_loop.checkpoints)
      << ",\"checksum\":" << analytic_loop.checksum << "},\"codegen\":{"
      << "\"final_state\":" << array_json(codegen_loop.state)
      << ",\"checkpoints\":" << array_json(codegen_loop.checkpoints)
      << ",\"checksum\":" << codegen_loop.checksum << "}}}";
    if (!std::isfinite(checksum)) throw std::runtime_error("nonfinite benchmark checksum");
    std::ofstream out(argv[2]); out << report.str() << '\n';
    if (!out) throw std::runtime_error("could not write structured report");
    std::cout << "RNEA observations written; Python must independently verify correctness\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
