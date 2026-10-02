#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

namespace {

constexpr std::size_t kNx = 4;
constexpr std::size_t kNu = 2;
constexpr std::size_t kHorizon = 40;
constexpr std::size_t kMaxIterations = 20;
constexpr std::size_t kWarmup = 5;
constexpr std::size_t kRepeats = 30;
volatile double sink = 0.0;

struct ProblemBundle {
  std::shared_ptr<crocoddyl::ShootingProblem> problem;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
};

std::shared_ptr<crocoddyl::ActionModelAbstract> make_lqr(bool terminal) {
  constexpr double dt = 0.05;
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(kNx, kNx);
  A(0, 2) = dt;
  A(1, 3) = dt;
  A(2, 2) = 0.98;
  A(3, 3) = 0.98;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(kNx, kNu);
  B(0, 0) = 0.5 * dt * dt;
  B(1, 1) = 0.5 * dt * dt;
  B(2, 0) = dt;
  B(3, 1) = dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kNx, kNx);
  Q.diagonal() << 20.0, 20.0, 1.0, 1.0;
  if (terminal) Q *= 25.0;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(kNu, kNu) * 0.1;
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(kNx, kNu);
  return std::make_shared<crocoddyl::ActionModelLQR>(A, B, Q, R, N);
}

ProblemBundle make_problem(bool numerical_derivatives) {
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running;
  running.reserve(kHorizon);
  for (std::size_t index = 0; index < kHorizon; ++index) {
    auto model = make_lqr(false);
    std::shared_ptr<crocoddyl::ActionModelAbstract> selected = model;
    if (numerical_derivatives) {
      selected = std::make_shared<crocoddyl::ActionModelNumDiff>(model, false);
    }
    running.push_back(selected);
  }
  auto terminal = make_lqr(true);
  std::shared_ptr<crocoddyl::ActionModelAbstract> terminal_model = terminal;
  if (numerical_derivatives) {
    terminal_model = std::make_shared<crocoddyl::ActionModelNumDiff>(terminal, false);
  }
  Eigen::VectorXd x0(kNx);
  x0 << 0.8, -0.6, 0.2, -0.15;
  auto problem = std::make_shared<crocoddyl::ShootingProblem>(x0, running, terminal_model);
  std::vector<Eigen::VectorXd> us(kHorizon, Eigen::VectorXd::Zero(kNu));
  std::vector<Eigen::VectorXd> xs = problem->rollout_us(us);
  return {problem, xs, us};
}

std::vector<double> measure(const std::function<void()>& operation) {
  for (std::size_t index = 0; index < kWarmup; ++index) operation();
  std::vector<double> values;
  values.reserve(kRepeats);
  for (std::size_t index = 0; index < kRepeats; ++index) {
    const auto started = std::chrono::steady_clock::now();
    operation();
    const auto ended = std::chrono::steady_clock::now();
    values.push_back(std::chrono::duration<double, std::nano>(ended - started).count());
  }
  return values;
}

std::vector<double> measure_prepared(const std::function<void()>& prepare,
                                     const std::function<void()>& operation) {
  for (std::size_t index = 0; index < kWarmup; ++index) {
    prepare();
    operation();
  }
  std::vector<double> values;
  values.reserve(kRepeats);
  for (std::size_t index = 0; index < kRepeats; ++index) {
    prepare();
    const auto started = std::chrono::steady_clock::now();
    operation();
    const auto ended = std::chrono::steady_clock::now();
    values.push_back(std::chrono::duration<double, std::nano>(ended - started).count());
  }
  return values;
}

double max_abs(const Eigen::MatrixXd& value) {
  return value.size() == 0 ? 0.0 : value.cwiseAbs().maxCoeff();
}

double derivative_error() {
  auto analytic = make_lqr(false);
  auto numerical = std::make_shared<crocoddyl::ActionModelNumDiff>(analytic, false);
  auto analytic_data = analytic->createData();
  auto numerical_data = numerical->createData();
  Eigen::VectorXd x(kNx);
  x << 0.31, -0.27, 0.42, -0.19;
  Eigen::VectorXd u(kNu);
  u << 0.17, -0.23;
  analytic->calc(analytic_data, x, u);
  analytic->calcDiff(analytic_data, x, u);
  numerical->calc(numerical_data, x, u);
  numerical->calcDiff(numerical_data, x, u);
  double error = 0.0;
  error = std::max(error, max_abs(analytic_data->Fx - numerical_data->Fx));
  error = std::max(error, max_abs(analytic_data->Fu - numerical_data->Fu));
  error = std::max(error, max_abs(analytic_data->Lx - numerical_data->Lx));
  error = std::max(error, max_abs(analytic_data->Lu - numerical_data->Lu));
  return error;
}

void write_values(std::ostream& output, const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: crocoddyl_mpc_profile OUTPUT.json\n";
    return 2;
  }
  try {
    auto analytic = make_problem(false);
    auto numdiff = make_problem(true);
    const double initial_cost = analytic.problem->calc(analytic.xs, analytic.us);
    crocoddyl::SolverFDDP solved(analytic.problem);
    const bool converged = solved.solve(
        analytic.xs, analytic.us, kMaxIterations, true, 1e-9);
    const double final_cost = solved.get_cost();
    const double final_state_norm = solved.get_xs().back().norm();
    std::vector<Eigen::VectorXd> replay = analytic.xs;
    analytic.problem->rollout(solved.get_us(), replay);
    double trajectory_error = 0.0;
    for (std::size_t index = 0; index < replay.size(); ++index) {
      trajectory_error = std::max(
          trajectory_error, (replay[index] - solved.get_xs()[index]).cwiseAbs().maxCoeff());
    }

    crocoddyl::SolverFDDP kernel_solver(analytic.problem);
    auto analytic_calc = measure([&] {
      sink += analytic.problem->calc(analytic.xs, analytic.us);
    });
    auto analytic_calc_diff = measure_prepared([&] {
      kernel_solver.setCandidate(analytic.xs, analytic.us, true);
    }, [&] {
      sink += kernel_solver.calcDiff();
    });
    auto prepare_direction = [&] {
      kernel_solver.setCandidate(analytic.xs, analytic.us, true);
      kernel_solver.calcDiff();
    };
    auto analytic_backward = measure_prepared(prepare_direction, [&] {
      kernel_solver.backwardPass();
      sink += kernel_solver.get_cost();
    });
    auto analytic_try_step = measure_prepared([&] {
      kernel_solver.setCandidate(analytic.xs, analytic.us, true);
      kernel_solver.calcDiff();
      kernel_solver.backwardPass();
    }, [&] {
      sink += kernel_solver.tryStep(1.0);
    });
    crocoddyl::SolverFDDP repeated_solver(analytic.problem);
    auto analytic_solve = measure([&] {
      sink += repeated_solver.solve(
                  analytic.xs, analytic.us, kMaxIterations, true, 1e-9)
                  ? 1.0
                  : 0.0;
    });
    std::vector<Eigen::VectorXd> rolled = analytic.xs;
    auto analytic_rollout = measure([&] {
      analytic.problem->rollout(analytic.us, rolled);
      sink += rolled.back().squaredNorm();
    });
    crocoddyl::SolverFDDP numdiff_solver(numdiff.problem);
    auto numdiff_calc_diff = measure_prepared([&] {
      numdiff_solver.setCandidate(numdiff.xs, numdiff.us, true);
    }, [&] {
      sink += numdiff_solver.calcDiff();
    });

    std::ofstream output(argv[1]);
    if (!output) throw std::runtime_error("cannot open output path");
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"contract\": {"
           << "\"name\":\"phase64_lqr_fddp_profile\","
           << "\"state_dim\":4,\"control_dim\":2,\"horizon\":40,"
           << "\"solver\":\"SolverFDDP\",\"max_iterations\":20,"
           << "\"warmup\":5,\"repeats\":30,"
           << "\"timing_keys\":[\"analytic_calc\",\"analytic_calc_diff\","
           << "\"analytic_backward_pass\",\"analytic_try_step\","
           << "\"analytic_solve\",\"analytic_rollout\",\"numdiff_calc_diff\"],"
           << "\"comparison\":\"analytic ActionModelLQR vs ActionModelNumDiff first derivatives\","
           << "\"codegen_replacement\":false},\n"
           << "  \"correctness\": {"
           << "\"derivative_max_abs_error\":" << derivative_error() << ','
           << "\"analytic_initial_cost\":" << initial_cost << ','
           << "\"analytic_final_cost\":" << final_cost << ','
           << "\"analytic_final_state_norm\":" << final_state_norm << ','
           << "\"trajectory_max_abs_error\":" << trajectory_error << ','
           << "\"analytic_solved\":" << (converged ? "true" : "false") << ','
           << "\"analytic_iterations\":" << solved.get_iter() << "},\n"
           << "  \"timing_spec\": {\"warmup\":5,\"repeats\":30},\n"
           << "  \"timings_ns\": {\n";
    const std::vector<std::pair<std::string, const std::vector<double>*>> series = {
        {"analytic_calc", &analytic_calc},
        {"analytic_calc_diff", &analytic_calc_diff},
        {"analytic_backward_pass", &analytic_backward},
        {"analytic_try_step", &analytic_try_step},
        {"analytic_solve", &analytic_solve},
        {"analytic_rollout", &analytic_rollout},
        {"numdiff_calc_diff", &numdiff_calc_diff},
    };
    for (std::size_t index = 0; index < series.size(); ++index) {
      output << "    \"" << series[index].first << "\":";
      write_values(output, *series[index].second);
      output << (index + 1 == series.size() ? "\n" : ",\n");
    }
    output << "  }\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
