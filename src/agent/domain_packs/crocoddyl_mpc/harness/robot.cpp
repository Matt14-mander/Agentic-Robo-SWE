#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/numdiff/action.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>
#include <map>
#include <algorithm>
#include "aba_codegen.hpp"
#ifdef ROBO_SWE_CODEGEN_TRIAL
#include "aba_action.hpp"
#endif

namespace {
using V = Eigen::VectorXd;
using Trajectory = std::vector<V>;
constexpr int H = 40, STEPS = 32, REPEATS = 10, WARMUP = 2, MAXITER = 100;
constexpr double DT = 0.02;
volatile double sink = 0;
V initial() { V x(4); x << 0.4, -0.3, 0., 0.; return x; }

struct Bundle {
  std::shared_ptr<crocoddyl::ShootingProblem> problem;
  std::shared_ptr<crocoddyl::ActionModelAbstract> running;
  V gravity;
};

Bundle make_problem(const pinocchio::Model& model,
                    std::shared_ptr<AbaBackend> backend={},bool generated=false) {
  auto state = std::make_shared<crocoddyl::StateMultibody>(
      std::make_shared<pinocchio::Model>(model));
  auto actuation = std::make_shared<crocoddyl::ActuationModelFull>(state);
  pinocchio::Data data(model);
  V zero = V::Zero(2);
  V gravity = pinocchio::rnea(model, data, zero, zero, zero);
  auto make_action = [&](bool terminal) {
    auto costs = std::make_shared<crocoddyl::CostModelSum>(state, 2);
    V weights(4); weights << 100., 100., 5., 5.;
    auto activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
    costs->addCost("state", std::make_shared<crocoddyl::CostModelResidual>(state,
        activation, std::make_shared<crocoddyl::ResidualModelState>(state, V::Zero(4), 2)),
        terminal ? 20. : 1.);
    if (!terminal) costs->addCost("effort", std::make_shared<crocoddyl::CostModelResidual>(
        state, std::make_shared<crocoddyl::ResidualModelControl>(state, gravity)), 0.001);
    std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dam =
        std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
        state, actuation, costs);
#ifdef ROBO_SWE_CODEGEN_TRIAL
    if(backend && !terminal)
      dam=std::make_shared<AbaDerivativeAction>(state,actuation,costs,backend,generated);
#endif
    return std::make_shared<crocoddyl::IntegratedActionModelEuler>(dam, terminal ? 0. : DT);
  };
  auto running = make_action(false);
  auto terminal = make_action(true);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(H, running);
  auto problem = std::make_shared<crocoddyl::ShootingProblem>(initial(), models, terminal);
  return {problem, running, gravity};
}

double measure(const std::function<void()>& action) {
  auto start = std::chrono::steady_clock::now(); action();
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now()-start).count();
}

void vector_json(std::ostream& out, const V& x) {
  out << '[';
  for (Eigen::Index i=0; i<x.size(); ++i) { if(i) out << ','; out << x[i]; }
  out << ']';
}
void trajectory_json(std::ostream& out, const Trajectory& xs) {
  out << '['; for(std::size_t i=0;i<xs.size();++i) {
    if(i) out << ','; vector_json(out,xs[i]); } out << ']';
}

struct Run {
  Trajectory states, controls;
  std::vector<double> costs, initial_costs, residuals, solve_ns;
  std::vector<int> iterations;
  bool all_solved = true;
};
Run run_mpc(const pinocchio::Model& model, bool warm,
            std::shared_ptr<AbaBackend> backend={},bool generated=false) {
  auto bundle = make_problem(model,backend,generated);
  crocoddyl::SolverFDDP solver(bundle.problem);
  solver.set_th_stop(1e-6);
  pinocchio::Data plant(model);
  V x = initial();
  Trajectory us(H, bundle.gravity);
  Run result; result.states.push_back(x);
  for(int step=0; step<STEPS; ++step) {
    bundle.problem->set_x0(x);
    if(!warm) us.assign(H, bundle.gravity);
    auto xs = bundle.problem->rollout_us(us);
    result.initial_costs.push_back(bundle.problem->calc(xs,us));
    bool solved = false;
    result.solve_ns.push_back(measure([&] {
      solved = solver.solve(xs,us,MAXITER,true,1e-9);
    }));
    result.all_solved = result.all_solved && solved;
    result.iterations.push_back(static_cast<int>(solver.get_iter()+1));
    result.costs.push_back(solver.get_cost());
    result.controls.push_back(solver.get_us().front());
    auto replay = bundle.problem->rollout_us(solver.get_us());
    double residual = 0;
    for(int i=0;i<=H;++i) residual = std::max(residual,
        (replay[i]-solver.get_xs()[i]).cwiseAbs().maxCoeff());
    result.residuals.push_back(residual);
    V a = pinocchio::aba(model,plant,x.head(2),x.tail(2),result.controls.back());
    x.tail(2) += DT*a; x.head(2) += DT*x.tail(2);
    result.states.push_back(x);
    us = solver.get_us();
    if(warm) { std::rotate(us.begin(),us.begin()+1,us.end()); us.back()=us[H-2]; }
  }
  return result;
}

void run_json(std::ostream& out,const Run& r) {
  out << "{\"states\":"; trajectory_json(out,r.states);
  out << ",\"controls\":"; trajectory_json(out,r.controls);
  auto numbers = [&](const char* key,const auto& xs) {
    out << ",\"" << key << "\":[";
    for(std::size_t i=0;i<xs.size();++i) { if(i) out << ','; out << xs[i]; } out << ']';
  };
  numbers("costs",r.costs); numbers("initial_costs",r.initial_costs);
  numbers("rollout_errors",r.residuals); numbers("solve_ns",r.solve_ns);
  numbers("iterations",r.iterations);
  out << ",\"all_solved\":" << (r.all_solved?"true":"false") << '}';
}
}

#ifdef ROBO_SWE_CODEGEN_TRIAL
#include "trial_main.hpp"
#else
int main(int argc,char** argv) {
  if(argc!=3) { std::cerr << "usage: robot OUTPUT CONTRACT\n"; return 2; }
  try {
    pinocchio::Model model; pinocchio::urdf::buildModel(ROBOT_URDF,model);
    if(model.nq!=2 || model.nv!=2) throw std::runtime_error("wrong robot dimensions");
    auto b=make_problem(model);
    crocoddyl::SolverFDDP solver(b.problem); solver.set_th_stop(1e-6);
    Trajectory us(H,b.gravity), xs=b.problem->rollout_us(us);
    // Check the integrated action against finite differences at a fixed nonzero input.
    V x=initial(), u=b.gravity; u[0]+=0.2; u[1]-=0.1;
    auto data=b.running->createData(); b.running->calc(data,x,u); b.running->calcDiff(data,x,u);
    crocoddyl::ActionModelNumDiff numerical(b.running,false);
    auto nd=numerical.createData(); numerical.calc(nd,x,u); numerical.calcDiff(nd,x,u);
    double derivative_error = std::max({(data->Fx-nd->Fx).cwiseAbs().maxCoeff(),
        (data->Fu-nd->Fu).cwiseAbs().maxCoeff(),(data->Lx-nd->Lx).cwiseAbs().maxCoeff(),
        (data->Lu-nd->Lu).cwiseAbs().maxCoeff()});
    std::map<std::string,std::vector<double>> stages;
    for(int i=-WARMUP;i<REPEATS;++i) {
      auto save = [&](const std::string& name,const std::function<void()>& action) {
        double ns=measure(action); if(i>=0) stages[name].push_back(ns);
      };
      save("calc",[&]{sink+=b.problem->calc(xs,us);});
      solver.setCandidate(xs,us,true);
      save("calc_diff",[&]{sink+=solver.calcDiff();});
      save("backward_pass",[&]{solver.backwardPass();});
      save("try_step",[&]{sink+=solver.tryStep(1.);});
      save("solve",[&]{sink+=solver.solve(xs,us,MAXITER,true,1e-9)?1.:0.;});
      auto rolled=xs;
      save("rollout",[&]{b.problem->rollout(us,rolled);sink+=rolled.back().sum();});
    }
    std::vector<double> cold_ns,warm_ns;
    std::vector<Run> cold_runs,warm_runs;
    Run cold,warm;
    for(int i=-WARMUP;i<REPEATS;++i) {
      // Alternate order to reduce systematic thermal/order bias.
      double c=0,w=0;
      auto run_c=[&]{c=measure([&]{cold=run_mpc(model,false);});};
      auto run_w=[&]{w=measure([&]{warm=run_mpc(model,true);});};
      if(i%2==0) {run_c();run_w();} else {run_w();run_c();}
      if(i>=0) {cold_ns.push_back(c);warm_ns.push_back(w);
        cold_runs.push_back(cold);warm_runs.push_back(warm);}
    }
    std::ifstream contract_file(argv[2]);
    std::string contract((std::istreambuf_iterator<char>(contract_file)),{});
    if(contract.empty()) throw std::runtime_error("missing contract");
    std::ofstream out(argv[1]); if(!out) throw std::runtime_error("cannot open output");
    out << std::setprecision(17) << "{\"schema_version\":2,\"contract\":" << contract
        << ",\"derivative_max_abs_error\":" << derivative_error << ",\"runs\":{\"cold\":";
    auto write_runs=[&](const std::vector<Run>& runs) {
      out << '['; for(std::size_t i=0;i<runs.size();++i) {
        if(i) out << ',';run_json(out,runs[i]); } out << ']';
    };
    write_runs(cold_runs); out << ",\"shifted_warm\":"; write_runs(warm_runs);
    out << "},\"stage_timings_ns\":{";
    bool first=true;
    for(const auto& item:stages) {
      if(!first) out << ','; first=false; out << '"' << item.first << "\":";
      V values=Eigen::Map<const V>(item.second.data(),item.second.size()); vector_json(out,values);
    }
    out << "},\"mpc_timings_ns\":{\"cold\":";
    vector_json(out,Eigen::Map<const V>(cold_ns.data(),cold_ns.size()));
    out << ",\"shifted_warm\":";
    vector_json(out,Eigen::Map<const V>(warm_ns.data(),warm_ns.size())); out << "}}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
#endif
