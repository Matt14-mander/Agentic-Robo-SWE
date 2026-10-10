#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <unistd.h>

namespace {
template<class Matrix> void flat_json(std::ostream& out,const Matrix& m) {
  out << '['; bool first=true;
  for(int r=0;r<m.rows();++r) for(int c=0;c<m.cols();++c) {
    if(!first) out << ','; first=false; out << m(r,c);
  } out << ']';
}
void action_json(std::ostream& out,const std::shared_ptr<crocoddyl::ActionDataAbstract>& d) {
  out << "{\"xnext\":";flat_json(out,d->xnext);
  out << ",\"Fx\":";flat_json(out,d->Fx);
  out << ",\"Fu\":";flat_json(out,d->Fu);
  out << ",\"Lx\":";flat_json(out,d->Lx);
  out << ",\"Lu\":";flat_json(out,d->Lu);
  out << ",\"Lxx\":";flat_json(out,d->Lxx);
  out << ",\"Lxu\":";flat_json(out,d->Lxu);
  out << ",\"Luu\":";flat_json(out,d->Luu);
  out << ",\"cost\":" << d->cost << '}';
}
}
int main(int argc,char** argv) {
  if(argc!=4) {std::cerr << "usage: trial OUTPUT CONTRACT LIBRARY_BASE\n";return 2;}
  try {
    pinocchio::Model model;pinocchio::urdf::buildModel(ROBOT_URDF,model);
    std::cerr << "[trial] recording ABA and compiling unique CodeGen library\n";
    auto backend=std::make_shared<AbaBackend>(model,argv[3]);
    std::array<Bundle,3> bundles{make_problem(model),make_problem(model,backend,false),
                                make_problem(model,backend,true)};
    const std::array<std::string,3> names{"native","cppad","codegen"};
    std::ifstream cf(argv[2]); std::string contract((std::istreambuf_iterator<char>(cf)),{});
    if(contract.empty()) throw std::runtime_error("missing contract");
    std::ofstream out(argv[1]);if(!out) throw std::runtime_error("cannot open output");
    out << std::setprecision(17) << "{\"schema_version\":3,\"process_id\":" << getpid()
        << ",\"contract\":" << contract
        << ",\"generation_compile_load_seconds\":" << backend->compile_seconds()
        << ",\"samples\":[";
    std::cerr << "[trial] 25 fixed action and derivative checks\n";
    for(int i=0;i<25;++i) {
      V x(4),u(2); std::vector<double> input(6);
      for(int j=0;j<6;++j) input[j]=i==24?0.:0.6*std::sin((i+1)*(j+1)*0.7);
      if(i==24) {input[4]=-16.4808;input[5]=-2.7468;}
      for(int j=0;j<4;++j) x[j]=input[j];for(int j=0;j<2;++j) u[j]=input[j+4];
      pinocchio::Data d(model);
      V value=pinocchio::aba(model,d,x.head(2),x.tail(2),u);
      pinocchio::computeABADerivatives(model,d,x.head(2),x.tail(2),u);
      Eigen::Matrix<double,2,6> jac;
      d.Minv.template triangularView<Eigen::StrictlyLower>() =
          d.Minv.transpose().template triangularView<Eigen::StrictlyLower>();
      jac << d.ddq_dq,d.ddq_dv,d.Minv;
      if(i) out << ',';
      out << "{\"input\":";vector_json(out,Eigen::Map<V>(input.data(),6));
      out << ",\"backends\":{";
      for(int b=0;b<3;++b) {
        if(b) out << ',';out << '"' << names[b] << "\":{\"value\":";
        if(b==0) flat_json(out,value);
        else {auto v=backend->forward(input,b==2);vector_json(out,Eigen::Map<V>(v.data(),2));}
        out << ",\"jacobian\":";
        if(b==0) flat_json(out,jac);
        else {auto j=backend->jacobian(input,b==2);vector_json(out,Eigen::Map<V>(j.data(),12));}
        auto data=bundles[b].running->createData();
        bundles[b].running->calc(data,x,u);bundles[b].running->calcDiff(data,x,u);
        out << ",\"action\":";action_json(out,data);
        auto terminal=bundles[b].problem->get_terminalModel();auto td=terminal->createData();
        terminal->calc(td,x);terminal->calcDiff(td,x);
        out << ",\"terminal\":";action_json(out,td);out << '}';
      } out << "}}";
    }
    std::map<std::string,std::vector<double>> stages,totals;
    std::array<std::vector<Run>,3> runs;
    for(int repeat=-WARMUP;repeat<REPEATS;++repeat) {
      std::cerr << "[trial] repetition " << repeat+1 << '/' << REPEATS << " (negative/zero=warmup)\n";
      for(int offset=0;offset<3;++offset) {
        int b=((repeat+WARMUP)%3+offset)%3;
        auto& bundle=bundles[b];Trajectory us(H,bundle.gravity);
        auto xs=bundle.problem->rollout_us(us);
        crocoddyl::SolverFDDP solver(bundle.problem);solver.set_th_stop(1e-6);
        solver.setCandidate(xs,us,true);
        double diff=measure([&]{sink+=solver.calcDiff();});
        Run run;double total=measure([&]{run=run_mpc(model,true,
            b==0?std::shared_ptr<AbaBackend>{}:backend,b==2);});
        if(repeat>=0) {stages[names[b]].push_back(diff);totals[names[b]].push_back(total);
                        runs[b].push_back(std::move(run));}
      }
    }
    out << "],\"runs\":{";
    for(int b=0;b<3;++b) {if(b) out << ',';out << '"' << names[b] << "\":[";
      for(int i=0;i<REPEATS;++i) {if(i) out << ',';run_json(out,runs[b][i]);}out << ']';}
    out << "},\"calc_diff_timings_ns\":{";
    auto timings=[&](const auto& section) {bool first=true;for(const auto& pair:section) {
      if(!first) out << ',';first=false;out << '"' << pair.first << "\":";
      vector_json(out,Eigen::Map<const V>(pair.second.data(),pair.second.size()));}};
    timings(stages);out << "},\"mpc_timings_ns\":{";timings(totals);out << "}}\n";
    return 0;
  } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
