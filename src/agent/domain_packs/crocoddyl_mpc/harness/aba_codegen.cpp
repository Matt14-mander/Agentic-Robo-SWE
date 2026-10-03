#include <pinocchio/codegen/cppadcg.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include "aba_codegen.hpp"
#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace {
template<class Base>
CppAD::ADFun<Base> record_aba(const pinocchio::Model& original) {
  if(original.nq!=2 || original.nv!=2 || original.njoints!=3)
    throw std::runtime_error("ABA trial requires the fixed two-joint robot");
  using AD = CppAD::AD<Base>;
  auto model = original.cast<AD>();
  pinocchio::DataTpl<AD> data(model);
  std::vector<AD> independent(6);
  for(int i=0;i<6;++i) independent[i]=0.1*(i+1);
  CppAD::Independent(independent);
  Eigen::Matrix<AD,Eigen::Dynamic,1> q(2),v(2),tau(2);
  for(int i=0;i<2;++i) {q[i]=independent[i];v[i]=independent[i+2];tau[i]=independent[i+4];}
  const auto acceleration=pinocchio::aba(model,data,q,v,tau);
  std::vector<AD> dependent{acceleration[0],acceleration[1]};
  CppAD::ADFun<Base> tape(independent,dependent);
  tape.optimize();
  return tape;
}
}

struct AbaBackend::Impl {
  CppAD::ADFun<double> ad;
  std::unique_ptr<CppAD::cg::DynamicLib<double>> library;
  std::unique_ptr<CppAD::cg::GenericModel<double>> generated;
  double compile_seconds=0;
  Impl(const pinocchio::Model& model,const std::string& base):ad(record_aba<double>(model)) {
    auto start=std::chrono::steady_clock::now();
    auto tape=record_aba<CppAD::cg::CG<double>>(model);
    CppAD::cg::ModelCSourceGen<double> source(tape,"aba_derivatives");
    source.setCreateForwardZero(true);source.setCreateJacobian(true);
    CppAD::cg::ModelLibraryCSourceGen<double> library_source(source);
    CppAD::cg::DynamicModelLibraryProcessor<double> processor(library_source,base);
    CppAD::cg::GccCompiler<double> compiler;
    compiler.setCompileFlags({"-O3","-DNDEBUG"});
    compiler.setTemporaryFolder(base+"_objects");
    compiler.setSourcesFolder(base+"_sources");
    compiler.setSaveToDiskFirst(true);
    std::filesystem::create_directories(base+"_sources");
    library=processor.createDynamicLibrary(compiler);
    generated=library->model("aba_derivatives");
    if(!generated) throw std::runtime_error("missing generated ABA model");
    compile_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  }
};
AbaBackend::AbaBackend(const pinocchio::Model& model,const std::string& base)
    :impl_(std::make_unique<Impl>(model,base)) {}
AbaBackend::~AbaBackend()=default;
std::vector<double> AbaBackend::forward(const std::vector<double>& x,bool generated) {
  if(x.size()!=6) throw std::runtime_error("ABA input must contain six scalars");
  return generated?impl_->generated->ForwardZero(x):impl_->ad.Forward(0,x);
}
std::vector<double> AbaBackend::jacobian(const std::vector<double>& x,bool generated) {
  if(x.size()!=6) throw std::runtime_error("ABA input must contain six scalars");
  return generated?impl_->generated->Jacobian(x):impl_->ad.Jacobian(x);
}
double AbaBackend::compile_seconds() const {return impl_->compile_seconds;}
