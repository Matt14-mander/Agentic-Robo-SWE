#pragma once
#include "aba_codegen.hpp"
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

// Fixed fully actuated two-link model with no armature, contacts or constraints.
// Only the ABA derivative calculation changes; inherited calc/terminal methods,
// collectors, actuation mapping and CostModelSum remain identical to the baseline.
class AbaDerivativeAction : public crocoddyl::DifferentialActionModelFreeFwdDynamics {
 public:
  using Base=crocoddyl::DifferentialActionModelFreeFwdDynamics;
  using Base::calcDiff;
  AbaDerivativeAction(std::shared_ptr<crocoddyl::StateMultibody> state,
      std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation,
      std::shared_ptr<crocoddyl::CostModelSum> costs,
      std::shared_ptr<AbaBackend> backend,bool generated)
      :Base(state,actuation,costs),backend_(std::move(backend)),generated_(generated) {
    if(state->get_nq()!=2 || state->get_nv()!=2 || get_nu()!=2)
      throw std::runtime_error("ABA trial supports only fixed nq=nv=nu=2");
  }
  void calcDiff(const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& u) override {
    if(x.size()!=4 || u.size()!=2) throw std::runtime_error("wrong action input dimensions");
    auto* d=static_cast<crocoddyl::DifferentialActionDataFreeFwdDynamics*>(data.get());
    get_actuation()->calcDiff(d->multibody.actuation,x,u);
    std::vector<double> input{x[0],x[1],x[2],x[3],
        d->multibody.actuation->tau[0],d->multibody.actuation->tau[1]};
    const auto jac=backend_->jacobian(input,generated_);
    if(jac.size()!=12) throw std::runtime_error("wrong generated ABA Jacobian dimensions");
    for(int row=0;row<2;++row) {
      for(int col=0;col<4;++col) d->Fx(row,col)=jac[row*6+col];
      for(int col=0;col<2;++col) d->pinocchio.Minv(row,col)=jac[row*6+col+4];
    }
    d->Fx.noalias() += d->pinocchio.Minv*d->multibody.actuation->dtau_dx;
    d->Fu.noalias() = d->pinocchio.Minv*d->multibody.actuation->dtau_du;
    d->multibody.joint->da_dx=d->Fx;d->multibody.joint->da_du=d->Fu;
    get_costs()->calcDiff(d->costs,x,u);
  }
 private:
  std::shared_ptr<AbaBackend> backend_;
  bool generated_;
};
