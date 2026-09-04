// Correct reference wrapper for standalone smoke runs; agents repair a workspace copy.
#pragma once
#include <pinocchio/algorithm/rnea.hpp>
template<class Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> candidate_rnea(
    const pinocchio::ModelTpl<Scalar>& model, pinocchio::DataTpl<Scalar>& data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> q = x.segment(0, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v = x.segment(2, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> a = x.segment(4, 2);
  return pinocchio::rnea(model, data, q, v, a);
}
