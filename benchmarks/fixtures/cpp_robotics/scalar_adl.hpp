#pragma once
#include <cmath>
// Contract: r(q,v)=sin(q)+q*v for any Scalar, including AD types.
template<class Scalar> Scalar residual(Scalar q, Scalar v) {
  return std::sin(static_cast<double>(q)) + q*v;
}
