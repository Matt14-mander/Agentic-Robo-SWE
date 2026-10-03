#pragma once
#include <memory>
#include <array>
#include <vector>
#include <string>
#include <pinocchio/multibody/model.hpp>

// Keep CppADCodeGen templates out of the Crocoddyl translation unit.
class AbaBackend {
 public:
  AbaBackend(const pinocchio::Model& model, const std::string& library_base);
  ~AbaBackend();
  std::vector<double> forward(const std::vector<double>& input, bool generated);
  std::vector<double> jacobian(const std::vector<double>& input, bool generated);
  double compile_seconds() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
