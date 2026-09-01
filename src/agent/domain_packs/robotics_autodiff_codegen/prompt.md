Robotics Autodiff CodeGen Pack:
- Inspect the target with inspect_ad_compatibility before changing it.
- Preserve the model function name, input/output dimensions, and documented variable order.
- Make source models Scalar-generic; do not cast AD values to double or introduce value-dependent tape branches.
- Modify only the supplied source model. Never edit the validator, harness, generated C/C++, build cache, or dynamic library.
- Use inspect_autodiff_codegen_environment to distinguish source defects from missing pinned dependencies.
- Run the exact official validator after the smallest source repair.
- Compilation is intermediate evidence. Completion requires original, CppAD, and CodeGen outputs plus the finite-difference Jacobian gate.
