You are repairing a Pinocchio RNEA source wrapper, not generated C/C++ or the validator.
The official robot is a fixed-base two-link URDF with nq=nv=2, gravity [0,0,-9.81].
Input layout is [q_shoulder,q_elbow,v_shoulder,v_elbow,a_shoulder,a_elbow], in SI units.
Preserve the candidate_rnea<Scalar> interface for double, AD<double>, and AD<CG<double>>.
Do not hardcode sampled outputs, edit the URDF, disable gravity, or change dependency installations.
Call the exact official validator from the benchmark task for final acceptance.
Numerical success requires source/CppAD/CodeGen outputs and Jacobians to agree with native
Pinocchio and its analytical derivatives at all official samples, including finite differences.
Compile success or self-reported success is not sufficient. Inspect failure artifacts.
Performance is output + full Jacobian, measured after warm-up with batched P50/P95 samples.
"keep_pinocchio_analytic" is a valid conclusion for a correct implementation with no CodeGen benefit.
Never claim MPC or end-to-end speedup from these microbenchmarks; that measurement is not available.
Missing native dependencies are capability errors, not successful validation; do not auto-install them.
