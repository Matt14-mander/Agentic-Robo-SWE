Use this pack to diagnose a pinned Crocoddyl LQR/FDDP solver chain before proposing CodeGen adoption.
The fixed comparison is analytic ActionModelLQR versus ActionModelNumDiff first derivatives on the
same problem. ActionModelLQR exposes no residual, so its NumDiff wrapper has no valid Gauss-Newton
Hessian and is intentionally not used as an FDDP solver replacement.
Correctness is checked before performance and raw observations are independently judged in Python.
Report calc, calcDiff, backward pass, try-step, solve, and rollout costs separately.
Do not call the Phase 6.4a result a CppADCodeGen or Pinocchio-RNEA end-to-end speedup: no generated
action model is substituted in this phase. A derivative-dominated result is only a candidate signal
for Phase 6.4b. Missing pinned native dependencies are capability failures; never auto-install them.
Phase 6.4b runs native Pinocchio two-link forward dynamics with torque controls, symplectic Euler,
and 32 receding-horizon MPC steps. Compare cold and shifted warm start with every repetition's
states/controls, convergence, shooting feasibility, and plant-equation checks. Warm-start speedup
is a solver initialization result and must not be attributed to CodeGen. Use the robot profile to
decide whether a generated dynamics action model is worth a subsequent controlled trial.
