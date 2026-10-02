Use this pack to diagnose a pinned Crocoddyl LQR/FDDP solver chain before proposing CodeGen adoption.
The fixed comparison is analytic ActionModelLQR versus ActionModelNumDiff first derivatives on the
same problem. ActionModelLQR exposes no residual, so its NumDiff wrapper has no valid Gauss-Newton
Hessian and is intentionally not used as an FDDP solver replacement.
Correctness is checked before performance and raw observations are independently judged in Python.
Report calc, calcDiff, backward pass, try-step, solve, and rollout costs separately.
Do not call the Phase 6.4a result a CppADCodeGen or Pinocchio-RNEA end-to-end speedup: no generated
action model is substituted in this phase. A derivative-dominated result is only a candidate signal
for Phase 6.4b. Missing pinned native dependencies are capability failures; never auto-install them.
