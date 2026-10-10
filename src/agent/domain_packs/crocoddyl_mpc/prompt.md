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
The controlled trial substitutes ONLY the running-action ABA acceleration Jacobian inside calcDiff.
Compare native analytic, CppAD and CodeGen with identical shifted warm start. calc values, costs,
terminal, integrator, solver and independent plant remain native. Gate 25 action/terminal samples
and all repetitions before performance. Report paired end-to-end speedups and generation/compile/load
amortization. A keep-native or noisy-profile result is valid; never promote a single-process trial
to production adoption, hard-real-time safety, or arbitrary robot/contact support.
Use the process trial for 3-20 sequential fresh native processes. Reuse only the compiled harness;
each process rebuilds its own generated library, tape and measurement state. Require unique native
PIDs and consistent dependency/toolchain/executable identities. Judge every raw trace again and
require benefit in EVERY process, not a pooled median. Stable results are only opt-in pilot
candidates on the measured machine/workload; the default native backend remains unchanged.
