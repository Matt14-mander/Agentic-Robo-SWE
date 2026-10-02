# Phase 6.4：Crocoddyl/MPC 链路诊断

状态：Phase 6.4a 代码已实现，Windows 离线门禁已接入；固定 Linux/WSL 原生构建和测量待验收。

## 6.4a 完成的能力

新增 `crocoddyl_mpc` Domain Pack，以固定 Crocoddyl 3.0.1 和 Pinocchio 3.4.0 构建一个
4 状态、2 控制量、40 节点的确定性 LQR/FDDP 问题。原生 harness 输出以下证据：

- 解析 LQR 与 `ActionModelNumDiff` 的 `Fx`、`Fu`、`Lx`、`Lu` 一致性；
- FDDP 成本下降、末状态收敛和优化轨迹重新 rollout 一致性；
- `calc()`、`calcDiff()`、backward pass、`tryStep()`、完整 solve 和 rollout 的独立 P50/P95；
- 数值微分相对解析 `calcDiff()` 的耗时倍数；
- `calcDiff()` 在 direction kernel 中的占比与保守的下一步建议。

Python 侧重新计算所有判定，不信任 C++ 自报的 pass。原始输出、正确性判定和 profile 会写入
`.agent_state/domain_artifacts/crocoddyl_mpc/<run-id>/`。

## 明确没有完成的能力

6.4a 没有把 CppADCodeGen 生成的 RNEA 或动作模型接入 Crocoddyl，也没有测量真实机器人 MPC
rollout。因此 `codegen_end_to_end_speedup` 必须为 null。`ActionModelLQR` 没有 residual，包装后的
`ActionModelNumDiff` 不具备可用于 FDDP 的 Gauss-Newton Hessian，本阶段只将它用于一阶导数正确性
和 `calcDiff()` 成本对照。

## Linux/WSL 验收

先复用 Phase 6.3 的固定依赖，再安装 Crocoddyl：

```bash
uv run python scripts/bootstrap_autodiff_codegen.py
uv run python scripts/bootstrap_pinocchio.py
uv run python scripts/bootstrap_crocoddyl.py
uv run pytest -q tests/test_crocoddyl_mpc.py -m crocoddyl
uv run python scripts/run_crocoddyl_mpc.py
```

若依赖缺失，结果必须停在 `capability`；若构建或运行失败，诊断文件必须保留。机器噪声导致
最大 CV 超过 0.30 时只给出 `inconclusive_noisy_profile`。

## 进入 6.4b 的条件

- 正确性门禁通过；
- profile 不是高噪声；
- `calcDiff()` 占 direction kernel 至少 50%，得到
  `candidate_for_codegen_action_model_trial`。

满足条件后，6.4b 才选择一个 residual/dynamics 模块完成解析/AD/CodeGen 三路替换，并比较单次
solve、多步 receding-horizon rollout、编译摊销和端到端吞吐。若 backward pass 或 line search
才是主要成本，则应优先优化求解器设置或问题结构，而不是继续扩大 CodeGen 改造。
