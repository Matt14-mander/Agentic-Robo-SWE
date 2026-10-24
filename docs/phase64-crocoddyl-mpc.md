# Phase 6.4：Crocoddyl/MPC 链路诊断

收尾状态（2026-10-24）：6.4a/b/c 和跨进程试验框架已实现；新增可配置采样与逐项性能拒绝诊断，
不改变默认 Native。最新用户复测数值正确性 3/3、性能门禁 2/3，因此保持原生；新增扩样路径
仍待 Linux 原生复验。命令、历史证据与结论边界见 [6.4c 收尾说明](phase64c-controlled-codegen.md)。

状态：Phase 6.4a 已在 WSL 原生验收。实测解析 calcDiff 占 direction kernel 25.6%，backward
pass 占 74.4%；该小型 LQR 建议优先检查求解器。Phase 6.4b 已接入真实两关节 MPC，
并于 2026-10-02 在当前 WSL Ubuntu-22.04/GCC 11.4 原生编译运行，通过独立 Python 门禁。

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

## 6.4b：真实 Pinocchio MPC 与 warm-start

6.4a 的 LQR 比例不能直接推断机器人动作模型的瓶颈，因此先建立真实动力学 baseline。
使用 Phase 6.3 的固定 URDF、重力和 SI 单位；动作模型为 StateMultibody、ActuationModelFull、
DifferentialActionModelFreeFwdDynamics 和 IntegratedActionModelEuler。控制量为关节力矩。
预测 horizon 为 40，dt=0.02s，闭环执行 32 步。独立 Pinocchio ABA plant 接收每次求解的首个
控制。冷启动使用参考姿态的重力补偿；warm-start 移位上一时刻控制序列，并从实际状态重新
rollout 得到可行初始状态序列。两种策略共享成本、积分、收敛阈值与迭代预算。

预热 2 次、测量 10 次，交替策略顺序；每次重复的状态、力矩、迭代数、cost、shooting rollout
残差与每步 solve 延迟均落盘。Python 使用固定 URDF 对应的独立两连杆刚体方程检查全部状态
转换，并检查末状态、限位、控制量、求解收敛和代价下降。cold 与 warm 的完整轨迹也进行对照。
stage profile 仍区分 calc、calcDiff、backward、tryStep、solve、rollout。

完整 MPC 计时包含问题/solver 创建、初始化 rollout、solve、回放检查、plant 推进与 warm-start
移位；它是带验证开销的固定仿真链路延迟。每步 solve 延迟另外保存，不将完整计时解释为真实
机器人实时控制延迟。warm-start 的收益单独报告，CodeGen 端到端加速字段继续为 null。

现有依赖均已安装时无需重建，只运行：

```bash
uv run python scripts/bootstrap_crocoddyl.py --check
uv run pytest -q tests/test_crocoddyl_robot_mpc.py -m crocoddyl
uv run python scripts/run_crocoddyl_mpc.py --robot
```

首次原生 harness 编译允许 900s，入口打印构建/运行阶段进度。安装系统构建依赖时应包含
`libboost-test-dev`；Crocoddyl 3.0.1 即使关闭 BUILD_TESTING 也要求 Boost unit_test_framework。

## 后续 CodeGen 动作模型替换的条件

受控替换现已作为 6.4c 单独实现，使用 `--codegen`；设计、门禁、计时范围与限制见
[受控 CodeGen 试验说明](phase64c-controlled-codegen.md)。本节保留触发该试验的 6.4b 基线证据。

本机首次原生测量（单进程，10 次重复，不代表跨机器收益）：

| 指标 | 结果 |
|---|---|
| 独立 plant 最大误差 | 1.33e-15 |
| cold/warm 完整状态轨迹最大差异 | 2.29e-10 |
| calcDiff P50 / backward P50 | 52.09 / 15.26 μs |
| calcDiff 在 direction kernel 中占比 | 77.34% |
| cold / warm 32 步 MPC P50 | 11.42 / 8.54 ms |
| warm-start P50 收益 | 1.34 倍 |
| 平均 FDDP 迭代数 cold / warm | 3 / 2.03125 |

真实动作模型建议为 `candidate_for_codegen_action_model_trial`；warm-start 建议为
`use_shifted_warm_start`。calcDiff 采用 SolverFDDP.calcDiff 的计时边界，包括 calc 和可行性
处理。这个结果支持下一步受控导数模块替换试验，尚未证明生成动作模型的加速收益。

- 正确性门禁通过；
- profile 不是高噪声；
- `calcDiff()` 占 direction kernel 至少 50%，得到
  `candidate_for_codegen_action_model_trial`。

满足真实机器人 profile 条件后，再选择一个 residual/dynamics 模块完成解析/AD/CodeGen 三路替换，并比较单次
solve、多步 receding-horizon rollout、编译摊销和端到端吞吐。若 backward pass 或 line search
才是主要成本，则应优先优化求解器设置或问题结构，而不是继续扩大 CodeGen 改造。
