# Phase 6.4c：受控 CodeGen 动作模块替换试验

目标是验证已有真实机器人 MPC 中的导数瓶颈能否转化为端到端收益，而不是把 warm-start
带来的收益算到 CodeGen 上。默认原生后端不变，本模块只是显式选择的试验。

## 唯一替换点

运行节点 `DifferentialActionModelFreeFwdDynamics.calcDiff()` 内的 ABA 加速度 Jacobian。
输入顺序固定为 `[q1, q2, v1, v2, tau1, tau2]`，输出为 2×6 row-major。
三路为 Pinocchio 原生解析导数、CppAD tape Jacobian、CppADCodeGen 生成库 Jacobian。

ABA 加速度求值、全驱动扭矩映射、状态/控制代价、终端代价、Euler 积分、FDDP、移位 warm-start
及独立 ABA plant 完全共用原来的配置。这里生成的是 **ABA 导数**，不是把之前的 RNEA
函数接到错误的 forward-dynamics 接口，也不是替换整套 solver。

仅支持固定的双转动关节、nq=nv=nu=2、无 armature、接触、约束和浮动基。不要泛化为任意机器人。

## 正确性与性能门禁

- 24 组固定非零输入和 1 组重力平衡输入；输入、形状、顺序、有限性均严格验证。
- 独立两连杆方程与中心差分校验三路 ABA 值和 Jacobian；独立校验积分后 xnext、Fx、Fu。
- 对照运行和终端动作的 cost、Lx/Lu、Lxx/Lxu/Luu，防止替换导致代价/Hessian 变化。
- 三路各 2 次预热、10 次重复、每次 32 步闭环；全轨迹、控制、代价、收敛、shooting
  一致性及独立 plant 转移均必须通过，不能只看最终姿态。
- 每轮轮换三路的执行顺序；所有路线均使用相同 warm-start，不与 cold baseline 混比。
- 报告 SolverFDDP.calcDiff、逐步 solve、完整 32 步验证 workload 的 P50/P95/方差与配对收益。

若 native/codegen 闭环 CV > 0.3，则 `inconclusive_noisy_profile`。
否则仅当 P50 至少 1.05 倍、至少 8/10 配对重复达到 1.05 倍、P95 不回退，才输出
`candidate_for_repeated_process_trial`；其他情况输出 `keep_native_no_measured_end_to_end_benefit`。
正确性通过但没有收益时 CLI 仍成功退出，这是有效的试验结论，不是验收失败。

编译摊销分母为完整 32 步 workload 的 P50 节省时间；分子为 CG tape、源码生成、gcc 编译和
动态加载的成本。不包含 harness 首次构建或 CppAD 对照 tape 的建立。无正节省时摊销为 null。
这些计数不能直接解释为真实机器人控制周期；本试验不是硬实时或生产安全认证。

## WSL/Linux 运行

复用固定依赖，不需要重新安装已有可用环境：

```bash
uv run python scripts/run_crocoddyl_mpc.py --codegen
uv run pytest -q tests/test_crocoddyl_codegen_trial.py -m crocoddyl
```

`--robot` 保留 6.4b 的 cold/warm baseline；不能同时传 `--robot --codegen`。
Windows 只跑离线门禁，原生试验要求固定版本的 Linux 依赖。原生 harness 使用单个编译作业，
配置限时 300s、构建 900s、运行 120s。首次模板编译可能持续数分钟无细粒度输出；入口会
提示当前阶段。请勿同时开启 pytest 和 CLI 的首次构建。后续相同源码/配置命中构建缓存。

每次运行生成唯一目录，包含 contract/raw/correctness/profile/validation、首个失败证据、
生成 `.so` 与源码目录。validation 记录库 SHA-256、依赖指纹、工具链和测量环境。
生成库不会覆盖上一次试验；临时对象目录也按运行隔离。

## 本机原生验收

最终源码保存版本已在 Ubuntu 22.04 / WSL2、GCC 11.4、Release、单编译作业下完成
真实 CMake harness 运行及独立 Python 门禁；CLI 编排另有离线测试。原始证据保存在
`.agent_state/domain_artifacts/crocoddyl_mpc/phase64c-final-native/`（本机忽略目录，非提交资产）。

| 指标 | 本次单进程观测 |
|---|---|
| 25 组动作/代价/导数、全部闭环门禁 | 通过 |
| 三路闭环状态最大差异 | 1.55e-15 |
| ABA Jacobian 对独立中心差分最大误差 | 3.49e-9 |
| 32 步 MPC P50：native / CppAD / CodeGen | 11.398 / 11.500 / 8.811 ms |
| CodeGen 端到端 P50 加速 / 配对达标次数 | 1.294 倍 / 10/10 |
| native / CodeGen 闭环 CV | 4.19% / 4.95% |
| CG 生成、源码落盘、编译和加载成本 | 1.632 s |
| 估计摊销 | 631 次 32 步 workload（20192 个验证控制步） |

calcDiff 的 P50 为 127.70 / 118.56 / 70.68 μs，但 CodeGen 的该微阶段 CV 为 34.37%，
局部加速比只能作为带噪声观测。采用候选判断由低噪声的配对端到端闭环决定，不由此微基准决定。
最终建议为 `candidate_for_repeated_process_trial`，不是生产采用结论。

## 后续边界

单进程收益只是进入多独立进程、不同负载和不同初始状态复测的候选证据。
先保持原生 fallback 和显式选择，再考虑更一般的动作模型及机器人任务；不自动切换 agent
或生产控制循环的默认后端，也不声称目前测到了训练吞吐收益。
