# Phase 6.3：Pinocchio RNEA 集成与性能决策

状态：首版实现；原生 Linux 编译、动力学数值与速度结果需在固定依赖环境验收。
Windows 上通过报告规则测试不代表 Pinocchio/CodeGen 实测通过。

## 实现范围

独立 `pinocchio_rnea` Domain Pack，不把 Pinocchio 特例写入 Agent Loop。
项目自带无网格依赖的双连杆 URDF，固定基座，`nq=nv=2`，重力 `[0,0,-9.81]`。
首版不接受任意 URDF 或浮动基座；`nq != nv` 时的流形扰动不在本次范围。

- 输入：`[q_shoulder,q_elbow,v_shoulder,v_elbow,a_shoulder,a_elbow]`；输出：两关节力矩。
- 完整 Jacobian：`2×6`，row-major，列顺序为 `[dq,dv,da]`。
- 24 组样本：零点、回归状态、边界状态及 16 组 seed=20260904 随机状态。
- 每个状态对比原生 RNEA、候选源函数、CppAD 和 CodeGen 输出。
- 导数比较 Pinocchio 解析导数、尺度自适应中心有限差分、CppAD、CodeGen。
- Pinocchio 的 `dτ/da = M` 只填上三角，harness 补全对称部分。
- Python 独立核对模型指纹、维度、样本覆盖、有限性及逐元素混合误差，不信任通过标志。

## 性能结论的边界

比较的工作量统一为“RNEA 输出 + 完整 Jacobian”，包含各封装的结果数组开销。
模型、data、输入预先创建；4 种后端各预热 32 次，再轮换执行顺序，采集 50 个批次，
解析基线复用 `computeRNEADerivatives` 已计算的力矩，不额外重复执行 RNEA。
每批 16 次。保留每批平均 ns/call；P50/P95 是批均值的分位数，不是逐次调用尾延迟。
报告均值、标准差、变异系数、CodeGen 编译秒数、系统和工具链信息。

采用建议首先受正确性约束：

| 条件 | 建议 |
| --- | --- |
| 数值错误 | `blocked_by_correctness`，不评估加速收益 |
| 缺少完整性能数据 | `insufficient_evidence` |
| Pinocchio 或 CodeGen 变异系数 > 25% | `inconclusive_noisy_measurements` |
| 相对解析 Pinocchio 中位数至少快 10%，且 P95 不退步 | `candidate_for_end_to_end_trial` |
| 其他情况 | `keep_pinocchio_analytic` |

`pinocchio_rnea.correctness` 接受“代码正确，但不建议使用 CodeGen”的结果。
可选严格门禁 `pinocchio_rnea.codegen_benefit` 要求正确性和上述性能收益同时成立，
用于明确承诺加速的任务，不应被当作所有机器必须通过的 CI 门禁。
端到端速度字段始终为 null；当前没有 MPC solve/rollout 测量，不能推导系统吞吐提升。

## Linux / WSL 执行

不要复用 Windows 创建的 `.venv` 或 CMake build/install 目录；使用独立 Linux checkout。
系统准备（显式安装）：

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++ git pkg-config libeigen3-dev liburdfdom-dev \
  libboost-filesystem-dev libboost-serialization-dev libboost-system-dev
uv sync --frozen --extra dev
uv run python scripts/bootstrap_autodiff_codegen.py
uv run python scripts/bootstrap_pinocchio.py
uv run python scripts/run_pinocchio_probe.py
uv run pytest -q tests/test_pinocchio_rnea.py -m pinocchio
```

Pinocchio 固定为 v3.4.0 / `187afafcfe22d7ac16a26241c0b13a76d04d82c1`，
复用已有 CppAD/CppADCodeGen 固定提交；只初始化构建系统子模块，不下载 example-robot-data。
系统 Eigen/Boost/URDFDOM 由 Linux 包管理器提供，尚未宣称整个 OS 镜像按哈希锁定。
普通导入、能力探测和 Validator 不会安装依赖；`bootstrap_pinocchio.py --check` 可离线运行。

无需 LLM 的正确源模型探针会打印产物路径：
`.agent_state/domain_artifacts/pinocchio_rnea/<run-id>/`，其中包含：

- `validation.json`：源码/URDF 指纹、规格、环境、依赖、构建与运行结果；
- `samples.json` / `samples.txt`：可复现输入；
- `raw.json`：原始输出、导数、计时数据；
- `correctness.json` / `failing-input.json`：逐项错误与首个失败输入；
- `performance.json`：统计和采用建议。

## SWE 任务

两题仍让 Agent 修复源代码，而不是生成 C 文件：

1. `rnea_acceleration_slice`：把速度切片误传为加速度。
2. `rnea_missing_velocity`：把速度硬置零，遗漏运动状态动力学项。

```bash
uv run python scripts/run_benchmark.py --manifest benchmarks/pinocchio_cases.json --validate-fixtures
uv run python scripts/run_benchmark.py --manifest benchmarks/pinocchio_cases.json \
  --max-loops 12 --task-timeout 1200
```

测试中的手工正确修复不能当作 Agent 成功率；后者需单独运行以上在线 Benchmark。
CI 增加独立 `pinocchio-rnea-checks`，不要求机器一定得到 CodeGen 加速结论。

## 后续验收与扩展

先完成 Linux 原生链路验收，保留真实数值误差和时间报告；再跑 Agent 两题及重复实验。
后续引入更高自由度模型、流形配置空间和端到端控制工作量，才评估进入 Phase 6.4。
本次不实现 ABA、多种动力学目标、任意 URDF、Hessian 或 Crocoddyl/MPC。

接口依据：固定版本的 [Pinocchio 源码](https://github.com/stack-of-tasks/pinocchio/tree/v3.4.0)，
其中 `include/pinocchio/algorithm/rnea-derivatives.hpp` 定义解析导数与上三角约定，
`unittest/cppad/algorithms.cpp` 和 `unittest/cppadcg/algorithms.cpp` 提供 AD Scalar 使用依据。
