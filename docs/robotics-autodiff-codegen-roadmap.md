# Robotics Autodiff & CodeGen 专业能力路线图

## 1. 战略定位

Agentic-Robo-SWE 不应收窄为“CppADCodeGen Agent”。长期定位是：

> 面向机器人动力学、数值优化与仿真系统的领域感知软件工程 Agent，能够完成代码理解、
> 依赖配置、算法修改、数值验证和性能评测。

CppAD / CppADCodeGen 是第一个专业能力包和垂直化切入口。项目继续保留通用 Agent Harness，
领域能力通过可插拔 Domain Pack 接入，后续可扩展 Pinocchio、Crocoddyl、Eigen、CasADi、
acados、ROS 2、IsaacLab、ProxSuite/Aligator 以及 CUDA/Torch/JAX 机器人计算栈。

专业化的主要价值不是让模型记住更多 API，而是让 Agent 能够通过领域规则回答：

- 导数是否数值正确；
- 生成代码是否与原始模型等价；
- 输入、输出、Jacobian 和 Hessian 的维度及排列是否一致；
- 修复是否在随机输入、极端姿态和隐藏测试中稳定；
- CodeGen 是否真正提高模块或完整求解链路的吞吐；
- 当前模块应使用解析导数、自动微分、代码生成还是有限差分。

## 2. 为什么需要领域能力包

通用 Coding Agent 通常能处理依赖安装、CMake 修改和简单示例，但在以下问题上容易出现
“可以编译、实际错误”的假成功：

- `double`、`AD<double>`、`CG<double>` 与 Eigen Scalar 混用；
- tape 录制了错误函数，或录制阶段执行了错误的数据依赖分支；
- Jacobian 输入顺序、列排列、稀疏结构或维度错误；
- 原始函数与生成函数输出不等价；
- 导数正确，但 CodeGen 比解析导数更慢；
- 单函数变快，但 Crocoddyl/MPC 完整链路没有获得收益。

因此，领域能力建设的优先级是：

1. 机器人算法专用 Validator 和可复现 Benchmark；
2. CppAD/CppADCodeGen/Pinocchio/Crocoddyl 工具接口；
3. 可固定依赖版本的 C++ 构建与执行环境；
4. C++ 模板、类型和符号级代码检索；
5. 专业文档 RAG、工作流 Prompt 与代码模板。

只增加文档 RAG 或专用 Prompt 不足以构成项目壁垒。领域文档用于帮助定位问题，最终结果必须
由编译、数值一致性、导数正确性和性能回归共同判定。

## 3. 目标架构

```text
Agentic-Robo-SWE Core
├── Agent Loop / HITL / Checkpointer
├── 混合检索、AST 与符号索引
├── Patch 编辑、Sandbox、Timeout、Cache、Tracing
├── 通用 Benchmark Harness
└── Validator Gate
        │
        └── Robotics Optimization Domain Pack
            ├── 领域工作流与按需知识
            ├── CMake / 依赖 / ABI 诊断
            ├── C++ AD 兼容性检查
            ├── CppAD / CppADCodeGen 生成工具
            ├── 导数数值验证
            ├── Pinocchio / Crocoddyl 集成验证
            └── 微基准与端到端性能 Benchmark
```

领域逻辑不得硬编码进 Agent Loop。Core 只定义 Domain Pack 的注册、工具、环境、Validator 和
Benchmark 契约，使不同领域包能够独立安装、测试和演进。

建议第一个能力包命名为 `robotics_autodiff_codegen`。Agent 修改的是源模型、类型模板和 CodeGen
配置，不直接修改生成出的 C/C++ 文件；生成产物应随源模型重新生成并验证。

## 4. 分阶段实施计划

### Phase 6.0：Domain Pack 基础契约

状态：✅ 已完成基础实现。当前内置 `cpp_reference` Pack 已覆盖 manifest、动态工具与 Prompt、
命名空间 Validator、Benchmark schema v2、工具链指纹、受限 CMake 工件目录和独立 CI Job。
真实 CppAD/CppADCodeGen 依赖仍按计划留在 Phase 6.1。

目标：证明专有能力可以接入而不污染通用 Harness。

- 定义 Domain Pack manifest、工具注册、Validator 注册和可选依赖契约；
- 增加固定版本的 C++/CMake 本地执行环境，并记录编译器、ABI 与依赖指纹；
- 为生成代码、构建目录和动态库建立独立缓存与失效规则；
- 将构建超时、运行超时、输出截断和工件落盘接入现有 Benchmark 报告；
- 首期保持本地后端，E2B 只有在自定义镜像能够固定工具链后再纳入对比。

完成标准：一个最小示例包能够被发现、按需加载、构建、运行和独立禁用；未安装 C++ 专用依赖
时，现有 Python/MuJoCo 测试不受影响。

### Phase 6.1：`robotics_autodiff_codegen` MVP

目标：完成第一个“源代码修复 → 生成 → 编译 → 验证”闭环。

新增 AD 兼容性检查工具，重点发现：

- 写死的 `double` 和不正确的 Scalar 模板；
- Eigen 类型没有保留模板 Scalar；
- AD 不支持的函数、不安全转换和黑盒调用；
- tape 录制中的数据依赖分支与动态维度；
- 输入、输出和变量顺序契约不一致。

首个 Demo 使用一个只能接受 `double` 的小型机器人动力学函数。Agent 应将其修复为 Scalar 泛型，
建立 CppAD tape，生成 Jacobian，并通过有限差分对照。

完成标准：至少 3 个带故障夹具稳定失败；最小正确修复通过编译、函数输出和 Jacobian 验证；
Agent 不允许通过修改 Validator 或生成文件绕过检查。

### Phase 6.2：导数与生成代码一致性 Validator

目标：让“导数正确”成为独立于 Agent 自述的官方门禁。

对多组固定随机输入和边界状态自动比较：

```text
原始函数输出  ↔  CppAD 输出  ↔  CodeGen 动态库输出
有限差分 Jacobian  ↔  AD Jacobian  ↔  CodeGen Jacobian
```

Jacobian 相对误差采用：

```text
e_J = ||J_codegen - J_FD||_F / max(1, ||J_FD||_F)
```

Validator 同时检查：

- 输出等价性、输入输出维度与 Jacobian 排列；
- 多 seed 随机输入和极端姿态；
- NaN/Inf、不可导点和条件分支行为；
- dense/sparse Jacobian，随后扩展 Hessian；
- 失败 seed、最大误差样本、编译日志和动态库指纹落盘。

完成标准：数值错误但可编译的实现必须被拒绝；同一 seed、工具链和源码指纹能够逐值复现结果。

### Phase 6.3：Pinocchio 集成与性能决策

目标：验证 CodeGen 在真实机器人动力学函数上的正确性和收益。

- 读取固定 URDF，记录 `nq`、`nv`、输入布局和模型指纹；
- 包装 ABA、RNEA 或质心动力学中的一个最小目标函数；
- 生成机器人专用代码并编译动态库；
- 对比 Pinocchio 原始输出、解析导数、CppAD、CppADCodeGen 和有限差分；
- 使用 warm-up、重复采样和 P50/P95 延迟，避免只报告单次计时；
- 把性能回归作为 Validator，CodeGen 无收益时允许输出“不建议采用”的正确结论。

完成标准：数值一致性必须先通过；性能结论包含测量环境、样本数、方差和端到端影响，不能只凭
微基准中的单个最优值做决策。

### Phase 6.4：Crocoddyl/MPC 链路诊断

目标：从单函数优化升级为完整求解链路的工程判断。

- profile `calc()`、`calcDiff()`、FDDP backward pass 和 line search；
- 定位数值差分 residual 或导数计算是否真的是主要瓶颈；
- 替换一个受控模块并执行正确性回归；
- 比较单次 solve、多次 MPC rollout 和多环境训练吞吐；
- 若瓶颈实际来自矩阵分解、串行 FDDP 或 CPU–GPU 调度，明确拒绝无收益的 CodeGen 改造。

完成标准：Agent 的建议由端到端 profile 支撑，并能区分局部加速与系统吞吐提升。

### M2：Robo-SWE 专业 Benchmark 与消融实验

建立 20–30 个小型、可复现任务，覆盖：

| 类别 | 代表任务 |
|------|----------|
| Build | CMake、依赖版本、链接、ABI 和 Python binding |
| Localization | Scalar 写死、变量顺序、维度和模板错误 |
| Correctness | Jacobian/Hessian、条件分支、数值稳定性和生成代码等价性 |
| Performance | tape 大小、稀疏导数、解析/AD/CodeGen 对比和端到端回归 |

对比以下四组配置：

1. 通用 Harness；
2. 通用 Harness + 领域文档 RAG；
3. 上述能力 + 专业工具；
4. 上述能力 + 专业 Validator 与固定构建环境。

核心指标包括任务成功率、首次编译通过率、导数验证通过率、隐藏测试通过率、工具调用数、Token、
完成时间、性能回归率，以及“错误但声称成功”的 false-positive rate。只有第 4 组相对第 1 组
产生稳定、可复现的提升，才能证明专业能力包设计有效。

## 5. 优先展示的三个 Demo

1. **AD 类型兼容自动修复**：定位 `double` 写死与 Eigen Scalar 问题，完成 tape、Jacobian 和
   有限差分验证。
2. **Pinocchio 动力学生成专用代码**：包装固定机器人模型的动力学函数，生成动态库，比较输出、
   导数和解析实现性能。
3. **Crocoddyl `calcDiff()` 瓶颈诊断**：profile 完整链路，仅在数值正确且端到端吞吐改善时保留
   CodeGen 方案。

## 6. 实施原则与风险控制

- CppADCodeGen 是切入口，不是项目全部；Core 始终保持领域无关。
- 先建立构建环境、专业工具和 Validator，再扩充文档 RAG。
- 编译成功只是中间状态，不能作为任务成功标准。
- 正确性门禁优先于性能；性能提升不得掩盖数值、安全或稳定性回归。
- 所有随机验证必须使用固定 seed，并保存失败输入与工具链指纹。
- 不把生成代码提交给 Agent 直接编辑，避免修复无法从源模型复现。
- C++ 工具链依赖较重，应作为 optional extra 或独立镜像，不拖慢默认 CI。
- 先从无外部机器人模型的小任务开始，再进入 Pinocchio/Crocoddyl 集成，控制调试面和构建成本。

## 7. 与当前路线的衔接

Phase 5.3 先完成 MuJoCo 失败轨迹时序落盘与诊断可视化，为后续领域 Validator 的报告格式建立
基础。随后进入 Phase 6.0 Domain Pack 契约，再按 6.1–6.4 推进 CppADCodeGen、Pinocchio 和
Crocoddyl。M2 Benchmark 与 Phase 6 同步积累题目，在专业工具和 Validator 稳定后执行正式消融。
