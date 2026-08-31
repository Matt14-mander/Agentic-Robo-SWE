# Agentic-Robo-SWE

An autonomous, simulation-in-the-loop software engineering agent for robotics. Built with LangGraph to debug and verify embodied AI algorithms.

设计哲学：把 **LangGraph 显式状态机**（可视、可中断、可恢复）与 **Princeton SWE-agent 的 Agent-Computer Interface** 思想（为 LLM 量身设计的工具集，而非裸 shell）结合起来，做一个能自主排障机器人物理算法仓库的 Agent。

---

## Phase 2：ReAct 循环（当前阶段）

```
                  ┌──────────────────────┐
START → planner → │ should_continue?     │
   ↑              │  - has tool_calls → "tools"
   │              │  - else → END
   │              │  - loop_step >= max → END (熔断)
   │              └──────────────────────┘
   │                       │ "tools"
   └─────── tools (ToolNode) ◀───────────┘
```

`planner` 是绑定了 5 个工具的 LLM 节点，每次输出可能含 `tool_calls` 的 `AIMessage`；
`ToolNode` 自动派发到对应工具、把返回写回 `ToolMessage`；
条件边 `should_continue` 根据"是否还有 tool_calls"和"loop_step 是否超额"决定继续还是结束。

### 工具集

| 工具 | 作用 |
|------|------|
| `list_dir(path, max_depth)` | 浏览目录结构 |
| `read_file_chunk(path, offset, limit)` | 分页读文件，返回带行号 |
| `grep_codebase(pattern, glob, max_results)` | 正则搜索仓库 |
| `execute_python(code, timeout)` | subprocess 沙盒中跑 Python，复现 bug / 验证修复 |
| `write_patch(path, old_string, new_string)` | 精确字符串替换（要求唯一匹配） |

> ✅ **Phase 2.5 已支持双后端**：`execute_python` 可通过 `EXECUTOR_BACKEND` env 切换 `local`（subprocess）或 `e2b`（E2B Code Interpreter 云沙盒，Firecracker microVM 真隔离）。

### 测试 fixture 设计

- `tests/fixtures/buggy_ik_template.py` —— 含 4 个 bug 的 2-link IK 函数，**永不修改**（题面源头）。
- `tests/fixtures/_buggy_ik_workspace.py` —— 每次 demo / E2E 测试启动时从 template 复制出的副本，agent 在这上面动手。

### 模型层

支持 **Claude 3.5 Sonnet / GPT-4o-mini / DeepSeek-Chat**，通过 `MODEL_PROVIDER` 环境变量切换，零代码改动。

### 执行后端（Phase 2.5）

`execute_python` 工具支持两种 sandbox，由 `EXECUTOR_BACKEND` 切换：

| 维度 | `local` | `e2b` |
|------|---------|-------|
| 隔离强度 | cwd + tempdir，**LLM 仍可读本机其它文件** | Firecracker microVM，真隔离 |
| 启动开销 | ~0.5s | ~3-5s（云端冷启动） |
| 需要凭据 | 无 | `E2B_API_KEY` |
| 项目源码可用性 | PYTHONPATH 注入 | 自动上传 src/+tests/+benchmarks/ 文件 |
| 第三方依赖 | 复用当前 venv | 需 E2B 镜像预装或由执行代码安装 |
| 适合场景 | 学习/快速迭代 | 生产/不可信代码 |

#### 启用 E2B

1. 安装 sandbox 依赖：`uv sync --extra dev --extra sandbox`
2. 去 [e2b.dev](https://e2b.dev) 注册账号、申请 API key（新用户有免费配额）
3. 在 `.env` 设置：
   ```
   EXECUTOR_BACKEND=e2b
   E2B_API_KEY=<your key>
   ```
4. 跑 demo 时切回 local 不需要改代码，只改 env：
   ```
   EXECUTOR_BACKEND=local
   ```

---

## Quickstart

### 1. 环境

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv venv
uv sync --extra dev
uv pip install -e .
```

> 国内网络不畅可加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env, 填入对应 provider 的 API key
```

> ⚠️ **重要**：`.env` 必须是**纯 ASCII**。`langgraph dev` 底层用的 python-dotenv 在 Windows 上默认按 GBK 编码读 `.env`，遇到 UTF-8 中文/em-dash 会直接崩。所以注释也请用英文或 ASCII 符号。

### 3. 跑通 demo

```bash
# 默认任务: 修复 buggy_ik workspace (启动时自动从 template 重置)
uv run python scripts/run_demo.py

# --stream 模式: 逐节点 print, 实时看 agent 思考过程 (强烈推荐)
uv run python scripts/run_demo.py --stream

# 自定义任务
uv run python scripts/run_demo.py "请用 grep_codebase 找出所有 TODO 注释并列举"
```

切换 provider：

```bash
# Windows PowerShell
$env:MODEL_PROVIDER="openai"; uv run python scripts/run_demo.py

# bash
MODEL_PROVIDER=openai uv run python scripts/run_demo.py
```

### 4. 跑测试

```bash
# 离线验收：不消耗 LLM/E2B API 额度，也是 CI 默认执行项
uv run pytest -q -m "not e2e"

# 完整测试：配置好当前模型和 E2B 凭据后执行
uv run pytest -v
```

带 `e2e` marker 的测试会真实调用 LLM 或 E2B；缺少对应 key/依赖时自动跳过。

### 5. M1 机器人任务评测

```bash
# 查看首批案例，不调用 LLM
uv run python scripts/run_benchmark.py --list

# 确认所有原始 fixture 都会被验证器判为失败，不调用 LLM
uv run python scripts/run_benchmark.py --validate-fixtures

# M1.0 单次通用策略基线（便于与已有报告对比）
uv run python scripts/run_benchmark.py --case angle_units --strategy general --repair-attempts 0 --max-loops 12

# M1.1 聚焦策略：直读目标、强制官方验证，失败后反馈修复一次
uv run python scripts/run_benchmark.py --case quaternion_normalization --strategy focused --repair-attempts 1 --max-loops 6

# M1.2 可选覆盖工具预算；不传时按 easy=5、medium=7、hard=9 自动分配
uv run python scripts/run_benchmark.py --case angle_units --strategy focused --tool-budget 5 --max-loops 6

# 稳定性评估：每题从原始 fixture 独立运行 3 次
uv run python scripts/run_benchmark.py --repeats 3 --strategy focused --repair-attempts 1 --max-loops 8 --task-timeout 300
```

每次运行都会从 `benchmarks/fixtures/` 重置一次性 workspace，并把成功率、耗时、
循环数、工具调用数和模型 token 写入 `benchmarks/results/<run-id>/report.json`。
M1.1 报告还记录每次尝试的官方 validator 调用合规率、模型宣称成功但独立验证失败的
false positive、反馈修复次数，以及多次运行的逐题成功率和平均成本。

长任务会实时输出题目、尝试次数、耗时和结果。每完成一题都会立即写入
`cases/repeat-<N>__<case-id>.json` 并原子更新 `checkpoint.json`；即使按 `Ctrl+C`
中断，已完成结果仍会保留。`--task-timeout` 是每次 Agent 尝试的硬超时（默认 300 秒），
超时子进程会被终止并计入报告；传 `--task-timeout 0` 可关闭超时。

M1.2 在 graph 路由层加入官方 validator gate：Focused Agent 若准备结束但没有成功的
官方验证证据，框架会自动执行验证；失败则把确定性证据送回修复循环。Focused 模式只
暴露 `read_file_chunk / write_patch / execute_python`，并压缩旧工具轮次，减少无关探索
和重复上下文。schema v4 报告增加逐题 `tool_budget`、`within_tool_budget` 与总体预算合规率。

### 6. LangGraph Studio 可视化（强烈推荐）

```bash
uv run langgraph dev
```

浏览器会自动打开 Studio，可以单步观察 `planner → reader → suggester` 的状态流转，是学习状态机理念最直观的方式。

> ⚠️ **Windows 中文环境必看**：Python 在中文 Windows 上默认 `open()` 编码是 GBK，会导致很多第三方库读自带的资源文件（如 OpenAPI schema）时崩溃。**一次性根治办法**：把 Python UTF-8 模式设为系统默认。
> ```powershell
> # 永久设置 (PowerShell 管理员或普通模式均可, 用户级)
> [System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'User')
> # 关闭并重开 PowerShell 窗口后生效
> ```
> 这个变量是 [PEP 540](https://peps.python.org/pep-0540/) 标准, 安全且向后兼容, 强烈建议所有中文 Windows 用户长期开启。

### 7. Phase 3.0 源码 RAG

安装 ChromaDB，同时保留开发和 E2B extras：

```bash
uv sync --extra dev --extra sandbox --extra rag
```

建立或增量刷新源码索引（默认只索引 `src/`，避免测试和故障 fixture 污染结果）：

```bash
uv run python scripts/index_codebase.py

# 同步后立即检查一次检索
uv run python scripts/index_codebase.py \
  --query "official validator gate benchmark completion" \
  --top-k 5

# 需要时显式加入其它源码根
uv run python scripts/index_codebase.py --root src --root examples
```

Phase 3.0 使用 AST 按模块、函数、类和方法切块，保存路径、符号、行号和内容哈希；
重复执行只更新变化 chunk，并删除已消失 chunk。默认 hashing embedding 完全本地运行，
无需下载模型或配置额外 API Key，向量持久化在被 Git 忽略的 `chroma_db/`。

通用 Agent 新增 `search_code_knowledge` 工具：目标未知、自然语言概念匹配或跨文件定位时
使用语义检索；精确字符串仍优先 `grep_codebase`，已知文件直接 `read_file_chunk`。
M1.2 focused benchmark 不暴露 RAG 工具，因此原有成本基线不会被改变。

### 8. Phase 3.1 RAG 评测

RAG 质量使用固定查询集 `benchmarks/rag_cases.json` 做确定性离线评测，不调用 LLM。
评测会自动增量同步索引，实时显示每条查询的首个相关结果排名，并将逐题结果和汇总报告
写入 `benchmarks/rag_results/<run-id>/`：

```bash
uv run python scripts/run_rag_eval.py

# 查看案例或只运行部分查询
uv run python scripts/run_rag_eval.py --list
uv run python scripts/run_rag_eval.py --case official_validator_gate --top-k 1 --top-k 5

# 用同一查询集比较向量与混合检索
uv run python scripts/run_rag_eval.py --strategy vector
uv run python scripts/run_rag_eval.py --strategy hybrid

# 已确认索引是最新时跳过同步
uv run python scripts/run_rag_eval.py --no-sync
```

报告包含 Hit Rate@K、Recall@K、MRR@K、目标文件命中率、目标符号命中率和平均查询延迟。
这些指标将作为后续混合检索、Embedding 替换和 Semantic Cache 的回归基线。
当前 10 条固定查询的向量基线为 Hit@1 50%、Hit@3 80%、Hit@5 90%、MRR@5 0.670；
Phase 3.1b 混合检索将其提升到 Hit@1 80%、Hit@3/5 100%、MRR@5 0.883，并将此前
Top-5 漏召回的 `semantic_search_tool` 提升到第 3 位。当前索引规模下，混合检索平均约
41 ms（包含安全的 revision 核对），仍远低于一次 LLM 调用。

### 9. Phase 3.2 Retrieval Cache

实际 Agent 的重复源码检索使用进程内 LRU Cache，默认最多保存 128 个结果。缓存键包含
标准化查询、`top_k`、路径前缀、检索策略和源码索引 revision。每次查询都会核对 Chroma
中的 chunk 内容哈希；本进程或其它进程刷新索引后，旧结果会自动清空，不会返回过期代码。

缓存记录 hit、miss、bypass、invalidation、eviction、命中率和估算节省延迟。可通过重复
查询检查行为：

```bash
uv run python scripts/index_codebase.py \
  --query "official validator gate" \
  --repeat-query 3
```

RAG 质量评测始终显式绕过缓存，并在报告中写入 `retrieval_cache_enabled: false`，因此
Phase 3.1/3.1b 的召回率和延迟基线不会被缓存命中污染。此缓存只保存确定性的只读检索结果，
不缓存补丁、工具执行、validator 或 LLM 回答。

### 10. Phase 4 Checkpointer 与 HITL

安装 SQLite Checkpointer，同时保留其它开发能力：

```bash
uv sync --extra dev --extra sandbox --extra rag --extra checkpoint
```

启动带持久 checkpoint 的普通 Agent。`--hitl` 会在每个 `write_patch` 前暂停；读取、检索
和沙盒执行仍自动进行。命令会输出稳定的 `thread_id`：

```bash
uv run python scripts/run_agent.py \
  "分析并修复 src/control.py 中的控制器问题" \
  --thread-id control-fix-01 \
  --hitl --stream
```

暂停后，文件尚未修改。可以关闭当前进程，检查提议后再从 SQLite 恢复：

```bash
# 查看 checkpoint、下一节点和待审批 patch
uv run python scripts/run_agent.py --thread-id control-fix-01 --status

# 批准并继续执行 write_patch
uv run python scripts/run_agent.py --thread-id control-fix-01 --resume approve

# 或拒绝，并把反馈交回 Planner
uv run python scripts/run_agent.py --thread-id control-fix-01 \
  --resume deny --feedback "不要改变公开函数签名"

# 从非 HITL 的普通待执行 checkpoint 继续
uv run python scripts/run_agent.py --thread-id control-fix-01 --continue
```

默认数据库位于 `.agent_state/checkpoints.sqlite`，可用 `--checkpoint-db` 覆盖，但路径仍须
位于项目目录内。M1 Benchmark 明确绕过 HITL，避免无人值守评测被审批暂停。旧的
`scripts/run_demo.py` 继续作为无持久化教学 Demo，持久任务统一使用 `scripts/run_agent.py`。

### 11. Phase 4.1 安全 Semantic Cache

Semantic Cache 只对显式只读会话开放。`--read-only` 会移除 `execute_python` 和
`write_patch`，只保留目录浏览、源码读取、grep 与 RAG；`--semantic-cache` 在此基础上缓存
最终纯文本回答。普通修复任务、Benchmark、工具调用、补丁和执行结果不会进入缓存。

```bash
# 首次运行：正常调用 LLM，最终回答写入缓存
uv run python scripts/run_agent.py \
  "解释 official validator gate 如何阻止假阳性" \
  --thread-id explain-gate-01 --read-only --semantic-cache

# 新会话中的相同或近重复问题可命中缓存
uv run python scripts/run_agent.py \
  "解释 official validator gate 如何阻止假阳性？" \
  --thread-id explain-gate-02 --read-only --semantic-cache
```

默认缓存位于 `.agent_state/semantic_cache.sqlite`，相似度阈值 0.97、TTL 7 天、最多 512 条。
缓存分区同时包含工作区内容指纹、模型标识、System Prompt 和只读工具 schema。源码、配置、
Prompt、模型或工具契约变化后不会复用旧回答。工作区指纹忽略 `.git`、虚拟环境、Chroma、
benchmark 结果和 `.agent_state` 等生成内容，避免缓存自身导致无意义失效。

这是回答级缓存而不是 Agent 轨迹缓存：命中时只返回以前的最终文本，绝不重放工具调用。
当前默认使用完全离线的确定性 Hashing Embedding，并刻意采用高阈值，只复用相同或近重复
问题，不承诺命中大幅改写的同义问法。环境变量 `SEMANTIC_CACHE_DB` 可以覆盖数据库位置，
但仍限制在项目目录内。

### 12. Phase 5.0 MuJoCo 闭环评测

安装无头 MuJoCo 仿真依赖：

```bash
uv sync --extra dev --extra checkpoint --extra simulation
```

首个真实动力学任务是单关节 PD 跟踪。Validator 在两个不同初始状态下跟踪正、负目标，
同时检查最终误差、尾段 RMSE、最终速度、稳定时间、最大关节偏移、数值有限性，以及控制器
自身是否遵守 8 N·m 扭矩限制。MuJoCo 执行器的物理限幅不能替代控制器内部限幅。

不调用 LLM 即可查看原始故障轨迹：

```bash
uv run python scripts/run_simulation.py

# 确认所有仿真题面都按设计失败
uv run python scripts/run_benchmark.py \
  --manifest benchmarks/sim_cases.json \
  --validate-fixtures
```

运行 Agent 修复闭环任务：

```bash
uv run python scripts/run_benchmark.py \
  --manifest benchmarks/sim_cases.json \
  --case sim_joint_pd_tracking \
  --strategy focused --repair-attempts 1 \
  --max-loops 10 --task-timeout 180
```

Phase 5 当前只支持 `EXECUTOR_BACKEND=local`。E2B 公共镜像未保证包含 MuJoCo；在没有自定义
沙盒镜像前，不应把 E2B 结果与本地仿真基线混合统计。仿真完全无 GUI、固定 timestep，
相同控制器的指标可逐值复现。

### 13. Phase 5.1 多关节轨迹与安全约束

第二个闭环任务扩展到平面 2-DOF 机械臂。目标不是只判断最终姿态，而是在两条平滑时变轨迹上
同时验证：

- 全程轨迹 RMSE、尾段 RMSE、最终最大关节误差和最终速度；
- 肩/肘关节分别为 10/8 N·m 的控制器内部扭矩限幅；
- MuJoCo 关节范围是否被突破；
- 连杆是否接触场景中的红色禁入区障碍物。

题面故意将两个关节的目标索引互换，并让原始控制量绕过 `torque_limits`。Agent 只修复
`TwoJointTrajectoryController`，官方仿真模型、轨迹和判分阈值保持不可修改。查看故障指标：

```bash
uv run python scripts/run_simulation.py --case two-joint
```

单独运行 Phase 5.1 Agent 修复任务：

```bash
uv run python scripts/run_benchmark.py \
  --manifest benchmarks/sim_cases.json \
  --case sim_two_joint_trajectory \
  --strategy focused --repair-attempts 1 \
  --max-loops 10 --task-timeout 180
```

`benchmarks/sim_cases.json` 包含单关节、双关节名义轨迹和鲁棒性任务；省略 `--case` 可连续评测。

### 14. Phase 5.2 固定种子扰动鲁棒性

鲁棒性评测不再只运行固定轨迹。默认的五个 seed 会分别生成安全的初始/目标关节状态，并对
以下条件进行有界扰动：

- 连杆质量与关节阻尼；
- 执行器输出衰减；
- 传感器高斯噪声与 0–3 timestep 控制延迟；
- 可复现的短时外力脉冲。

每个 seed 都创建新的控制器实例，避免有状态控制器在不同试验间泄漏状态。聚合报告包含通过率、
P50/P95 轨迹 RMSE、最坏最终误差、最坏尾段 RMSE、碰撞总数和 0–100 鲁棒性评分。碰撞、关节
越界、扭矩越界或非有限数值属于安全硬门禁，即使平均分较高也不能通过。

不调用 LLM 运行扰动矩阵：

```bash
uv run python scripts/run_robustness.py
```

结果默认写入 `benchmarks/results/robustness/<timestamp>/report.json`，同时在相邻 `seeds/`
目录为每个 seed 写入独立 JSON，失败可通过对应 seed 精确复现。可以用多个 `--seed` 覆盖默认
矩阵，或用 `--output` 指定项目内报告路径。

运行 Agent 的 Phase 5.2 泛化修复任务：

```bash
uv run python scripts/run_benchmark.py \
  --manifest benchmarks/sim_cases.json \
  --case sim_two_joint_robustness \
  --strategy focused --repair-attempts 1 \
  --max-loops 10 --task-timeout 180
```

---

## 项目结构

```
src/agent/
  benchmark.py    # M1 任务加载、运行、指标采集与报告
  state.py        # AgentState TypedDict (messages/current_file/loop_step + 预留)
  graph.py        # build_graph() + 顶层 graph
  persistence.py  # Phase 4 SQLite Checkpointer 生命周期
  semantic_cache.py # Phase 4.1 工作区隔离的只读回答缓存
  simulation/     # Phase 5 MuJoCo 无头闭环与指标
  config.py       # get_chat_model() 模型工厂
  nodes/          # planner / reader / suggester (状态→状态的纯函数)
  tools/          # read_file (Phase 2 扩展为 chunk/search/execute_python)
  rag/             # Phase 3 AST 切块、hash embedding、Chroma 增量索引
tests/
  fixtures/buggy_ik.py   # 故意带 bug 的 2-link IK 示例
  test_graph_e2e.py      # 端到端 + 单元测试
scripts/
  run_demo.py     # CLI 演示
  run_benchmark.py # M1 benchmark CLI
  index_codebase.py # Phase 3 源码索引与查询 CLI
  run_rag_eval.py # Phase 3.1 RAG 检索质量评测 CLI
  run_agent.py    # Phase 4 持久会话、状态查看和 HITL 恢复 CLI
  run_simulation.py # Phase 5 单独运行控制器闭环并输出指标
  run_robustness.py # Phase 5.2 固定 seed 扰动矩阵与逐 seed 报告
benchmarks/
  cases.json      # 评测任务清单
  rag_cases.json  # RAG 固定查询与相关路径/符号标注
  sim_cases.json  # MuJoCo 闭环修复任务清单
  fixtures/       # 永不修改的带 bug 题面
  validators.py   # 独立确定性判分器
langgraph.json    # Studio 入口
```

---

## 后续阶段路线图

| Phase | 主题 | 状态 |
|-------|------|------|
| 1 | 线性骨架 (planner → reader → suggester) | ✅ 完成 |
| 2 | ReAct + 5 工具集 (本地 sandbox) | ✅ 完成 |
| 2.5 | E2B Code Interpreter 真沙盒（与 local 并存，env 切换） | ✅ 完成 |
| M1 | 机器人代码任务评测、确定性验证门禁与成本预算 | 🚧 M1.2 |
| 3.0 | Agentic RAG — Chroma 源码索引与条件式检索 | ✅ 完成 |
| 3.1 | RAG 评测 — Recall@K、MRR、文件/符号命中率 | ✅ 完成 |
| 3.1b | 混合检索 — 语义 + 词法/符号召回与重排 | ✅ 完成 |
| 3.2 | 安全 Retrieval Cache — 索引版本感知、命中与失效指标 | ✅ 完成 |
| 4 | Checkpointer (SqliteSaver) + HITL interrupt | ✅ 完成 |
| 4.1 | LLM Semantic Cache — 工作区指纹隔离，仅限安全只读场景 | ✅ 完成 |
| 5.0 | 机器人闭环 — MuJoCo 单关节 PD 跟踪与安全指标 | ✅ 完成 |
| 5.1 | 多关节闭环 — 2-DOF 轨迹跟踪、碰撞与约束评测 | ✅ 完成 |
| 5.2 | 仿真评测扩展 — 固定种子矩阵、扰动鲁棒性与聚合评分 | ✅ 完成 |
| 5.3 | 仿真诊断 — 失败轨迹时序落盘与可视化报告 | ⏳ 下一阶段 |
| 6.0 | Robotics Domain Pack — 可插拔领域工具、环境与 Validator 契约 | 📋 规划中 |
| 6.1 | Autodiff CodeGen MVP — AD 兼容检查、CppAD tape 与 Jacobian 验证 | 📋 规划中 |
| 6.2 | 数值正确性门禁 — FD/AD/CodeGen 等价性与固定 seed 验证 | 📋 规划中 |
| 6.3 | Pinocchio 集成 — 机器人动力学 CodeGen 与解析导数性能决策 | 📋 规划中 |
| 6.4 | Crocoddyl/MPC — `calcDiff()` 瓶颈定位与端到端吞吐验证 | 📋 规划中 |
| M2 | Robo-SWE 专业 Benchmark — 20–30 题与 Harness/RAG/Tools/Validator 消融 | 📋 规划中 |

CppADCodeGen 作为首个机器人优化专业能力包，而不是项目的全部。领域逻辑保持可插拔，建设顺序为
专业构建环境与工具 → 数值/性能 Validator → 专业文档 RAG。详细方案见
[Robotics Autodiff & CodeGen 专业能力路线图](docs/robotics-autodiff-codegen-roadmap.md)。
