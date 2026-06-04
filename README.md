# Agentic-Robo-SWE

An autonomous, simulation-in-the-loop software engineering agent for robotics. Built with LangGraph to debug and verify embodied AI algorithms.

设计哲学：把 **LangGraph 显式状态机**（可视、可中断、可恢复）与 **Princeton SWE-agent 的 Agent-Computer Interface** 思想（为 LLM 量身设计的工具集，而非裸 shell）结合起来，做一个能自主排障机器人物理算法仓库的 Agent。

---

## Phase 1：线性骨架（当前阶段）

```
START → planner → reader → suggester → END
```

- **planner** —— 用结构化输出从用户 task 中抽取目标文件路径。
- **reader** —— 调 `read_file` 工具读取文件全文。
- **suggester** —— LLM 综合 task + file_content 输出 markdown 修复建议（含 diff）。

模型层支持 **Claude 3.5 Sonnet / GPT-4o-mini / DeepSeek-Coder**，通过 `MODEL_PROVIDER` 环境变量切换，零代码改动。

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
uv run python scripts/run_demo.py "请分析并修复 tests/fixtures/buggy_ik.py 中的 bug"
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
uv run pytest -v
```

`test_linear_flow_*` 会真实调用 LLM；无 key 时自动跳过。

### 5. LangGraph Studio 可视化（强烈推荐）

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

---

## 项目结构

```
src/agent/
  state.py        # AgentState TypedDict (messages/current_file/loop_step + 预留)
  graph.py        # build_graph() + 顶层 graph
  config.py       # get_chat_model() 模型工厂
  nodes/          # planner / reader / suggester (状态→状态的纯函数)
  tools/          # read_file (Phase 2 扩展为 chunk/search/execute_python)
tests/
  fixtures/buggy_ik.py   # 故意带 bug 的 2-link IK 示例
  test_graph_e2e.py      # 端到端 + 单元测试
scripts/
  run_demo.py     # CLI 演示
langgraph.json    # Studio 入口
```

---

## 后续阶段路线图

| Phase | 主题 | 关键变化 |
|-------|------|---------|
| 2 | ReAct + 工具集 | 加 `execute_python` (E2B 沙盒), `read_file_chunk`, `grep_codebase`；planner 升级为 router |
| 3 | Agentic RAG | chromadb 索引源码 + 论文 PDF，新增 `search_codebase` / `search_docs` |
| 4 | Checkpointer + HITL | SqliteSaver 持久化，suggester 前 `interrupt_before` 人工 review |
| 5 | 机器人闭环 | `run_simulation` 接 PyBullet/MuJoCo，构造 IK/控制器/SLAM bug 评测集 |

详细方案见 `C:\Users\Rog\.claude\plans\langgraph-swe-agent-agent-code-executio-shiny-dragonfly.md`。
