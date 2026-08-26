"""planner 节点 —— Phase 2 起是 ReAct 循环的"大脑"。

行为:
1. 若 messages 为空且 state.task 有值, 把 task 转成首条 HumanMessage。
2. 调 LLM (绑定全部工具) 一次, 输出可能含 tool_calls 的 AIMessage。
3. ``loop_step += 1``; 路由判断交给 ``should_continue`` 条件边。
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.config import get_chat_model
from agent.state import AgentState
from agent.tools import ALL_TOOLS, FOCUSED_BENCHMARK_TOOLS

_DEFAULT_MAX_LOOP_STEPS = 15

_SYSTEM = """你是一个面向机器人算法仓库的自主排障 Agent。

你可以调用以下工具来探索和修改代码:
- `list_dir(path, max_depth)` —— 浏览目录结构
- `read_file_chunk(path, offset, limit)` —— 分段读取文件 (返回带行号)
- `grep_codebase(pattern, glob, max_results)` —— 正则搜索整个仓库
- `search_code_knowledge(query, top_k, path_prefix)` —— 语义检索已建立的源码索引
- `execute_python(code, timeout)` —— 在沙盒中跑 Python, 复现 bug 或验证修复
- `write_patch(path, old_string, new_string)` —— 精确字符串替换修改文件 (old_string 必须唯一)

工作流程 (严格按这个顺序):
1. **理解任务**: 目标未知或跨文件概念定位时先用 search_code_knowledge；精确字符串用
   grep_codebase；已知文件直接 read_file_chunk。不要无条件同时调用所有搜索工具。
2. **形成假设**: 用 execute_python 复现 bug、打印中间变量, 拿到具体证据。
3. **应用修复**: 用 write_patch 实施改动。每次修一处, 改完用 read_file_chunk 复核。
4. **验证**: 再次用 execute_python 跑相同的测试代码, 确认问题消失。
5. **总结**: 完成后**不再调用工具**, 直接输出 markdown 总结 (含问题、修复、验证证据)。

注意:
- write_patch 的 old_string 必须在文件中唯一出现; 不唯一时多包几行上下文。
- 不要一次性大改; 每个 write_patch 改一处, 改后立即验证。
- 路径都相对仓库根 (e.g. "tests/fixtures/buggy_ik.py")。
- 你的总循环预算有限 ({max_loop_steps} 步), 不要做无关探索。
"""


_BENCHMARK_FOCUSED = """

Benchmark focused mode (this section overrides conflicting exploration steps above):
- The task gives an exact target file and an official validator command.
- Read the target file directly. Do not call list_dir or grep_codebase unless the
  target cannot be read or the supplied information is internally inconsistent.
- Make the smallest targeted patch, then run the exact supplied official validator
  with execute_python. A self-written check is not a substitute.
- Do not write a separate reproduction script and do not re-read after a successful
  write_patch. The official validator is the sole post-patch check.
- Stay within the supplied tool-call budget. Prefer exactly: read target, patch,
  official validator, then finish.
- Do not claim completion unless the official validator reports passed=True.
- If it fails, use its details as evidence and continue fixing within the loop budget.
"""


def build_system_prompt(state: AgentState) -> str:
    """Build the planner prompt and apply the requested benchmark strategy."""
    max_steps = state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS)
    prompt = _SYSTEM.format(max_loop_steps=max_steps)
    if state.get("benchmark_mode") and state.get("benchmark_strategy", "focused") == "focused":
        budget = state.get("benchmark_tool_budget", 6)
        prompt += _BENCHMARK_FOCUSED + f"\nTool-call budget: {budget}.\n"
    return prompt


def focused_message_context(
    messages: list[BaseMessage], tool_rounds: int = 3
) -> list[BaseMessage]:
    """Keep the task plus recent complete tool rounds for benchmark LLM calls."""
    if tool_rounds < 1 or len(messages) <= 2:
        return messages
    round_starts = [
        index
        for index, message in enumerate(messages)
        if getattr(message, "tool_calls", None)
    ]
    if len(round_starts) <= tool_rounds:
        return messages
    start = round_starts[-tool_rounds]
    return [
        messages[0],
        HumanMessage(content="Earlier benchmark tool rounds omitted; use the current workspace state."),
        *messages[start:],
    ]


def planner(state: AgentState) -> dict:
    messages = list(state.get("messages") or [])
    loop_step = state.get("loop_step", 0)

    # 第一次进入时, 把 task 注入为首条 HumanMessage
    if not messages:
        task = state.get("task")
        if not task:
            raise ValueError("AgentState must contain either 'messages' or 'task'.")
        messages = [HumanMessage(content=task)]

    focused = bool(
        state.get("benchmark_mode")
        and state.get("benchmark_strategy", "focused") == "focused"
    )
    tools = FOCUSED_BENCHMARK_TOOLS if focused else ALL_TOOLS
    llm = get_chat_model(temperature=0.0).bind_tools(tools)

    system = SystemMessage(content=build_system_prompt(state))
    model_messages = (
        focused_message_context(messages, state.get("benchmark_context_rounds", 3))
        if focused
        else messages
    )
    response = llm.invoke([system, *model_messages])

    update: dict = {
        "messages": [response],
        "loop_step": loop_step + 1,
    }

    # 若 LLM 已无工具调用, 把回答内容同步到 ``suggestion`` (Phase 1 兼容)
    if not getattr(response, "tool_calls", None):
        content = response.content if isinstance(response.content, str) else str(response.content)
        update["suggestion"] = content

    return update
