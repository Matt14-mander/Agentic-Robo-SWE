"""planner 节点 —— Phase 2 起是 ReAct 循环的"大脑"。

行为:
1. 若 messages 为空且 state.task 有值, 把 task 转成首条 HumanMessage。
2. 调 LLM (绑定全部工具) 一次, 输出可能含 tool_calls 的 AIMessage。
3. ``loop_step += 1``; 路由判断交给 ``should_continue`` 条件边。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import get_chat_model
from agent.state import AgentState
from agent.tools import ALL_TOOLS

_DEFAULT_MAX_LOOP_STEPS = 15

_SYSTEM = """你是一个面向机器人算法仓库的自主排障 Agent。

你可以调用以下工具来探索和修改代码:
- `list_dir(path, max_depth)` —— 浏览目录结构
- `read_file_chunk(path, offset, limit)` —— 分段读取文件 (返回带行号)
- `grep_codebase(pattern, glob, max_results)` —— 正则搜索整个仓库
- `execute_python(code, timeout)` —— 在沙盒中跑 Python, 复现 bug 或验证修复
- `write_patch(path, old_string, new_string)` —— 精确字符串替换修改文件 (old_string 必须唯一)

工作流程 (严格按这个顺序):
1. **理解任务**: 用 list_dir / grep / read_file_chunk 摸清现状。不要凭空猜文件位置。
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


def planner(state: AgentState) -> dict:
    messages = list(state.get("messages") or [])
    loop_step = state.get("loop_step", 0)
    max_steps = state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS)

    # 第一次进入时, 把 task 注入为首条 HumanMessage
    if not messages:
        task = state.get("task")
        if not task:
            raise ValueError("AgentState must contain either 'messages' or 'task'.")
        messages = [HumanMessage(content=task)]

    llm = get_chat_model(temperature=0.0).bind_tools(ALL_TOOLS)

    system = SystemMessage(content=_SYSTEM.format(max_loop_steps=max_steps))
    response = llm.invoke([system, *messages])

    update: dict = {
        "messages": [response],
        "loop_step": loop_step + 1,
    }

    # 若 LLM 已无工具调用, 把回答内容同步到 ``suggestion`` (Phase 1 兼容)
    if not getattr(response, "tool_calls", None):
        content = response.content if isinstance(response.content, str) else str(response.content)
        update["suggestion"] = content

    return update
