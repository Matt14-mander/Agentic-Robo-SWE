"""reader 节点 —— 调 ``read_file`` 工具读取 ``current_file`` 全文。

Phase 1 不走 LLM, 是纯工具调用; Phase 2 改为 ToolNode 后由 LLM 决定调用时机。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.state import AgentState
from agent.tools.file_ops import read_file


def reader(state: AgentState) -> dict:
    path = state.get("current_file")
    if not path:
        return {
            "file_content": None,
            "messages": [AIMessage(content="[reader] current_file 为空, 跳过读取")],
        }

    try:
        content = read_file(path)
    except (FileNotFoundError, ValueError) as e:
        return {
            "file_content": None,
            "messages": [AIMessage(content=f"[reader] 读取失败: {e}")],
        }

    n_lines = content.count("\n") + 1
    log = f"[reader] 已读取 {path} ({n_lines} 行, {len(content)} 字符)"
    return {
        "file_content": content,
        "messages": [AIMessage(content=log)],
    }
