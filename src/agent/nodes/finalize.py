"""finalize 节点 —— 循环预算耗尽时的收尾节点。

当 ``should_continue`` 发现 loop_step 已达 max_loop_steps, 但 planner 最后一条
消息还带着待执行的 tool_calls 时, 路由到这里而不是直接 END:

1. 给每个悬挂的 tool_call 补一条 "SKIPPED" 的 ToolMessage —— Anthropic/OpenAI
   的 API 都要求每个 tool_use 之后必须紧跟对应的 tool_result, 否则下一次
   llm.invoke 会直接报错。
2. 再调一次 LLM (不绑工具), 逼它基于已有信息输出总结, 而不是让熔断悄无声息地
   发生、用户只拿到一堆没有结论的消息。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, ToolMessage

from agent.config import get_chat_model
from agent.state import AgentState

_FINALIZE_SYSTEM = """你的操作步数预算已经用完, 不能再调用任何工具。

请基于目前为止已经做过的探索/修改/验证, 直接输出一段 markdown 总结, 包含:
- 你诊断出的问题 (如果有)
- 你已经做的修改 (如果有, 引用具体文件)
- 尚未完成或未验证的部分
- 建议的下一步

不要道歉、不要请求更多步数, 直接给出总结。
"""


def finalize(state: AgentState) -> dict:
    messages = list(state.get("messages") or [])
    skip_messages: list[ToolMessage] = []

    if messages:
        last = messages[-1]
        pending_calls = getattr(last, "tool_calls", None) or []
        for call in pending_calls:
            skip_messages.append(
                ToolMessage(
                    content="SKIPPED: step budget exhausted before this tool call could run.",
                    tool_call_id=call["id"],
                    name=call.get("name", "unknown"),
                )
            )

    # 不绑工具 —— 就算 LLM 想继续调用, 也没有工具可选, 只能吐文本。
    llm = get_chat_model(temperature=0.0)
    system = SystemMessage(content=_FINALIZE_SYSTEM)
    response = llm.invoke([system, *messages, *skip_messages])

    content = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "messages": [*skip_messages, response],
        "suggestion": content,
    }
