"""planner 节点 —— 从用户 task 中抽取目标文件路径。

用 Pydantic 结构化输出, 避免 LLM 自由发挥导致路径格式漂移。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.config import get_chat_model
from agent.state import AgentState

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SYSTEM = """你是一个机器人算法仓库的排障 Agent 的 Planner 节点。
你的唯一职责: 从用户指令中识别出**唯一一个**需要分析的目标文件相对路径。

规则:
- 路径必须相对于仓库根目录, 使用正斜杠 (e.g. "tests/fixtures/buggy_ik.py")。
- 如果用户在指令中已经写明路径, 直接抽取。
- 不要捏造不存在的文件。如果用户没有给出明确文件, 在 reasoning 中说明。
"""


class PlannerOutput(BaseModel):
    target_file: str = Field(description="相对于仓库根目录的目标文件路径")
    reasoning: str = Field(description="为什么选这个文件 (一两句话)")


def planner(state: AgentState) -> dict:
    task = state["task"]
    # method="function_calling" 是三家 provider 最大公约数:
    #   - OpenAI: 走 functions / tools
    #   - Anthropic: 走 tool use
    #   - DeepSeek: 走 OpenAI 兼容的 functions (它不支持 json_schema 严格模式)
    llm = get_chat_model(temperature=0.0).with_structured_output(
        PlannerOutput, method="function_calling"
    )
    result: PlannerOutput = llm.invoke(  # type: ignore[assignment]
        [
            SystemMessage(content=_SYSTEM),
            ("human", f"用户指令: {task}"),
        ]
    )

    # 路径存在性校验 —— 不存在则保留 LLM 的猜测但记录到 message, 由 reader 节点报错
    target = result.target_file.strip().replace("\\", "/")
    abs_path = (_PROJECT_ROOT / target).resolve()
    exists = abs_path.exists()

    log = (
        f"[planner] target_file={target} (exists={exists})\n"
        f"          reasoning={result.reasoning}"
    )

    return {
        "current_file": target,
        "messages": [AIMessage(content=log)],
    }
