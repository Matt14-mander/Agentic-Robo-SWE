"""suggester 节点 —— 综合 task + file_content 给出修复建议。

输出 markdown, 含问题诊断、修复方案、diff 片段; 写回 ``state.suggestion``。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from agent.config import get_chat_model
from agent.state import AgentState

_SYSTEM = """你是一个资深机器人算法工程师, 同时精通 Python。
你的职责: 阅读用户给出的代码文件, 结合用户指令, 给出**具体可落地**的修复或改进建议。

输出格式 (markdown):
## 问题诊断
- 用 1-3 条 bullet 写清楚代码中存在的问题或风险, 涉及具体行号。

## 修复方案
- 描述思路, 解释为什么这么改。

## 代码修改 (diff)
```diff
- 旧代码
+ 新代码
```

如果文件内容缺失或无法判断问题, 直接说明 "无法分析: <原因>"。
"""

_USER_TEMPLATE = """用户指令:
{task}

目标文件: `{path}`

文件内容:
```python
{content}
```
"""


def suggester(state: AgentState) -> dict:
    task = state.get("task", "")
    path = state.get("current_file") or "<unknown>"
    content = state.get("file_content")

    if not content:
        msg = "无法分析: 文件内容为空或读取失败。"
        return {
            "suggestion": msg,
            "messages": [AIMessage(content=f"[suggester] {msg}")],
        }

    llm = get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            ("human", _USER_TEMPLATE.format(task=task, path=path, content=content)),
        ]
    )

    suggestion = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "suggestion": suggestion,
        "messages": [AIMessage(content=suggestion)],
    }
