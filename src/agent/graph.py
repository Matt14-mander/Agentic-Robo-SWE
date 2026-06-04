"""图编排 —— Phase 1 线性骨架。

  START → planner → reader → suggester → END

后续阶段:
- Phase 2 把 reader 换成 ToolNode, planner 升级为 router (条件边)。
- Phase 4 在 ``compile()`` 时传 checkpointer 启用持久化。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import planner, reader, suggester
from agent.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("reader", reader)
    workflow.add_node("suggester", suggester)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "reader")
    workflow.add_edge("reader", "suggester")
    workflow.add_edge("suggester", END)

    return workflow.compile()


# Studio / langgraph.json 通过此顶层名发现入口
graph = build_graph()
