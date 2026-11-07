#Мой LangGraph-аналитик
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .metrics import analyze_files, build_feedback
from .report import render_markdown


class AnalystState(TypedDict, total=False):
    baseline_path: str
    logs_path: str
    target_agent: str
    weights_path: str | None
    expert_worst_case_path: str | None

    result: dict
    feedback: dict
    report_md: str


def compute_node(state: AnalystState):
    result = analyze_files(
        baseline_path=state["baseline_path"],
        logs_path=state["logs_path"],
        target_agent=state.get("target_agent", "langgraph"),
        weights_path=state.get("weights_path"),
        expert_worst_case_path=state.get("expert_worst_case_path"),
    )
    return {"result": result}


def feedback_node(state: AnalystState):
    return {"feedback": build_feedback(state["result"])}


def report_node(state: AnalystState):
    return {"report_md": render_markdown(state["result"])}


def build_analyst_graph():
    builder = StateGraph(AnalystState)
    builder.add_node("compute", compute_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "compute")
    builder.add_edge("compute", "feedback")
    builder.add_edge("feedback", "report")
    builder.add_edge("report", END)

    return builder.compile()
