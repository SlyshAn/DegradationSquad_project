#Тесты, чтобы убедиться, что работает, на выдуманных логах
from __future__ import annotations

import pandas as pd

from src.analyst.metrics import analyze_frames, build_feedback
from src.analyst.report import render_markdown


def test_worst_case_coverage_and_robustness():
    baseline = pd.DataFrame(
        [
            {"agent": "langgraph", "correct": True},
            {"agent": "langgraph", "correct": True},
        ]
    )

    mutated = pd.DataFrame(
        [
            {
                "agent": "langgraph",
                "mutation_type": "paraphrase",
                "intensity": "easy",
                "correct": True,
                "steps": 2,
                "cost_usd": 0.01,
                "latency_ms": 100,
            },
            {
                "agent": "langgraph",
                "mutation_type": "paraphrase",
                "intensity": "easy",
                "correct": False,
                "steps": 3,
                "cost_usd": 0.02,
                "latency_ms": 200,
            },
            {
                "agent": "langgraph",
                "mutation_type": "missing_condition",
                "intensity": "hard",
                "correct": False,
                "steps": 5,
                "cost_usd": 0.05,
                "latency_ms": 500,
            },
        ]
    )

    result = analyze_frames(baseline, mutated, target_agent="langgraph")
    agent = result["agents"]["langgraph"]

    assert agent["baseline_accuracy"] == 1.0
    assert agent["worst_case"]["mutation_type"] == "missing_condition"
    assert agent["worst_case"]["intensity"] == "hard"
    assert agent["coverage"] == 2 / 21
    assert agent["robustness_score"] == 0.25


def test_feedback_and_report_contain_worst_case():
    baseline = pd.DataFrame([{"agent": "langgraph", "correct": True}])
    mutated = pd.DataFrame(
        [
            {
                "agent": "langgraph",
                "mutation_type": "missing_condition",
                "intensity": "hard",
                "correct": False,
                "steps": 4,
                "cost_usd": 0.03,
                "latency_ms": 300,
            }
        ]
    )

    result = analyze_frames(baseline, mutated, target_agent="langgraph")
    feedback = build_feedback(result)
    report = render_markdown(result)

    assert feedback["worst_case"]["mutation_type"] == "missing_condition"
    assert "missing_condition" in report
    assert "Robustness score" in report
