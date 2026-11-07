from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyst.metrics import analyze_files, build_feedback
from analyst.report import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--target-agent", default="langgraph")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--expert-worst-case", default=None)

    parser.add_argument("--out", default="reports/robustness_profile.md")
    parser.add_argument("--metrics-json", default="reports/metrics.json")
    parser.add_argument("--feedback-json", default="reports/feedback_to_mutator.json")

    args = parser.parse_args()

    result = analyze_files(
        baseline_path=args.baseline,
        logs_path=args.logs,
        target_agent=args.target_agent,
        weights_path=args.weights,
        expert_worst_case_path=args.expert_worst_case,
    )
    feedback = build_feedback(result)
    report_md = render_markdown(result)

    for path, content in [
        (args.metrics_json, json.dumps(result, ensure_ascii=False, indent=2)),
        (args.feedback_json, json.dumps(feedback, ensure_ascii=False, indent=2)),
        (args.out, report_md),
    ]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")

    agent = result["agents"][result["target_agent"]]
    worst = agent["worst_case"]
    print("Target agent:", result["target_agent"])
    print("Robustness score:", round(agent["robustness_score"], 3))
    print("Coverage:", f"{agent['coverage']:.1%}")
    print(
        "Worst case:",
        worst["mutation_type"],
        worst["intensity"],
        "delta=",
        round(worst["delta"], 3),
    )
    print("Reports:", args.out, args.metrics_json, args.feedback_json)


if __name__ == "__main__":
    main()
