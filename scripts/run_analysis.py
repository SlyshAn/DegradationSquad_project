from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyst.plots import generate_delta_plot
from analyst.adapters import adapt_robustness_report
from analyst.metrics import analyze_frames, build_feedback, load_logs, logs_to_df
from analyst.report import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["jsonl", "robustness_report"], default="jsonl")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--logs", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--enriched", default="data/clean/v0.1_enriched.jsonl")
    parser.add_argument("--mutations", default="data/calibration/manual_mutations.jsonl")
    parser.add_argument("--target-agent", default="langgraph")
    parser.add_argument("--out", default="reports/robustness_profile.md")
    parser.add_argument("--metrics-json", default="reports/metrics.json")
    parser.add_argument("--feedback-json", default="reports/feedback_to_mutator.json")

    args = parser.parse_args()

    if args.format == "jsonl":
        if not args.baseline or not args.logs:
            raise SystemExit("Для --format jsonl нужны --baseline и --logs")
        baseline_df = logs_to_df(load_logs(args.baseline))
        mutated_df = logs_to_df(load_logs(args.logs))
    else:
        if not args.report:
            raise SystemExit("Для --format robustness_report нужен --report")
        logs = adapt_robustness_report(args.report, args.enriched, args.mutations, args.target_agent)
        df = logs_to_df(logs)
        baseline_df = df[df["mutation_type"] == "clean"]
        mutated_df = df[df["mutation_type"] != "clean"]

    result = analyze_frames(baseline_df, mutated_df, target_agent=args.target_agent)
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

        # Генерация графиков
    try:
        plot_path = generate_delta_plot(result)
        print("График сохранен:", plot_path)
    except Exception as e:
        print("Не удалось построить график:", e)

    agent = result["agents"][result["target_agent"]]
    worst = agent["worst_case"]
    print("Target agent:", result["target_agent"])
    print("Robustness score:", round(agent["robustness_score"], 3))
    print("Coverage:", f"{agent['coverage']:.1%}")
    print("Worst case:", worst["mutation_type"], worst["intensity"], "delta=", round(worst["delta"], 3))
    print("Reports:", args.out, args.metrics_json, args.feedback_json)


if __name__ == "__main__":
    main()
