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

    # Делаем красивый словарь для metrics.json с переводами
    metrics_pretty = {
        "target_agent (целевой агент)": result["target_agent"],
        "baseline_accuracy (базовая точность)": {k: v for k, v in result["baseline_accuracy"].items()},
        "agents (агенты)": {}
    }
    for agent_name, agent_data in result["agents"].items():
        metrics_pretty["agents (агенты)"][agent_name] = {
            "baseline_accuracy (базовая точность)": agent_data["baseline_accuracy"],
            "weighted_delta (взвешенное падение)": agent_data["weighted_delta"],
            "robustness_score (оценка устойчивости)": agent_data["robustness_score"],
            "coverage (покрытие)": agent_data["coverage"],
            "worst_case (худший случай)": {
                "mutation_type (тип мутации)": agent_data["worst_case"]["mutation_type"],
                "intensity (интенсивность)": agent_data["worst_case"]["intensity"],
                "accuracy (точность)": agent_data["worst_case"]["accuracy"],
                "delta (падение)": agent_data["worst_case"]["delta"],
            },
            "type_summary (сводка по типам)": {
                k: {"mean_delta (среднее падение)": v["mean_delta"], "n_combos (комбинаций)": v["n_combos"]}
                for k, v in agent_data["type_summary"].items()
            },
            "combos (все комбинации)": [
                {
                    "mutation_type (тип)": c["mutation_type"],
                    "intensity (интенсивность)": c["intensity"],
                    "n (тестов)": c["n"],
                    "accuracy (точность)": c["accuracy"],
                    "delta (падение)": c["delta"],
                    "ci_low (ДИ низ)": c["ci_low"],
                    "ci_high (ДИ верх)": c["ci_high"],
                    "avg_steps (шагов)": c["avg_steps"],
                    "avg_cost_usd (стоимость)": c["avg_cost_usd"],
                    "avg_latency_ms (задержка)": c["avg_latency_ms"],
                } for c in agent_data["combos"]
            ]
        }

    for path, content in [
        (args.metrics_json, json.dumps(metrics_pretty, ensure_ascii=False, indent=2)),
        (args.feedback_json, json.dumps(feedback, ensure_ascii=False, indent=2)),
        (args.out, report_md),
    ]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")

    try:
        from analyst.plots import generate_delta_plot
        plot_path = generate_delta_plot(result)
        print("График сохранен:", plot_path)
    except Exception as e:
        print("Не удалось построить график:", e)

    agent = result["agents"][result["target_agent"]]
    worst = agent["worst_case"]
    print("Целевой агент (Target Agent):", result["target_agent"])
        # Безопасный вывод (защита от None, если базовая точность 0)
    rs = agent["robustness_score"]
    rs_print = round(rs, 3) if rs is not None else "Н/Д (базовая точность = 0)"
    print("Оценка устойчивости (Robustness score):", rs_print)
    print("Покрытие (Coverage):", f"{agent['coverage']:.1%}")
    print("Худший случай (Worst case):", worst["mutation_type"], worst["intensity"], "падение (delta) =", round(worst["delta"], 3))
    print("Отчеты сохранены в:", args.out, args.metrics_json, args.feedback_json)

if __name__ == "__main__":
    main()
