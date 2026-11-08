#Расчёт метрик. accuracy, Δ, robustness score, worst-case, coverage, стоимость, latency, bootstrap-CI.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import RunLog

DEFAULT_TYPES = [
    "paraphrase",
    "typo",
    "mixed_lang",
    "redundancy",
    "missing_condition",
    "contradiction",
    "reorder",
]
INTENSITIES = ["easy", "medium", "hard"]


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_logs(path: str | Path) -> list[RunLog]:
    return [RunLog.model_validate(row) for row in read_jsonl(path)]


def logs_to_df(logs: list[RunLog]) -> pd.DataFrame:
    return pd.DataFrame([log.model_dump() for log in logs])


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def analyze_frames(
    baseline_df: pd.DataFrame,
    mutated_df: pd.DataFrame,
    target_agent: str = "langgraph",
    type_weights: dict[str, float] | None = None,
    expert_worst_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if baseline_df.empty:
        raise ValueError("baseline_df пуст")
    if mutated_df.empty:
        raise ValueError("mutated_df пуст")

    baseline_acc = baseline_df.groupby("agent")["correct"].mean().to_dict()
    agents = sorted(mutated_df["agent"].dropna().unique().tolist())

    result: dict[str, Any] = {
        "target_agent": target_agent,
        "baseline_accuracy": {k: float(v) for k, v in baseline_acc.items()},
        "agents": {},
        "expert_worst_case": expert_worst_case,
    }

    for agent in agents:
        base = baseline_acc.get(agent)
        if base is None:
            continue

        adf = mutated_df[mutated_df["agent"] == agent]
        combos: list[dict[str, Any]] = []

        for (mutation_type, intensity), group in adf.groupby(
            ["mutation_type", "intensity"], dropna=False
        ):
            vals = group["correct"].astype(bool).tolist()
            acc = float(np.mean(vals)) if vals else 0.0
            ci_low, ci_high = bootstrap_ci([1.0 if v else 0.0 for v in vals])

            combos.append(
                {
                    "mutation_type": str(mutation_type),
                    "intensity": str(intensity),
                    "n": int(len(vals)),
                    "accuracy": acc,
                    "delta": float(base - acc),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "avg_steps": float(group["steps"].mean()),
                    "avg_cost_usd": float(group["cost_usd"].mean()),
                    "avg_latency_ms": float(group["latency_ms"].mean()),
                }
            )

        if not combos:
            continue

        weights = type_weights or {}
        wsum = 0.0
        wdelta = 0.0
        for combo in combos:
            w = float(weights.get(combo["mutation_type"], 1.0))
            wsum += w
            wdelta += w * combo["delta"]

        weighted_delta = wdelta / wsum if wsum else 0.0
        robustness = 1.0 - (weighted_delta / base) if base else None
        worst = max(combos, key=lambda c: c["delta"])

        type_summary: dict[str, dict[str, float]] = {}
        for mutation_type in sorted({c["mutation_type"] for c in combos}):
            sub = [c for c in combos if c["mutation_type"] == mutation_type]
            type_summary[mutation_type] = {
                "mean_delta": float(np.mean([c["delta"] for c in sub])),
                "n_combos": int(len(sub)),
            }

        coverage = len(combos) / (len(DEFAULT_TYPES) * len(INTENSITIES))

        result["agents"][agent] = {
            "baseline_accuracy": float(base),
            "weighted_delta": float(weighted_delta),
            "robustness_score": robustness,
            "worst_case": worst,
            "coverage": float(coverage),
            "type_summary": type_summary,
            "combos": sorted(combos, key=lambda c: (c["mutation_type"], c["intensity"])),
        }

    if target_agent not in result["agents"]:
        if not result["agents"]:
            raise ValueError("Нет ни одного агента с baseline accuracy")
        target_agent = next(iter(result["agents"]))
        result["target_agent"] = target_agent

    return result


def analyze_files(
    baseline_path: str | Path,
    logs_path: str | Path,
    target_agent: str = "langgraph",
    weights_path: str | Path | None = None,
    expert_worst_case_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline_df = logs_to_df(load_logs(baseline_path))
    mutated_df = logs_to_df(load_logs(logs_path))

    type_weights = None
    if weights_path:
        type_weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))

    expert = None
    if expert_worst_case_path:
        expert = json.loads(Path(expert_worst_case_path).read_text(encoding="utf-8"))

    return analyze_frames(
        baseline_df=baseline_df,
        mutated_df=mutated_df,
        target_agent=target_agent,
        type_weights=type_weights,
        expert_worst_case=expert,
    )


def build_feedback(result: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    agent = result["agents"][result["target_agent"]]
    ranked = sorted(agent["type_summary"].items(), key=lambda kv: kv[1]["mean_delta"], reverse=True)
    wc = agent["worst_case"]
    
    return {
        "target_agent (целевой агент)": result["target_agent"],
        "worst_case (худший случай)": {
            "mutation_type (тип мутации)": wc["mutation_type"],
            "intensity (интенсивность)": wc["intensity"],
            "n (количество тестов)": wc["n"],
            "accuracy (точность)": round(wc["accuracy"], 3),
            "delta (падение)": round(wc["delta"], 3),
            "ci (доверительный интервал)": [round(wc["ci_low"], 3), round(wc["ci_high"], 3)],
            "avg_steps (среднее шагов)": round(wc["avg_steps"], 2),
            "avg_cost_usd (средняя стоимость)": round(wc["avg_cost_usd"], 4),
        },
        "top_mutation_types (топ типов мутаций)": [
            {
                "mutation_type (тип мутации)": k,
                "mean_delta (среднее падение)": round(float(v["mean_delta"]), 3),
                "n_combos (количество комбинаций)": int(v["n_combos"])
            }
            for k, v in ranked[:top_k]
        ],
        "action (рекомендация)": "Увеличить число мутаций этих типов и проверить интенсивности с максимальным падением. Передать результат Мутатору."
    }
