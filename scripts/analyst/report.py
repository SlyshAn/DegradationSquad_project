#Генерация отчёта
from __future__ import annotations

from typing import Any


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    agent_name = result["target_agent"]
    agent = result["agents"][agent_name]
    worst = agent["worst_case"]

    lines = [
        "# Robustness profile",
        "",
        f"**Целевой агент:** `{agent_name}`",
        f"**Baseline accuracy:** {_fmt(agent['baseline_accuracy'])}",
        f"**Robustness score:** {_fmt(agent['robustness_score'])}",
        f"**Coverage:** {agent['coverage']:.1%}",
        "",
        "## Worst case",
        "",
        f"- Тип мутации: `{worst['mutation_type']}`",
        f"- Интенсивность: `{worst['intensity']}`",
        f"- Accuracy: {_fmt(worst['accuracy'])}",
        f"- Delta к baseline: {_fmt(worst['delta'])}",
        f"- 95% CI accuracy: [{_fmt(worst['ci_low'])}, {_fmt(worst['ci_high'])}]",
        "",
        "## Все комбинации",
        "",
        "| mutation_type | intensity | n | accuracy | delta | 95% CI | avg_steps | avg_cost_usd |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]

    for combo in agent["combos"]:
        ci = f"{_fmt(combo['ci_low'])}, {_fmt(combo['ci_high'])}"
        lines.append(
            "| {mutation_type} | {intensity} | {n} | {accuracy} | {delta} | {ci} | {steps} | {cost} |".format(
                mutation_type=combo["mutation_type"],
                intensity=combo["intensity"],
                n=combo["n"],
                accuracy=_fmt(combo["accuracy"]),
                delta=_fmt(combo["delta"]),
                ci=ci,
                steps=_fmt(combo["avg_steps"], 2),
                cost=_fmt(combo["avg_cost_usd"], 4),
            )
        )

    lines += [
        "",
        "## Интерпретация",
        "",
        "Сравнивайте не только среднюю accuracy, но и worst-case. "
        "Если worst-case сильно ниже средней, одного числа accuracy недостаточно.",
        "",
    ]

    return "\n".join(lines)
