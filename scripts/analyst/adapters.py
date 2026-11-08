from __future__ import annotations
from .llm_judge import check_answer_is_correct

import json
import uuid
from pathlib import Path
from typing import Any

from .schemas import RunLog


def load_expected_outputs(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["task_id"]] = row.get("expected_output", "")
    return mapping


def load_mutations_meta(path: str | Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id not in meta:
                meta[task_id] = {
                    "mutation_type": row.get("mutation_type", "unknown"),
                    "intensity": row.get("intensity", "unknown"),
                }
    return meta


def normalize_correct(output: str, expected: str) -> bool:
    # Вместо жесткого сравнения строк, используем GigaChat
    return check_answer_is_correct(output, expected)


def adapt_robustness_report(
    report_path: str | Path,
    enriched_path: str | Path,
    mutations_path: str | Path,
    agent_name: str = "langgraph",
) -> list[RunLog]:
    expected = load_expected_outputs(enriched_path)
    mutations_meta = load_mutations_meta(mutations_path)

    with open(report_path, encoding="utf-8") as f:
        rows = json.load(f)

    logs: list[RunLog] = []
    for row in rows:
        task_id = row["task_id"]
        prompt_type = row.get("prompt_type", "clean")
        expected_output = expected.get(task_id, "")
        output = row.get("model_output", "")

        if prompt_type == "clean":
            mutation_type = "clean"
            intensity = "none"
        else:
            meta = mutations_meta.get(task_id, {})
            mutation_type = meta.get("mutation_type", "unknown")
            intensity = meta.get("intensity", "unknown")

        tokens = row.get("tokens", {})
        trajectory = row.get("trajectory", [])
        steps = row.get("steps_count", len(trajectory))

        logs.append(
            RunLog(
                run_id=row.get("mutation_id") or str(uuid.uuid4()),
                mutation_id=row.get("mutation_id", ""),
                task_id=task_id,
                agent=agent_name,
                mutation_type=mutation_type,
                intensity=intensity,
                input=row.get("prompt_text", ""),
                output=output,
                expected_output=expected_output,
                correct=normalize_correct(output, expected_output),
                steps=steps,
                trajectory=trajectory,
                tokens_in=tokens.get("input_tokens", 0),
                tokens_out=tokens.get("output_tokens", 0),
                cost_usd=row.get("cost_usd", 0.0),
                latency_ms=row.get("latency_ms", 0),
            )
        )
    return logs
