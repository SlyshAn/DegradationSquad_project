#Схемы
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class MutationRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    mutation_id: str
    task_id: str
    mutation_type: str
    intensity: str
    mutated_input: str = ""


class RunLog(BaseModel):
    """Одна запись лога одного прогона агента."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    mutation_id: str = ""
    task_id: str = ""
    agent: str

    mutation_type: str = "clean"
    intensity: str = "none"

    input: str = ""
    output: str = ""
    expected_output: str = ""

    correct: bool
    steps: int = 0
    trajectory: list[dict[str, Any]] = Field(default_factory=list)

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    model_version: str = ""
    temperature: float = 0.0
    seed: int = 0
    timestamp: str = ""
