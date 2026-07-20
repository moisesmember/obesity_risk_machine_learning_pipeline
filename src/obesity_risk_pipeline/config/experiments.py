"""External experiment/model selection plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Names of ablations and model families enabled for one run."""

    experiments: tuple[str, ...]
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.experiments or not self.models:
            raise ValueError("experiment plan requires experiments and models")
        if len(set(self.experiments)) != len(self.experiments):
            raise ValueError("experiment plan contains duplicate experiments")
        if len(set(self.models)) != len(self.models):
            raise ValueError("experiment plan contains duplicate models")


def load_experiment_plan(path: Path) -> ExperimentPlan:
    """Load a strict JSON plan without model implementation details."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid experiment plan: {path}") from exc
    if set(payload) != {"experiments", "models"}:
        raise ValueError("experiment plan must contain only experiments and models")
    return ExperimentPlan(tuple(payload["experiments"]), tuple(payload["models"]))
