"""Typed, fail-closed promotion policy loaded from an explicit JSON file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Business-owned numeric limits required for an automated gate decision."""

    min_holdout_macro_f1: float
    min_holdout_balanced_accuracy: float
    min_recall_per_class: float
    max_holdout_ordinal_mae: float
    max_cv_macro_f1_std: float
    max_gender_macro_f1_gap: float

    def __post_init__(self) -> None:
        for name in (
            "min_holdout_macro_f1",
            "min_holdout_balanced_accuracy",
            "min_recall_per_class",
            "max_cv_macro_f1_std",
            "max_gender_macro_f1_gap",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"promotion threshold {name} must be between 0 and 1")
        if not 0.0 <= float(self.max_holdout_ordinal_mae) <= 6.0:
            raise ValueError(
                "promotion threshold max_holdout_ordinal_mae must be between 0 and 6"
            )


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Complete policy required to evaluate and optionally register one run."""

    policy_version: str
    model_name: str
    thresholds: PromotionThresholds
    require_human_approval: bool = True
    require_mlflow_logged: bool = True
    require_no_failed_candidates: bool = True
    require_no_unavailable_candidates: bool = False
    allowed_aliases: tuple[str, ...] = ("candidate", "champion")

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("promotion policy_version must not be empty")
        if not self.model_name.strip():
            raise ValueError("promotion model_name must not be empty")
        if not self.allowed_aliases:
            raise ValueError("promotion allowed_aliases must not be empty")
        for alias in self.allowed_aliases:
            if not alias or not alias.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"invalid MLflow model alias: {alias!r}")


def load_promotion_policy(path: Path) -> PromotionPolicy:
    """Load a promotion policy and reject missing or unknown configuration fields."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"promotion policy does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"promotion policy is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("promotion policy root must be a JSON object")
    required = {"policy_version", "model_name", "thresholds"}
    optional = {
        "require_human_approval",
        "require_mlflow_logged",
        "require_no_failed_candidates",
        "require_no_unavailable_candidates",
        "allowed_aliases",
    }
    _validate_keys(payload, required, optional, "promotion policy")
    raw_thresholds = payload["thresholds"]
    if not isinstance(raw_thresholds, dict):
        raise ValueError("promotion policy thresholds must be a JSON object")
    threshold_fields = {
        "min_holdout_macro_f1",
        "min_holdout_balanced_accuracy",
        "min_recall_per_class",
        "max_holdout_ordinal_mae",
        "max_cv_macro_f1_std",
        "max_gender_macro_f1_gap",
    }
    _validate_keys(
        raw_thresholds,
        threshold_fields,
        set(),
        "promotion policy thresholds",
    )
    try:
        thresholds = PromotionThresholds(
            **{name: float(raw_thresholds[name]) for name in threshold_fields}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid promotion thresholds: {exc}") from exc
    aliases = payload.get("allowed_aliases", ["candidate", "champion"])
    if not isinstance(aliases, list) or not all(
        isinstance(value, str) for value in aliases
    ):
        raise ValueError("promotion allowed_aliases must be a JSON string array")
    return PromotionPolicy(
        policy_version=str(payload["policy_version"]),
        model_name=str(payload["model_name"]),
        thresholds=thresholds,
        require_human_approval=_boolean(payload, "require_human_approval", True),
        require_mlflow_logged=_boolean(payload, "require_mlflow_logged", True),
        require_no_failed_candidates=_boolean(
            payload, "require_no_failed_candidates", True
        ),
        require_no_unavailable_candidates=_boolean(
            payload, "require_no_unavailable_candidates", False
        ),
        allowed_aliases=tuple(aliases),
    )


def _validate_keys(
    payload: dict[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{context} fields are invalid; missing={missing!r}, unknown={unknown!r}"
        )


def _boolean(payload: dict[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"promotion field {name} must be boolean")
    return value
