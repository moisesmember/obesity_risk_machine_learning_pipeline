"""Deterministic, fail-closed gates for model promotion decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obesity_risk_pipeline.config.promotion import PromotionPolicy
from obesity_risk_pipeline.data.modeling import TARGET_CLASSES


REQUIRED_PROMOTION_ARTIFACTS = frozenset(
    {
        "model.joblib",
        "evaluation.json",
        "leaderboard.csv",
        "explainability.json",
        "distribution_profile.json",
        "optuna.json",
        "environment.json",
        "training_events.jsonl",
    }
)


@dataclass(frozen=True, slots=True)
class HumanApproval:
    """Traceable human authorization attached to a promotion decision."""

    approved_by: str
    approval_ticket: str

    def __post_init__(self) -> None:
        for name, value in (
            ("approved_by", self.approved_by),
            ("approval_ticket", self.approval_ticket),
        ):
            normalized = value.strip()
            if not normalized or len(normalized) > 160:
                raise ValueError(f"{name} must contain between 1 and 160 characters")
            if any(ord(character) < 32 for character in normalized):
                raise ValueError(f"{name} must not contain control characters")


@dataclass(frozen=True, slots=True)
class GateResult:
    """One auditable comparison contributing to the final decision."""

    name: str
    passed: bool
    actual: Any
    operator: str
    required: Any
    detail: str


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    """Pure evaluation result before any registry mutation occurs."""

    decision: str
    source_run_id: str
    selected_candidate: str
    policy_version: str
    policy_sha256: str
    model_name: str
    gates: tuple[GateResult, ...]
    approval: HumanApproval | None
    mlflow_run_id: str | None

    @property
    def technically_approved(self) -> bool:
        return all(
            gate.passed for gate in self.gates if gate.name != "human_approval"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "source_run_id": self.source_run_id,
            "selected_candidate": self.selected_candidate,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "model_name": self.model_name,
            "technically_approved": self.technically_approved,
            "gates": [asdict(gate) for gate in self.gates],
            "approval": asdict(self.approval) if self.approval else None,
            "mlflow_run_id": self.mlflow_run_id,
        }


def evaluate_run_for_promotion(
    run_directory: Path,
    policy: PromotionPolicy,
    approval: HumanApproval | None = None,
) -> PromotionEvaluation:
    """Evaluate immutable run artifacts without modifying MLflow or local state."""

    root = run_directory.expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    evaluation = _load_json(root / "evaluation.json")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("model manifest does not contain a run_id")
    source_run_id = str(manifest["run_id"])
    selected_candidate = str(evaluation.get("selected_candidate", ""))
    if not selected_candidate:
        raise ValueError("evaluation does not identify the selected candidate")
    selected = evaluation.get("candidates", {}).get(selected_candidate)
    if not isinstance(selected, dict) or not isinstance(selected.get("cv"), dict):
        raise ValueError("selected candidate CV report is absent from evaluation")
    holdout = evaluation.get("holdout")
    if not isinstance(holdout, dict):
        raise ValueError("holdout report is absent from evaluation")

    thresholds = policy.thresholds
    gates = [
        _minimum_gate(
            "holdout_macro_f1",
            holdout.get("macro_f1"),
            thresholds.min_holdout_macro_f1,
        ),
        _minimum_gate(
            "holdout_balanced_accuracy",
            holdout.get("balanced_accuracy"),
            thresholds.min_holdout_balanced_accuracy,
        ),
        _maximum_gate(
            "holdout_ordinal_mae",
            holdout.get("ordinal_mae"),
            thresholds.max_holdout_ordinal_mae,
        ),
        _minimum_mapping_gate(
            "holdout_recall_per_class",
            holdout.get("recall_by_class"),
            thresholds.min_recall_per_class,
            expected_keys=TARGET_CLASSES,
        ),
        _maximum_gate(
            "cv_macro_f1_std",
            selected["cv"].get("metric_std", {}).get("macro_f1"),
            thresholds.max_cv_macro_f1_std,
        ),
        _maximum_gate(
            "gender_macro_f1_gap",
            _gender_macro_f1_gap(holdout.get("metrics_by_gender")),
            thresholds.max_gender_macro_f1_gap,
        ),
        GateResult(
            "artifact_integrity",
            *_artifact_integrity(root, manifest),
        ),
        GateResult(
            "serialization_parity",
            bool(manifest.get("serialization_parity_validated")),
            bool(manifest.get("serialization_parity_validated")),
            "is",
            True,
            "serialized and in-memory model predictions must match",
        ),
    ]
    if policy.require_mlflow_logged:
        mlflow_status = manifest.get("mlflow", {}).get("status")
        gates.append(
            GateResult(
                "mlflow_tracking",
                mlflow_status == "logged",
                mlflow_status,
                "==",
                "logged",
                "the source training run must be present in MLflow",
            )
        )
    if policy.require_no_failed_candidates:
        failed = evaluation.get("failed_candidates", {})
        gates.append(
            GateResult(
                "failed_candidates",
                isinstance(failed, dict) and not failed,
                sorted(failed) if isinstance(failed, dict) else "invalid",
                "==",
                [],
                "required candidates must not fail silently",
            )
        )
    if policy.require_no_unavailable_candidates:
        unavailable = evaluation.get("unavailable_candidates", {})
        gates.append(
            GateResult(
                "unavailable_candidates",
                isinstance(unavailable, dict) and not unavailable,
                sorted(unavailable) if isinstance(unavailable, dict) else "invalid",
                "==",
                [],
                "all policy-required model backends must be available",
            )
        )
    technical_passed = all(gate.passed for gate in gates)
    if policy.require_human_approval:
        gates.append(
            GateResult(
                "human_approval",
                approval is not None,
                asdict(approval) if approval else None,
                "is not",
                None,
                "a named approver and approval ticket are required",
            )
        )
    if not technical_passed:
        decision = "rejected"
    elif policy.require_human_approval and approval is None:
        decision = "pending_approval"
    else:
        decision = "approved"
    tracking = manifest.get("mlflow", {})
    return PromotionEvaluation(
        decision=decision,
        source_run_id=source_run_id,
        selected_candidate=selected_candidate,
        policy_version=policy.policy_version,
        policy_sha256=_policy_hash(policy),
        model_name=policy.model_name,
        gates=tuple(gates),
        approval=approval,
        mlflow_run_id=(
            str(tracking["run_id"])
            if isinstance(tracking, dict) and tracking.get("run_id")
            else None
        ),
    )


def _minimum_gate(name: str, actual: Any, required: float) -> GateResult:
    numeric = _finite_float(actual)
    return GateResult(
        name,
        numeric is not None and numeric >= required,
        numeric,
        ">=",
        required,
        f"{name} must satisfy the business-owned minimum",
    )


def _maximum_gate(name: str, actual: Any, required: float) -> GateResult:
    numeric = _finite_float(actual)
    return GateResult(
        name,
        numeric is not None and numeric <= required,
        numeric,
        "<=",
        required,
        f"{name} must satisfy the business-owned maximum",
    )


def _minimum_mapping_gate(
    name: str,
    actual: Any,
    required: float,
    *,
    expected_keys: tuple[str, ...],
) -> GateResult:
    if not isinstance(actual, dict) or not actual:
        return GateResult(
            name,
            False,
            actual,
            ">= for every class",
            required,
            "missing class metrics",
        )
    normalized = {
        label: _finite_float(actual.get(label)) for label in expected_keys
    }
    unknown = sorted(set(map(str, actual)) - set(expected_keys))
    passed = not unknown and all(
        value is not None and value >= required for value in normalized.values()
    )
    return GateResult(
        name,
        passed,
        normalized,
        ">= for every class",
        required,
        "every governed target class must meet the recall floor",
    )


def _gender_macro_f1_gap(metrics: Any) -> float | None:
    if not isinstance(metrics, dict) or len(metrics) < 2:
        return None
    values = [
        _finite_float(group.get("macro_f1"))
        for group in metrics.values()
        if isinstance(group, dict)
    ]
    finite = [value for value in values if value is not None]
    return max(finite) - min(finite) if len(finite) >= 2 else None


def _artifact_integrity(
    root: Path, manifest: dict[str, Any]
) -> tuple[bool, Any, str, Any, str]:
    artifacts = manifest.get("artifacts")
    failures: list[str] = []
    if not isinstance(artifacts, dict) or not artifacts:
        failures.append("manifest.artifacts")
    else:
        failures.extend(sorted(REQUIRED_PROMOTION_ARTIFACTS - set(artifacts)))
        for name, metadata in artifacts.items():
            path = (root / str(name)).resolve()
            if (
                not path.is_relative_to(root)
                or not isinstance(metadata, dict)
                or not path.is_file()
            ):
                failures.append(str(name))
                continue
            payload = path.read_bytes()
            if (
                hashlib.sha256(payload).hexdigest() != metadata.get("sha256")
                or len(payload) != metadata.get("byte_size")
            ):
                failures.append(str(name))
    return (
        not failures,
        sorted(set(failures)),
        "==",
        [],
        "all artifacts declared by the immutable manifest must match size and SHA-256",
    )


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _policy_hash(policy: PromotionPolicy) -> str:
    payload = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"required promotion artifact does not exist: {path.name}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"required promotion artifact is invalid JSON: {path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"required promotion artifact must be an object: {path.name}")
    return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
