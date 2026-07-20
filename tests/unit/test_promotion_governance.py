from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from obesity_risk_pipeline.config.promotion import (
    PromotionPolicy,
    PromotionThresholds,
    load_promotion_policy,
)
from obesity_risk_pipeline.data.modeling import TARGET_CLASSES
from obesity_risk_pipeline.models.governance import (
    HumanApproval,
    evaluate_run_for_promotion,
)
from obesity_risk_pipeline.pipelines.promotion import promote_run


def _policy() -> PromotionPolicy:
    return PromotionPolicy(
        policy_version="test-v1",
        model_name="obesity-risk-test",
        thresholds=PromotionThresholds(
            min_holdout_macro_f1=0.80,
            min_holdout_balanced_accuracy=0.80,
            min_recall_per_class=0.70,
            max_holdout_ordinal_mae=0.30,
            max_cv_macro_f1_std=0.05,
            max_gender_macro_f1_gap=0.10,
        ),
    )


def _run_directory(tmp_path: Path, *, class_recall: float = 0.82) -> Path:
    root = tmp_path / "run-1"
    root.mkdir()
    artifact_names = (
        "model.joblib",
        "leaderboard.csv",
        "explainability.json",
        "distribution_profile.json",
        "optuna.json",
        "environment.json",
        "training_events.jsonl",
    )
    for name in artifact_names:
        (root / name).write_bytes(f"test-{name}".encode())
    evaluation = {
        "selected_candidate": "A_full__logistic_regression",
        "candidates": {
            "A_full__logistic_regression": {
                "cv": {"metric_std": {"macro_f1": 0.02}}
            }
        },
        "holdout": {
            "macro_f1": 0.86,
            "balanced_accuracy": 0.85,
            "ordinal_mae": 0.14,
            "recall_by_class": {
                label: class_recall if index == 0 else 0.84
                for index, label in enumerate(TARGET_CLASSES)
            },
            "metrics_by_gender": {
                "Female": {"macro_f1": 0.80},
                "Male": {"macro_f1": 0.82},
            },
        },
        "failed_candidates": {},
        "unavailable_candidates": {},
    }
    evaluation_path = root / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    artifacts = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_size": path.stat().st_size,
        }
        for path in (*(root / name for name in artifact_names), evaluation_path)
    }
    manifest = {
        "run_id": "run-1",
        "selected_candidate": "A_full__logistic_regression",
        "serialization_parity_validated": True,
        "mlflow": {"status": "logged", "run_id": "mlflow-run-1"},
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_gate_passes_technically_but_waits_for_human_approval(tmp_path: Path) -> None:
    evaluation = evaluate_run_for_promotion(_run_directory(tmp_path), _policy())

    assert evaluation.technically_approved is True
    assert evaluation.decision == "pending_approval"
    assert next(
        gate for gate in evaluation.gates if gate.name == "human_approval"
    ).passed is False


def test_gate_approves_only_with_metrics_integrity_tracking_and_approval(
    tmp_path: Path,
) -> None:
    evaluation = evaluate_run_for_promotion(
        _run_directory(tmp_path),
        _policy(),
        HumanApproval("ml-owner@example.com", "CHANGE-123"),
    )

    assert evaluation.decision == "approved"
    assert all(gate.passed for gate in evaluation.gates)


def test_gate_rejects_any_class_below_the_recall_floor(tmp_path: Path) -> None:
    evaluation = evaluate_run_for_promotion(
        _run_directory(tmp_path, class_recall=0.60),
        _policy(),
        HumanApproval("owner", "CHANGE-123"),
    )

    assert evaluation.decision == "rejected"
    recall_gate = next(
        gate for gate in evaluation.gates if gate.name == "holdout_recall_per_class"
    )
    assert recall_gate.passed is False


def test_gate_rejects_missing_target_class_metric(tmp_path: Path) -> None:
    root = _run_directory(tmp_path)
    evaluation_path = root / "evaluation.json"
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    payload["holdout"]["recall_by_class"].pop(TARGET_CLASSES[-1])
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation.json"] = {
        "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "byte_size": evaluation_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evaluation = evaluate_run_for_promotion(
        root,
        _policy(),
        HumanApproval("owner", "CHANGE-123"),
    )

    recall_gate = next(
        gate for gate in evaluation.gates if gate.name == "holdout_recall_per_class"
    )
    assert evaluation.decision == "rejected"
    assert recall_gate.passed is False
    assert recall_gate.actual[TARGET_CLASSES[-1]] is None


def test_gate_rejects_manifest_that_omits_a_required_artifact(
    tmp_path: Path,
) -> None:
    root = _run_directory(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].pop("model.joblib")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evaluation = evaluate_run_for_promotion(
        root,
        _policy(),
        HumanApproval("owner", "CHANGE-123"),
    )

    integrity = next(
        gate for gate in evaluation.gates if gate.name == "artifact_integrity"
    )
    assert evaluation.decision == "rejected"
    assert integrity.passed is False
    assert "model.joblib" in integrity.actual


def test_gate_rejects_artifact_tampering(tmp_path: Path) -> None:
    root = _run_directory(tmp_path)
    (root / "model.joblib").write_bytes(b"tampered")

    evaluation = evaluate_run_for_promotion(
        root,
        _policy(),
        HumanApproval("owner", "CHANGE-123"),
    )

    assert evaluation.decision == "rejected"
    integrity = next(
        gate for gate in evaluation.gates if gate.name == "artifact_integrity"
    )
    assert integrity.passed is False
    assert integrity.actual == ["model.joblib"]


def test_rejected_registration_request_publishes_report_without_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_registry_call(**_: object) -> None:
        raise AssertionError("registry must not be called for a rejected run")

    monkeypatch.setattr(
        "obesity_risk_pipeline.pipelines.promotion.register_approved_model",
        forbidden_registry_call,
    )
    root = _run_directory(tmp_path, class_recall=0.60)

    first = promote_run(
        run_directory=root,
        policy=_policy(),
        output_root=tmp_path / "promotions",
        approval=HumanApproval("owner", "CHANGE-123"),
        register=True,
    )
    second = promote_run(
        run_directory=root,
        policy=_policy(),
        output_root=tmp_path / "promotions",
        approval=HumanApproval("owner", "CHANGE-123"),
        register=True,
    )

    assert first.report_path == second.report_path
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["evaluation"]["decision"] == "rejected"
    assert report["registry"]["status"] == "blocked_by_gates"


def test_policy_loader_rejects_missing_business_thresholds(tmp_path: Path) -> None:
    policy_path = tmp_path / "promotion.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_version": "v1",
                "model_name": "model",
                "thresholds": {"min_holdout_macro_f1": 0.8},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields are invalid"):
        load_promotion_policy(policy_path)


def test_repository_experimental_promotion_policy_is_complete() -> None:
    policy = load_promotion_policy(Path("configs/promotion.json"))

    assert policy.policy_version == "experimental-v1"
    assert policy.model_name == "obesity-risk-multiclass"
    assert policy.require_human_approval is True
    assert policy.require_mlflow_logged is True
    assert policy.require_no_failed_candidates is True
    assert policy.thresholds.min_recall_per_class == 0.68
