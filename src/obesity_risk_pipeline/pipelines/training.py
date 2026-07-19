"""Governed baseline training without test-driven model selection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import load_canonical_dataset, split_dataset
from obesity_risk_pipeline.models.baselines import build_baseline_candidates
from obesity_risk_pipeline.models.evaluation import EvaluationReport, evaluate_classifier


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and governed reports produced by a completed baseline run."""

    run_id: str
    selected_candidate: str
    validation_reports: dict[str, EvaluationReport]
    test_report: EvaluationReport
    run_directory: Path
    model_path: Path
    evaluation_path: Path
    manifest_path: Path


def train_baselines(settings: ModelingSettings) -> TrainingResult:
    """Select on validation, evaluate once on test, and persist the fitted winner."""

    frame = load_canonical_dataset(settings)
    partitions = split_dataset(frame, settings)
    candidates = build_baseline_candidates(settings.random_state)
    validation_reports: dict[str, EvaluationReport] = {}
    fitted_estimators: dict[str, Any] = {}

    for name, candidate in candidates.items():
        train_features = partitions.train.features.loc[:, candidate.features]
        validation_features = partitions.validation.features.loc[:, candidate.features]
        candidate.estimator.fit(train_features, partitions.train.target)
        validation_reports[name] = evaluate_classifier(
            candidate.estimator,
            validation_features,
            partitions.validation.target,
            partitions.validation.features["Gender"],
        )
        fitted_estimators[name] = candidate.estimator

    selected_name = min(
        validation_reports,
        key=lambda name: (-validation_reports[name].macro_f1, name),
    )
    selected_candidate = candidates[selected_name]
    selected_estimator = fitted_estimators[selected_name]
    test_report = evaluate_classifier(
        selected_estimator,
        partitions.test.features.loc[:, selected_candidate.features],
        partitions.test.target,
        partitions.test.features["Gender"],
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"baseline-{timestamp}-{settings.expected_sha256[:8]}-{uuid4().hex[:8]}"
    run_directory = settings.output_root / run_id
    settings.output_root.mkdir(parents=True, exist_ok=True)
    publication_directory = settings.output_root / f".{run_id}.publishing"
    publication_directory.mkdir(parents=False, exist_ok=False)
    publication_model_path = publication_directory / "model.joblib"
    publication_evaluation_path = publication_directory / "evaluation.json"
    publication_manifest_path = publication_directory / "manifest.json"

    evaluation_payload = {
        "selection_partition": "validation",
        "selection_metric": "macro_f1",
        "selected_candidate": selected_name,
        "candidate_features": {
            name: list(candidate.features) for name, candidate in candidates.items()
        },
        "validation": {
            name: report.to_dict() for name, report in validation_reports.items()
        },
        "test": test_report.to_dict(),
    }
    try:
        joblib.dump(selected_estimator, publication_model_path)
        persisted_estimator = joblib.load(publication_model_path)
        test_features = partitions.test.features.loc[:, selected_candidate.features]
        if not np.array_equal(
            selected_estimator.predict(test_features),
            persisted_estimator.predict(test_features),
        ):
            raise RuntimeError(
                "serialized model predictions differ from the fitted pipeline"
            )
        _write_json(publication_evaluation_path, evaluation_payload)
        manifest_payload = {
            "manifest_schema_version": 1,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "dataset": {
                "sha256": settings.expected_sha256,
                "path": str(settings.dataset_path),
            },
            "configuration": {
                "test_size": settings.test_size,
                "validation_size": settings.validation_size,
                "random_state": settings.random_state,
                "feature_set_version": settings.feature_set_version,
            },
            "partitions": {
                "train_rows": len(partitions.train.target),
                "validation_rows": len(partitions.validation.target),
                "test_rows": len(partitions.test.target),
            },
            "selected_candidate": selected_name,
            "serialization_parity_validated": True,
            "artifacts": {
                "model.joblib": _artifact_metadata(publication_model_path),
                "evaluation.json": _artifact_metadata(publication_evaluation_path),
            },
            "mlflow_status": "not_logged_policy_pending",
            "promotion_status": "not_requested",
        }
        _write_json(publication_manifest_path, manifest_payload)
        os.replace(publication_directory, run_directory)
    except Exception:
        shutil.rmtree(publication_directory, ignore_errors=True)
        raise

    model_path = run_directory / "model.joblib"
    evaluation_path = run_directory / "evaluation.json"
    manifest_path = run_directory / "manifest.json"

    return TrainingResult(
        run_id=run_id,
        selected_candidate=selected_name,
        validation_reports=validation_reports,
        test_report=test_report,
        run_directory=run_directory,
        model_path=model_path,
        evaluation_path=evaluation_path,
        manifest_path=manifest_path,
    )


def _artifact_metadata(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "byte_size": path.stat().st_size}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
