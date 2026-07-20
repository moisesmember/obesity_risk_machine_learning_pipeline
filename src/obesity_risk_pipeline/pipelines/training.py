"""Governed cross-validation, selection, final holdout and atomic publication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import (
    MODEL_FEATURES,
    TARGET_CLASSES,
    load_canonical_dataset,
    split_dataset,
)
from obesity_risk_pipeline.data.validation import (
    build_distribution_profile,
    validate_canonical_frame,
)
from obesity_risk_pipeline.features.experiments import build_experiment_catalog
from obesity_risk_pipeline.models.catalog import (
    DEFAULT_MODEL_NAMES,
    Candidate,
    build_candidate_catalog,
)
from obesity_risk_pipeline.models.evaluation import (
    EvaluationReport,
    evaluate_classifier,
    predict_probabilities,
)
from obesity_risk_pipeline.models.explainability import (
    build_explainability_report,
    write_explainability_plot,
)
from obesity_risk_pipeline.models.optimization import optimize_candidate
from obesity_risk_pipeline.models.telemetry import TrainingEventRecorder
from obesity_risk_pipeline.models.tracking import (
    log_post_run_artifacts,
    log_training_run,
)
from obesity_risk_pipeline.models.validation import (
    CrossValidationReport,
    cross_validate_candidate,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and reports produced by a complete governed experiment run."""

    run_id: str
    selected_candidate: str
    cv_reports: dict[str, CrossValidationReport]
    holdout_report: EvaluationReport
    unavailable_candidates: dict[str, str]
    failed_candidates: dict[str, str]
    run_directory: Path
    model_path: Path
    evaluation_path: Path
    leaderboard_path: Path
    predictions_path: Path
    manifest_path: Path
    training_events_path: Path

    @property
    def test_report(self) -> EvaluationReport:
        """Compatibility alias; the governed name is holdout_report."""

        return self.holdout_report


def run_experiments(
    settings: ModelingSettings,
    *,
    experiment_names: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
) -> TrainingResult:
    """Select entirely by development CV and evaluate the winner once on holdout."""
    run_id = _run_id(settings)
    recorder = TrainingEventRecorder(run_id, LOGGER)
    recorder.record(
        "training_run",
        "started",
        dataset_sha256=settings.expected_sha256,
        cv_folds=settings.cv_folds,
        holdout_size=settings.holdout_size,
        random_state=settings.random_state,
    )

    with recorder.stage("dataset_loading") as stage:
        frame = load_canonical_dataset(settings)
        stage["rows"] = len(frame)
        stage["columns"] = len(frame.columns)
    with recorder.stage("data_contract_validation"):
        validate_canonical_frame(frame, settings)
    with recorder.stage("dataset_split") as stage:
        partitions = split_dataset(frame, settings)
        stage["development_rows"] = len(partitions.development.target)
        stage["holdout_rows"] = len(partitions.holdout.target)

    with recorder.stage("candidate_catalog") as stage:
        experiment_catalog = build_experiment_catalog()
        selected_experiment_names = tuple(
            experiment_names or experiment_catalog.keys()
        )
        unknown_experiments = set(selected_experiment_names) - set(
            experiment_catalog
        )
        if unknown_experiments:
            raise ValueError(
                f"unknown experiments: {sorted(unknown_experiments)!r}"
            )
        experiments = [
            experiment_catalog[name] for name in selected_experiment_names
        ]
        candidate_catalog = build_candidate_catalog(
            experiments,
            settings.random_state,
            tuple(model_names or DEFAULT_MODEL_NAMES),
        )
        if not candidate_catalog.available:
            raise RuntimeError("no model candidate is available")
        stage["available_candidates"] = len(candidate_catalog.available)
        stage["unavailable_candidates"] = len(candidate_catalog.unavailable)
    for name, reason in candidate_catalog.unavailable.items():
        recorder.record(
            "candidate_availability", "unavailable", candidate=name, reason=reason
        )

    cv_reports: dict[str, CrossValidationReport] = {}
    failed_candidates: dict[str, str] = {}
    with recorder.stage(
        "cross_validation", candidates=len(candidate_catalog.available)
    ) as stage:
        for name, candidate in candidate_catalog.available.items():
            recorder.record("candidate_validation", "started", candidate=name)
            try:
                report = cross_validate_candidate(
                    candidate.estimator,
                    partitions.development.features,
                    partitions.development.target,
                    partitions.development.features["Gender"],
                    settings,
                    candidate_name=name,
                    recorder=recorder,
                )
                cv_reports[name] = report
                recorder.record(
                    "candidate_validation",
                    "completed",
                    candidate=name,
                    macro_f1_mean=report.metric_mean["macro_f1"],
                    macro_f1_std=report.metric_std["macro_f1"],
                    ordinal_mae_mean=report.metric_mean["ordinal_mae"],
                    total_fit_seconds=report.total_fit_seconds,
                )
            except Exception as exc:
                failed_candidates[name] = f"{type(exc).__name__}: {exc}"
                recorder.record(
                    "candidate_validation",
                    "failed",
                    candidate=name,
                    error_type=type(exc).__name__,
                )
        if not cv_reports:
            raise RuntimeError(f"all model candidates failed: {failed_candidates!r}")
        stage["successful_candidates"] = len(cv_reports)
        stage["failed_candidates"] = len(failed_candidates)

    selected_name = min(
        cv_reports,
        key=lambda name: _selection_key(name, cv_reports[name]),
    )
    selected_candidate = candidate_catalog.available[selected_name]
    recorder.record(
        "candidate_selection",
        "completed",
        candidate=selected_name,
        objective="macro_f1_mean",
        macro_f1_mean=cv_reports[selected_name].metric_mean["macro_f1"],
    )
    with recorder.stage("optimization", candidate=selected_name) as stage:
        optimization = optimize_candidate(
            selected_candidate,
            partitions.development.features,
            partitions.development.target,
            partitions.development.features["Gender"],
            settings,
            recorder=recorder,
        )
        stage["optimization_status"] = optimization.summary.get(
            "status", "unknown"
        )
    selected_estimator = optimization.estimator
    with recorder.stage("final_fit", candidate=selected_name):
        selected_estimator.fit(
            partitions.development.features, partitions.development.target
        )
    with recorder.stage("holdout_evaluation", candidate=selected_name) as stage:
        holdout_report = evaluate_classifier(
            selected_estimator,
            partitions.holdout.features,
            partitions.holdout.target,
            partitions.holdout.features["Gender"],
        )
        probabilities = predict_probabilities(
            selected_estimator, partitions.holdout.features
        )
        predictions = _prediction_frame(
            selected_estimator,
            partitions.holdout.identifiers,
            partitions.holdout.features,
            partitions.holdout.target,
            probabilities,
        )
        stage["rows"] = holdout_report.row_count
        stage["macro_f1"] = holdout_report.macro_f1
        stage["weighted_f1"] = holdout_report.weighted_f1
        stage["balanced_accuracy"] = holdout_report.balanced_accuracy
        stage["ordinal_mae"] = holdout_report.ordinal_mae
    with recorder.stage("explainability"):
        explainability = build_explainability_report(
            selected_estimator,
            partitions.holdout.features,
            partitions.holdout.target,
            random_state=settings.random_state,
        )
        profile = build_distribution_profile(partitions.development.features)

    publication = settings.output_root / f".{run_id}.publishing"
    run_directory = settings.output_root / run_id
    settings.output_root.mkdir(parents=True, exist_ok=True)
    publication.mkdir(exist_ok=False)
    paths = {
        "model.joblib": publication / "model.joblib",
        "evaluation.json": publication / "evaluation.json",
        "leaderboard.csv": publication / "leaderboard.csv",
        "predictions.csv": publication / "predictions.csv",
        "explainability.json": publication / "explainability.json",
        "explainability.png": publication / "explainability.png",
        "distribution_profile.json": publication / "distribution_profile.json",
        "optuna.json": publication / "optuna.json",
        "environment.json": publication / "environment.json",
        "training_events.jsonl": publication / "training_events.jsonl",
        "manifest.json": publication / "manifest.json",
    }
    metadata = {
        name: {
            "model_name": candidate.model_name,
            "experiment_name": candidate.experiment.name,
            "age_mode": candidate.experiment.age_mode,
            "categorical_encoding": candidate.experiment.categorical_encoding,
            "features": list(candidate.experiment.output_features),
            "proxy_note": candidate.experiment.proxy_note,
            "parameters": _estimator_parameters(candidate.estimator),
        }
        for name, candidate in candidate_catalog.available.items()
        if name in cv_reports
    }
    evaluation_payload = {
        "selection_partition": "development_cross_validation",
        "selection_metric": "macro_f1_mean",
        "selected_candidate": selected_name,
        "candidates": {
            name: {"metadata": metadata[name], "cv": report.to_dict()}
            for name, report in cv_reports.items()
        },
        "unavailable_candidates": dict(candidate_catalog.unavailable),
        "failed_candidates": failed_candidates,
        "holdout": holdout_report.to_dict(),
    }
    try:
        with recorder.stage("artifact_generation"):
            joblib.dump(selected_estimator, paths["model.joblib"])
            persisted = joblib.load(paths["model.joblib"])
            if not np.array_equal(
                selected_estimator.predict(partitions.holdout.features),
                persisted.predict(partitions.holdout.features),
            ):
                raise RuntimeError("serialized model predictions differ from memory")
            _write_json(paths["evaluation.json"], evaluation_payload)
            _leaderboard(cv_reports, metadata).to_csv(
                paths["leaderboard.csv"], index=False
            )
            predictions.to_csv(paths["predictions.csv"], index=False)
            _write_json(paths["explainability.json"], explainability)
            write_explainability_plot(explainability, paths["explainability.png"])
            _write_json(paths["distribution_profile.json"], profile.to_dict())
            _write_json(paths["optuna.json"], optimization.summary)
            _write_json(paths["environment.json"], _environment_payload())
        with recorder.stage("mlflow_tracking") as stage:
            tracking = log_training_run(
                settings=settings,
                run_name=run_id,
                selected_candidate=selected_name,
                cv_reports=cv_reports,
                candidate_metadata=metadata,
                artifact_paths=[
                    paths["evaluation.json"],
                    paths["leaderboard.csv"],
                    paths["explainability.json"],
                    paths["explainability.png"],
                    paths["distribution_profile.json"],
                    paths["optuna.json"],
                    paths["environment.json"],
                    paths["model.joblib"],
                ],
                holdout_metrics=holdout_report.to_dict(),
                selected_parameters=_estimator_parameters(selected_estimator),
                optimization_summary=optimization.summary,
                stage_durations=recorder.stage_durations,
                partition_rows={
                    "development_rows": len(partitions.development.target),
                    "holdout_rows": len(partitions.holdout.target),
                },
            )
            stage["tracking_status"] = tracking["status"]
        recorder.record(
            "privacy_boundary",
            "enforced",
            excluded_from_mlflow=["predictions.csv"],
            reason="contains row identifiers and per-record outcomes",
        )
        recorder.record("publication", "prepared", artifact_count=len(paths))
        recorder.record(
            "mlflow_final_artifacts",
            "requested" if tracking["status"] == "logged" else "skipped",
            parent_tracking_status=tracking["status"],
        )
        recorder.record(
            "training_run",
            "completed",
            selected_candidate=selected_name,
            mlflow_status=tracking["status"],
            duration_seconds=recorder.elapsed_seconds,
        )
        recorder.write_jsonl(paths["training_events.jsonl"])
        artifact_names = tuple(name for name in paths if name != "manifest.json")
        manifest = {
            "manifest_schema_version": 2,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "dataset": {"sha256": settings.expected_sha256},
            "configuration": {
                "holdout_size": settings.holdout_size,
                "cv_folds": settings.cv_folds,
                "random_state": settings.random_state,
                "feature_set_version": settings.feature_set_version,
                "distribution_psi_threshold": settings.distribution_psi_threshold,
                "numeric_bounds": {
                    name: {"minimum": bounds.minimum, "maximum": bounds.maximum}
                    for name, bounds in settings.numeric_bounds.items()
                },
            },
            "partitions": {
                "development_rows": len(partitions.development.target),
                "holdout_rows": len(partitions.holdout.target),
            },
            "input_features": list(MODEL_FEATURES),
            "target_classes": list(TARGET_CLASSES),
            "selected_candidate": selected_name,
            "selected_metadata": metadata[selected_name],
            "serialization_parity_validated": True,
            "stage_durations_seconds": recorder.stage_durations,
            "mlflow": tracking,
            "optuna": {"status": optimization.summary.get("status", "unknown")},
            "promotion_status": "not_requested",
            "artifacts": {
                name: _artifact_metadata(paths[name]) for name in artifact_names
            },
        }
        _write_json(paths["manifest.json"], manifest)
        log_post_run_artifacts(
            settings=settings,
            tracking=tracking,
            artifact_paths=[
                paths["training_events.jsonl"],
                paths["manifest.json"],
            ],
            metrics={
                **{
                    f"stage_{name}_seconds": value
                    for name, value in recorder.stage_durations.items()
                },
                "training_run_seconds": recorder.elapsed_seconds,
            },
        )
        os.replace(publication, run_directory)
    except Exception as exc:
        recorder.record(
            "publication", "failed", error_type=type(exc).__name__
        )
        shutil.rmtree(publication, ignore_errors=True)
        raise

    return TrainingResult(
        run_id,
        selected_name,
        cv_reports,
        holdout_report,
        dict(candidate_catalog.unavailable),
        failed_candidates,
        run_directory,
        run_directory / "model.joblib",
        run_directory / "evaluation.json",
        run_directory / "leaderboard.csv",
        run_directory / "predictions.csv",
        run_directory / "manifest.json",
        run_directory / "training_events.jsonl",
    )


def train_baselines(settings: ModelingSettings) -> TrainingResult:
    """Compatibility entry point for a fast, governed baseline comparison."""

    return run_experiments(
        settings,
        experiment_names=("A_full", "D_behavioral", "E_body_bmi"),
        model_names=("dummy", "logistic_regression"),
    )


def _selection_key(name: str, report: CrossValidationReport) -> tuple[Any, ...]:
    return (
        -report.metric_mean["macro_f1"],
        report.metric_std["macro_f1"],
        report.metric_mean["ordinal_mae"],
        -report.metric_mean["quadratic_weighted_kappa"],
        report.total_fit_seconds,
        name,
    )


def _leaderboard(
    reports: dict[str, CrossValidationReport],
    metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for name, report in reports.items():
        info = metadata[name]
        rows.append(
            {
                "candidate": name,
                "experiment": info["experiment_name"],
                "features": "|".join(info["features"]),
                "age_representation": info["age_mode"],
                "model": info["model_name"],
                "macro_f1_mean": report.metric_mean["macro_f1"],
                "macro_f1_std": report.metric_std["macro_f1"],
                "accuracy_mean": report.metric_mean["accuracy"],
                "log_loss_mean": report.metric_mean["log_loss"],
                "ordinal_mae_mean": report.metric_mean["ordinal_mae"],
                "fit_seconds": report.total_fit_seconds,
                "proxy_note": info["proxy_note"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "macro_f1_std", "ordinal_mae_mean", "candidate"],
        ascending=[False, True, True, True],
    )


def _prediction_frame(
    estimator: BaseEstimator,
    identifiers: pd.Series,
    features: pd.DataFrame,
    target: pd.Series,
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": identifiers.to_numpy(),
            "actual_class": target.to_numpy(),
            "predicted_class": estimator.predict(features),
        }
    )
    if probabilities is not None:
        for index, label in enumerate(TARGET_CLASSES):
            frame[f"probability_{label}"] = probabilities[:, index]
    return frame


def _run_id(settings: ModelingSettings) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"experiment-{timestamp}-{settings.expected_sha256[:8]}-{uuid4().hex[:8]}"


def _artifact_metadata(path: Path) -> dict[str, int | str]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_size": path.stat().st_size,
    }


def _estimator_parameters(estimator: BaseEstimator) -> dict[str, Any]:
    """Return only serializable classifier parameters for audit and tracking."""

    parameters: dict[str, Any] = {}
    for name, value in sorted(estimator.get_params(deep=True).items()):
        if not name.startswith("classifier__"):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            parameters[name] = value
        elif isinstance(value, (list, tuple)) and all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value
        ):
            parameters[name] = list(value)
    return parameters


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment_payload() -> dict[str, Any]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "catboost",
        "lightgbm",
        "xgboost",
        "mlflow",
        "optuna",
        "shap",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return {
        "python": platform.python_version(),
        "packages": versions,
        "code_revision": os.getenv("GITHUB_SHA", "unavailable"),
    }
