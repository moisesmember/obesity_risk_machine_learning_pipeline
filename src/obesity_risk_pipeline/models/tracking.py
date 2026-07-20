"""Optional MLflow tracking adapter; domain training remains backend-independent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.models.validation import CrossValidationReport


LOGGER = logging.getLogger(__name__)


def log_training_run(
    *,
    settings: ModelingSettings,
    run_name: str,
    selected_candidate: str,
    cv_reports: dict[str, CrossValidationReport],
    candidate_metadata: dict[str, dict[str, Any]],
    artifact_paths: list[Path],
    holdout_metrics: dict[str, Any],
    selected_parameters: dict[str, Any] | None = None,
    optimization_summary: dict[str, Any] | None = None,
    stage_durations: dict[str, float] | None = None,
    partition_rows: dict[str, int] | None = None,
) -> dict[str, str]:
    """Log a parent run and identifiable nested candidate runs when enabled."""

    if not settings.mlflow_enabled:
        LOGGER.info("MLflow tracking disabled by configuration")
        return {"status": "disabled"}
    try:
        import mlflow
    except ImportError:
        LOGGER.warning("MLflow tracking requested but the package is unavailable")
        return {"status": "unavailable", "reason": "mlflow_not_installed"}

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=run_name) as parent:
            mlflow.set_tags(
                {
                    "dataset_sha256": settings.expected_sha256,
                    "feature_set_version": settings.feature_set_version,
                    "split_type": "stratified_holdout_with_cv",
                    "selected_candidate": selected_candidate,
                }
            )
            mlflow.log_params(
                {
                    "random_state": settings.random_state,
                    "holdout_size": settings.holdout_size,
                    "cv_folds": settings.cv_folds,
                    **(partition_rows or {}),
                    **{
                        f"selected_{name}": value
                        for name, value in _safe_params(
                            selected_parameters or {}
                        ).items()
                    },
                }
            )
            mlflow.log_metrics(
                {
                    f"holdout_{name}": float(value)
                    for name, value in holdout_metrics.items()
                    if isinstance(value, (int, float)) and value is not None
                }
            )
            optimization = optimization_summary or {}
            if optimization.get("status") == "completed":
                mlflow.log_params(
                    {
                        f"optuna_best_{name}": value
                        for name, value in _safe_params(
                            optimization.get("best_params", {})
                        ).items()
                    }
                )
                if optimization.get("best_value") is not None:
                    mlflow.log_metric(
                        "optuna_best_cv_macro_f1",
                        float(optimization["best_value"]),
                    )
            mlflow.log_metrics(
                {
                    f"stage_{name}_seconds": float(value)
                    for name, value in (stage_durations or {}).items()
                }
            )
            for path in artifact_paths:
                mlflow.log_artifact(str(path), artifact_path="governed")
            for name, report in cv_reports.items():
                with mlflow.start_run(run_name=name, nested=True):
                    metadata = candidate_metadata[name]
                    mlflow.set_tags(
                        {
                            "candidate": name,
                            "model": str(metadata["model_name"]),
                            "feature_set": str(metadata["experiment_name"]),
                            "age_representation": str(metadata["age_mode"]),
                        }
                    )
                    mlflow.log_params(
                        {
                            "feature_count": len(metadata.get("features", [])),
                            "categorical_encoding": metadata.get(
                                "categorical_encoding", "unknown"
                            ),
                            **_safe_params(metadata.get("parameters", {})),
                        }
                    )
                    mlflow.log_metrics(
                        {
                            f"cv_{metric}_mean": value
                            for metric, value in report.metric_mean.items()
                        }
                    )
                    mlflow.log_metric(
                        "cv_total_fit_seconds", report.total_fit_seconds
                    )
                    mlflow.log_metrics(
                        {
                            f"cv_{metric}_std": value
                            for metric, value in report.metric_std.items()
                        }
                    )
                    mlflow.log_metrics(
                        {
                            f"cv_{metric}_{label}_mean": value
                            for metric, labels in report.per_class_mean.items()
                            for label, value in labels.items()
                        }
                    )
                    mlflow.log_metrics(
                        {
                            f"cv_{metric}_{label}_std": value
                            for metric, labels in report.per_class_std.items()
                            for label, value in labels.items()
                        }
                    )
                    for step, fold in enumerate(report.fold_reports):
                        fold_metrics = {
                            "fold_macro_f1": fold.macro_f1,
                            "fold_weighted_f1": fold.weighted_f1,
                            "fold_accuracy": fold.accuracy,
                            "fold_balanced_accuracy": fold.balanced_accuracy,
                            "fold_ordinal_mae": fold.ordinal_mae,
                            "fold_quadratic_weighted_kappa": (
                                fold.quadratic_weighted_kappa
                            ),
                        }
                        if fold.log_loss is not None:
                            fold_metrics["fold_log_loss"] = fold.log_loss
                        for metric, value in fold_metrics.items():
                            mlflow.log_metric(metric, value, step=step)
            for trial in optimization.get("trials", []):
                trial_number = int(trial["number"])
                with mlflow.start_run(
                    run_name=f"{selected_candidate}.optuna_trial_{trial_number}",
                    nested=True,
                ):
                    mlflow.set_tags(
                        {
                            "candidate": selected_candidate,
                            "run_type": "optuna_trial",
                            "trial_number": str(trial_number),
                            "trial_state": str(trial["state"]),
                        }
                    )
                    mlflow.log_params(_safe_params(trial.get("params", {})))
                    trial_metrics = {
                        f"trial_{name}": float(value)
                        for name, value in trial.get("user_attrs", {}).items()
                        if isinstance(value, (int, float)) and value is not None
                    }
                    if trial.get("value") is not None:
                        trial_metrics["trial_cv_macro_f1"] = float(trial["value"])
                    if trial_metrics:
                        mlflow.log_metrics(trial_metrics)
            result = {"status": "logged", "run_id": parent.info.run_id}
            LOGGER.info("MLflow run logged run_id=%s", parent.info.run_id)
            return result
    except Exception as exc:  # MLflow is explicitly optional in project policy.
        LOGGER.warning("Optional MLflow logging failed: %s", type(exc).__name__)
        return {"status": "failed_optional", "reason": type(exc).__name__}


def log_post_run_artifacts(
    *,
    settings: ModelingSettings,
    tracking: dict[str, str],
    artifact_paths: list[Path],
    metrics: dict[str, float] | None = None,
) -> dict[str, str]:
    """Append final audit artifacts to an already completed parent run."""

    run_id = tracking.get("run_id")
    if tracking.get("status") != "logged" or not run_id:
        return {"status": "skipped", "reason": tracking.get("status", "unknown")}
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_id=run_id):
            if metrics:
                mlflow.log_metrics(metrics)
            for path in artifact_paths:
                mlflow.log_artifact(str(path), artifact_path="governed")
        LOGGER.info("Final MLflow audit artifacts logged run_id=%s", run_id)
        return {"status": "logged", "run_id": run_id}
    except Exception as exc:  # MLflow is optional by the project policy.
        LOGGER.warning(
            "Optional final MLflow artifact logging failed: %s", type(exc).__name__
        )
        return {"status": "failed_optional", "reason": type(exc).__name__}


def _safe_params(parameters: dict[str, Any]) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for name, value in sorted(parameters.items()):
        if value is None:
            safe[name] = "None"
        elif isinstance(value, (str, int, float, bool)):
            safe[name] = value
        elif isinstance(value, (list, tuple)):
            safe[name] = ",".join(str(item) for item in value)
    return safe
