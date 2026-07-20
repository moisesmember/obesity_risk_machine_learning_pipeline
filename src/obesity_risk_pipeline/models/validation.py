"""Manual stratified CV that keeps every fitted transformation inside a fold."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import build_cross_validator
from obesity_risk_pipeline.data.modeling import TARGET_CLASSES
from obesity_risk_pipeline.models.evaluation import (
    EvaluationReport,
    evaluate_classifier,
)
from obesity_risk_pipeline.models.telemetry import TrainingEventRecorder


@dataclass(frozen=True, slots=True)
class CrossValidationReport:
    """Fold reports plus mean/std values used by governed selection."""

    fold_reports: tuple[EvaluationReport, ...]
    fold_fit_seconds: tuple[float, ...]
    metric_mean: dict[str, float]
    metric_std: dict[str, float]
    per_class_mean: dict[str, dict[str, float]]
    per_class_std: dict[str, dict[str, float]]
    total_fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cross_validate_candidate(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    gender: pd.Series,
    settings: ModelingSettings,
    *,
    candidate_name: str = "candidate",
    recorder: TrainingEventRecorder | None = None,
) -> CrossValidationReport:
    """Fit fresh clones in each development fold and aggregate governed metrics."""

    fold_reports: list[EvaluationReport] = []
    fit_seconds: list[float] = []
    splitter = build_cross_validator(settings)
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features, target), start=1
    ):
        if recorder is not None:
            recorder.record(
                "cross_validation_fold",
                "started",
                candidate=candidate_name,
                fold=fold_number,
                train_rows=len(train_indices),
                validation_rows=len(validation_indices),
            )
        fold_started = time.perf_counter()
        try:
            fold_estimator = clone(estimator)
            fit_started = time.perf_counter()
            fold_estimator.fit(
                features.iloc[train_indices], target.iloc[train_indices]
            )
            fit_duration = time.perf_counter() - fit_started
            fit_seconds.append(fit_duration)
            evaluation_started = time.perf_counter()
            report = evaluate_classifier(
                fold_estimator,
                features.iloc[validation_indices],
                target.iloc[validation_indices],
                gender.iloc[validation_indices],
            )
            evaluation_duration = time.perf_counter() - evaluation_started
        except Exception as exc:
            if recorder is not None:
                recorder.record(
                    "cross_validation_fold",
                    "failed",
                    candidate=candidate_name,
                    fold=fold_number,
                    duration_seconds=time.perf_counter() - fold_started,
                    error_type=type(exc).__name__,
                )
            raise
        fold_reports.append(report)
        if recorder is not None:
            recorder.record(
                "cross_validation_fold",
                "completed",
                candidate=candidate_name,
                fold=fold_number,
                fit_seconds=fit_duration,
                evaluation_seconds=evaluation_duration,
                macro_f1=report.macro_f1,
                weighted_f1=report.weighted_f1,
                balanced_accuracy=report.balanced_accuracy,
                ordinal_mae=report.ordinal_mae,
                quadratic_weighted_kappa=report.quadratic_weighted_kappa,
            )

    metric_names = (
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "accuracy",
        "log_loss",
        "ordinal_mae",
        "quadratic_weighted_kappa",
    )
    means: dict[str, float] = {}
    deviations: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray(
            [getattr(report, name) for report in fold_reports], dtype=float
        )
        means[name] = float(np.nanmean(values))
        deviations[name] = float(np.nanstd(values, ddof=0))
    per_class_mean: dict[str, dict[str, float]] = {}
    per_class_std: dict[str, dict[str, float]] = {}
    for metric_name in ("precision_by_class", "recall_by_class", "f1_by_class"):
        mean_values: dict[str, float] = {}
        std_values: dict[str, float] = {}
        for label in TARGET_CLASSES:
            values = np.asarray(
                [getattr(report, metric_name)[label] for report in fold_reports]
            )
            mean_values[label] = float(np.mean(values))
            std_values[label] = float(np.std(values, ddof=0))
        per_class_mean[metric_name] = mean_values
        per_class_std[metric_name] = std_values
    return CrossValidationReport(
        tuple(fold_reports),
        tuple(fit_seconds),
        means,
        deviations,
        per_class_mean,
        per_class_std,
        float(sum(fit_seconds)),
    )
