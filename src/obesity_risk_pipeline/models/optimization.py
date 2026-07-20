"""Optional, bounded Optuna optimization that never receives holdout data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.models.catalog import Candidate
from obesity_risk_pipeline.models.telemetry import TrainingEventRecorder
from obesity_risk_pipeline.models.validation import cross_validate_candidate


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Tuned estimator and serializable study summary."""

    estimator: BaseEstimator
    summary: dict[str, Any]


def optimize_candidate(
    candidate: Candidate,
    features: pd.DataFrame,
    target: pd.Series,
    gender: pd.Series,
    settings: ModelingSettings,
    *,
    recorder: TrainingEventRecorder | None = None,
) -> OptimizationResult:
    """Optimize mean CV Macro-F1 with a small model-specific search space."""

    if settings.optuna_trials == 0 or candidate.model_name == "dummy":
        if recorder is not None:
            recorder.record(
                "hyperparameter_optimization",
                "skipped",
                candidate=candidate.name,
                reason=(
                    "disabled" if settings.optuna_trials == 0 else "dummy_model"
                ),
            )
        return OptimizationResult(candidate.estimator, {"status": "skipped"})
    try:
        import optuna
    except ImportError:
        if recorder is not None:
            recorder.record(
                "hyperparameter_optimization",
                "unavailable",
                candidate=candidate.name,
                reason="optuna_not_installed",
            )
        return OptimizationResult(
            candidate.estimator,
            {"status": "unavailable", "reason": "optuna is not installed"},
        )

    def objective(trial: Any) -> float:
        parameters = _suggest_parameters(trial, candidate.model_name)
        estimator = clone(candidate.estimator).set_params(**parameters)
        report = cross_validate_candidate(
            estimator,
            features,
            target,
            gender,
            settings,
            candidate_name=f"{candidate.name}.optuna_trial_{trial.number}",
            recorder=recorder,
        )
        trial.set_user_attr("macro_f1_std", report.metric_std["macro_f1"])
        trial.set_user_attr("ordinal_mae", report.metric_mean["ordinal_mae"])
        trial.set_user_attr("fit_seconds", report.total_fit_seconds)
        return report.metric_mean["macro_f1"]

    if recorder is not None:
        recorder.record(
            "hyperparameter_optimization",
            "started",
            candidate=candidate.name,
            objective="cv_macro_f1_mean",
            direction="maximize",
            n_trials=settings.optuna_trials,
        )
    started = time.perf_counter()
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=settings.random_state),
            pruner=optuna.pruners.MedianPruner(),
            study_name=f"{candidate.name}-{settings.expected_sha256[:8]}",
        )
        study.optimize(objective, n_trials=settings.optuna_trials, n_jobs=1)
    except Exception as exc:
        if recorder is not None:
            recorder.record(
                "hyperparameter_optimization",
                "failed",
                candidate=candidate.name,
                duration_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
            )
        raise
    if not study.trials or study.best_trial.value is None:
        raise RuntimeError("Optuna completed without a valid trial")
    best_parameters = _parameter_paths(candidate.model_name, study.best_params)
    estimator = clone(candidate.estimator).set_params(**best_parameters)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    if recorder is not None:
        recorder.record(
            "hyperparameter_optimization",
            "completed",
            candidate=candidate.name,
            duration_seconds=time.perf_counter() - started,
            valid_trials=sum(trial.value is not None for trial in study.trials),
            best_value=study.best_value,
        )
    return OptimizationResult(
        estimator,
        {
            "status": "completed",
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": trials,
        },
    )


def _suggest_parameters(trial: Any, model_name: str) -> dict[str, Any]:
    plain: dict[str, Any]
    if model_name == "logistic_regression":
        plain = {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
    elif model_name in {"extra_trees", "random_forest"}:
        plain = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    elif model_name == "hist_gradient_boosting":
        plain = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-6, 10.0, log=True
            ),
        }
    elif model_name == "catboost":
        plain = {
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    elif model_name == "lightgbm":
        plain = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        }
    elif model_name == "xgboost":
        plain = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        }
    else:
        raise ValueError(f"model {model_name!r} has no Optuna search space")
    return _parameter_paths(model_name, plain)


def _parameter_paths(model_name: str, values: dict[str, Any]) -> dict[str, Any]:
    prefix = (
        "classifier__estimator__"
        if model_name in {"lightgbm", "xgboost"}
        else "classifier__"
    )
    return {f"{prefix}{name}": value for name, value in values.items()}
