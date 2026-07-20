"""Multiclass, ordinal and subgroup evaluation without estimator mutation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from obesity_risk_pipeline.data.modeling import TARGET_CLASSES


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """JSON-compatible statistical, ordinal and subgroup metrics."""

    row_count: int
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    accuracy: float
    log_loss: float | None
    multiclass_brier_score: float | None
    ordinal_mae: float
    quadratic_weighted_kappa: float
    precision_by_class: dict[str, float]
    recall_by_class: dict[str, float]
    f1_by_class: dict[str, float]
    confusion_matrix: list[list[int]]
    metrics_by_gender: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_classifier(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    gender: pd.Series,
) -> EvaluationReport:
    """Evaluate an already-fitted classifier without changing its state."""

    predictions = np.asarray(estimator.predict(features), dtype=object)
    probabilities = predict_probabilities(estimator, features)
    return evaluate_predictions(target, predictions, probabilities, gender)


def evaluate_predictions(
    target: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    gender: pd.Series | np.ndarray,
) -> EvaluationReport:
    """Calculate governed metrics from aligned labels and probabilities."""

    target_values = np.asarray(target, dtype=object)
    predicted_values = np.asarray(predictions, dtype=object)
    precision = precision_score(
        target_values,
        predicted_values,
        labels=TARGET_CLASSES,
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        target_values,
        predicted_values,
        labels=TARGET_CLASSES,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        target_values,
        predicted_values,
        labels=TARGET_CLASSES,
        average=None,
        zero_division=0,
    )
    group_metrics = _metrics_by_gender(
        target_values, predicted_values, np.asarray(gender, dtype=object)
    )
    log_loss_value, brier = _probability_metrics(target_values, probabilities)
    class_to_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    true_indices = np.asarray([class_to_index[str(value)] for value in target_values])
    predicted_indices = np.asarray(
        [class_to_index[str(value)] for value in predicted_values]
    )
    return EvaluationReport(
        row_count=len(target_values),
        macro_f1=float(
            f1_score(
                target_values,
                predicted_values,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                target_values,
                predicted_values,
                labels=TARGET_CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(target_values, predicted_values)
        ),
        accuracy=float(accuracy_score(target_values, predicted_values)),
        log_loss=log_loss_value,
        multiclass_brier_score=brier,
        ordinal_mae=float(np.mean(np.abs(true_indices - predicted_indices))),
        quadratic_weighted_kappa=float(
            cohen_kappa_score(true_indices, predicted_indices, weights="quadratic")
        ),
        precision_by_class=dict(zip(TARGET_CLASSES, map(float, precision))),
        recall_by_class=dict(zip(TARGET_CLASSES, map(float, recall))),
        f1_by_class=dict(zip(TARGET_CLASSES, map(float, per_class_f1))),
        confusion_matrix=confusion_matrix(
            target_values, predicted_values, labels=TARGET_CLASSES
        ).tolist(),
        metrics_by_gender=group_metrics,
    )


def _metrics_by_gender(
    target: np.ndarray,
    predictions: np.ndarray,
    gender: np.ndarray,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for value in sorted(set(map(str, gender))):
        mask = gender.astype(str) == value
        groups[value] = {
            "row_count": int(mask.sum()),
            "macro_f1": float(
                f1_score(
                    target[mask],
                    predictions[mask],
                    labels=TARGET_CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_by_class": dict(
                zip(
                    TARGET_CLASSES,
                    map(
                        float,
                        recall_score(
                            target[mask],
                            predictions[mask],
                            labels=TARGET_CLASSES,
                            average=None,
                            zero_division=0,
                        ),
                    ),
                )
            ),
            "f1_by_class": dict(
                zip(
                    TARGET_CLASSES,
                    map(
                        float,
                        f1_score(
                            target[mask],
                            predictions[mask],
                            labels=TARGET_CLASSES,
                            average=None,
                            zero_division=0,
                        ),
                    ),
                )
            ),
        }
    return groups


def _probability_metrics(
    target: np.ndarray,
    probabilities: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if probabilities is None:
        return None, None
    one_hot = np.zeros_like(probabilities)
    class_to_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    true_indices = np.asarray([class_to_index[str(value)] for value in target])
    one_hot[np.arange(len(target)), true_indices] = 1.0
    true_probability = probabilities[np.arange(len(target)), true_indices]
    log_loss_value = float(-np.mean(np.log(np.clip(true_probability, 1e-15, 1.0))))
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    return log_loss_value, brier


def predict_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
) -> np.ndarray | None:
    """Return probabilities ordered by the canonical target-class contract."""
    if not hasattr(estimator, "predict_proba"):
        return None
    values = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = tuple(str(value) for value in getattr(estimator, "classes_", ()))
    if not classes and hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
        classes = tuple(str(value) for value in getattr(classifier, "classes_", ()))
    if frozenset(classes) != frozenset(TARGET_CLASSES):
        raise ValueError(f"classifier probability classes are invalid: {classes!r}")
    return values[:, [classes.index(label) for label in TARGET_CLASSES]]
