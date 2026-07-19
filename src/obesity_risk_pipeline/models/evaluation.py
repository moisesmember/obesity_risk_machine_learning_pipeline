"""Pure evaluation functions for governed multiclass reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from obesity_risk_pipeline.data.modeling import TARGET_CLASSES


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Metrics that can be serialized without samples or sensitive attributes."""

    row_count: int
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    log_loss: float | None
    multiclass_brier_score: float | None
    recall_by_class: dict[str, float]
    confusion_matrix: list[list[int]]
    metrics_by_gender: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def evaluate_classifier(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    gender: pd.Series,
) -> EvaluationReport:
    """Evaluate an already-fitted classifier without changing its state."""

    predictions = np.asarray(estimator.predict(features), dtype=object)
    target_values = target.to_numpy(dtype=object)
    probabilities = _ordered_probabilities(estimator, features)
    class_recall = recall_score(
        target_values,
        predictions,
        labels=TARGET_CLASSES,
        average=None,
        zero_division=0,
    )
    group_metrics: dict[str, dict[str, float | int]] = {}
    for group_value in sorted(gender.astype(str).unique()):
        mask = gender.astype(str).to_numpy() == group_value
        observed_group_labels = tuple(sorted(set(target_values[mask])))
        group_metrics[group_value] = {
            "row_count": int(mask.sum()),
            "macro_f1": float(
                f1_score(
                    target_values[mask],
                    predictions[mask],
                    labels=TARGET_CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(
                recall_score(
                    target_values[mask],
                    predictions[mask],
                    labels=observed_group_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    probability_log_loss: float | None = None
    brier_score: float | None = None
    if probabilities is not None:
        one_hot_target = np.zeros_like(probabilities)
        class_to_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
        for row_index, label in enumerate(target_values):
            one_hot_target[row_index, class_to_index[str(label)]] = 1.0
        true_class_probabilities = probabilities[
            np.arange(len(target_values)),
            [class_to_index[str(label)] for label in target_values],
        ]
        probability_log_loss = float(
            -np.mean(np.log(np.clip(true_class_probabilities, 1e-15, 1.0)))
        )
        brier_score = float(
            np.mean(np.sum((probabilities - one_hot_target) ** 2, axis=1))
        )

    return EvaluationReport(
        row_count=len(target_values),
        macro_f1=float(
            f1_score(
                target_values,
                predictions,
                labels=TARGET_CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        balanced_accuracy=float(balanced_accuracy_score(target_values, predictions)),
        accuracy=float(accuracy_score(target_values, predictions)),
        log_loss=probability_log_loss,
        multiclass_brier_score=brier_score,
        recall_by_class={
            label: float(value) for label, value in zip(TARGET_CLASSES, class_recall)
        },
        confusion_matrix=confusion_matrix(
            target_values, predictions, labels=TARGET_CLASSES
        ).tolist(),
        metrics_by_gender=group_metrics,
    )


def _ordered_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    raw_probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = tuple(str(label) for label in getattr(estimator, "classes_", ()))
    if not classes and hasattr(estimator, "named_steps"):
        classifier = estimator.named_steps.get("classifier")
        classes = tuple(str(label) for label in getattr(classifier, "classes_", ()))
    if frozenset(classes) != frozenset(TARGET_CLASSES):
        raise ValueError(f"classifier probability classes are invalid: {classes!r}")
    indices = [classes.index(label) for label in TARGET_CLASSES]
    return raw_probabilities[:, indices]
