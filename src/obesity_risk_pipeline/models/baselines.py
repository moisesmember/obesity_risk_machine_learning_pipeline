"""Auditable baseline candidates for obesity-level classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from obesity_risk_pipeline.data.modeling import (
    HABITS_FEATURES,
    MODEL_FEATURES,
    TARGET_CLASSES,
)
from obesity_risk_pipeline.features.preprocessing import build_preprocessor


@dataclass(frozen=True, slots=True)
class BaselineCandidate:
    """Named estimator and the governed feature set it is allowed to consume."""

    name: str
    estimator: BaseEstimator
    features: tuple[str, ...]


class BmiRuleClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic audit baseline; not a clinical diagnostic rule."""

    classes_: np.ndarray

    def fit(self, features: pd.DataFrame, target: pd.Series) -> BmiRuleClassifier:
        """Validate required columns and expose the canonical class contract."""

        self._validate_features(features)
        self.classes_ = np.asarray(TARGET_CLASSES, dtype=object)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Map derived BMI to the dataset's seven ordered labels."""

        self._validate_features(features)
        bmi = features["Weight"].to_numpy(dtype=float) / np.square(
            features["Height"].to_numpy(dtype=float)
        )
        bins = np.asarray([-np.inf, 18.5, 25.0, 27.5, 30.0, 35.0, 40.0, np.inf])
        indices = np.digitize(bmi, bins[1:-1], right=False)
        return np.asarray(TARGET_CLASSES, dtype=object)[indices]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return deterministic one-hot probabilities for metric parity."""

        predictions = self.predict(features)
        class_to_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
        probabilities = np.zeros((len(predictions), len(TARGET_CLASSES)), dtype=float)
        for row_index, label in enumerate(predictions):
            probabilities[row_index, class_to_index[str(label)]] = 1.0
        return probabilities

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        missing = {"Height", "Weight"} - set(features.columns)
        if missing:
            raise ValueError(f"BMI baseline requires features {sorted(missing)!r}")
        if (features["Height"] <= 0).any():
            raise ValueError("BMI baseline requires positive Height values")


def build_baseline_candidates(random_state: int) -> Mapping[str, BaselineCandidate]:
    """Create fixed, untuned baselines under the same train/validation contract."""

    candidates = (
        BaselineCandidate(
            name="bmi_rule",
            estimator=Pipeline(steps=(("classifier", BmiRuleClassifier()),)),
            features=("Height", "Weight"),
        ),
        BaselineCandidate(
            name="dummy_stratified",
            estimator=Pipeline(
                steps=(
                    ("preprocessor", build_preprocessor(MODEL_FEATURES)),
                    (
                        "classifier",
                        DummyClassifier(
                            strategy="stratified", random_state=random_state
                        ),
                    ),
                )
            ),
            features=MODEL_FEATURES,
        ),
        BaselineCandidate(
            name="logistic_full",
            estimator=Pipeline(
                steps=(
                    ("preprocessor", build_preprocessor(MODEL_FEATURES)),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1_000, random_state=random_state),
                    ),
                )
            ),
            features=MODEL_FEATURES,
        ),
        BaselineCandidate(
            name="tree_full",
            estimator=Pipeline(
                steps=(
                    ("preprocessor", build_preprocessor(MODEL_FEATURES)),
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            max_depth=5,
                            min_samples_leaf=10,
                            random_state=random_state,
                        ),
                    ),
                )
            ),
            features=MODEL_FEATURES,
        ),
        BaselineCandidate(
            name="logistic_without_anthropometrics",
            estimator=Pipeline(
                steps=(
                    ("preprocessor", build_preprocessor(HABITS_FEATURES)),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1_000, random_state=random_state),
                    ),
                )
            ),
            features=HABITS_FEATURES,
        ),
    )
    return {candidate.name: candidate for candidate in candidates}
