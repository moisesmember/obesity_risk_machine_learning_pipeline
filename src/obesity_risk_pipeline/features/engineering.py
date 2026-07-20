"""Leakage-safe deterministic feature engineering for sklearn pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Select governed inputs and derive BMI or one exclusive Age representation."""

    def __init__(
        self,
        source_features: Sequence[str],
        *,
        include_bmi: bool = False,
        age_mode: str = "continuous",
    ) -> None:
        self.source_features = tuple(source_features)
        self.include_bmi = include_bmi
        self.age_mode = age_mode

    def fit(
        self, features: pd.DataFrame, target: pd.Series | None = None
    ) -> FeatureEngineer:
        """Validate the feature contract; no statistics are learned."""

        self._validate(features)
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return an independent frame with deterministic derived columns."""

        self._validate(features)
        transformed = features.loc[:, self.source_features].copy()
        if self.include_bmi:
            height = pd.to_numeric(transformed["Height"], errors="raise")
            if (height <= 0).any():
                raise ValueError("BMI requires strictly positive Height values")
            transformed["BMI"] = pd.to_numeric(
                transformed["Weight"], errors="raise"
            ) / np.square(height)

        if "Age" in transformed and self.age_mode != "continuous":
            age = pd.to_numeric(transformed.pop("Age"), errors="raise")
            if self.age_mode == "completed":
                transformed["Age_completed"] = np.floor(age).astype("int64")
            elif self.age_mode == "grouped":
                transformed["Age_group"] = pd.cut(
                    age,
                    bins=(-np.inf, 18, 25, 35, 45, 55, np.inf),
                    right=False,
                    labels=(
                        "under_18",
                        "18_24",
                        "25_34",
                        "35_44",
                        "45_54",
                        "55_plus",
                    ),
                ).astype(str)
            else:
                raise ValueError(
                    "age_mode must be one of: continuous, completed, grouped"
                )
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Expose the deterministic output schema."""

        names = list(self.source_features)
        if self.include_bmi:
            names.append("BMI")
        if "Age" in names and self.age_mode != "continuous":
            names.remove("Age")
            names.append(
                "Age_completed" if self.age_mode == "completed" else "Age_group"
            )
        return np.asarray(names, dtype=object)

    def _validate(self, features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("FeatureEngineer requires a pandas DataFrame")
        missing = set(self.source_features) - set(features.columns)
        if missing:
            raise ValueError(f"missing source features: {sorted(missing)!r}")
        if self.include_bmi and not {"Height", "Weight"}.issubset(self.source_features):
            raise ValueError("BMI experiments require both Height and Weight")
        if self.age_mode not in {"continuous", "completed", "grouped"}:
            raise ValueError("invalid age_mode")
