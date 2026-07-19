"""Factories for fit/transform-compatible feature preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from obesity_risk_pipeline.data.modeling import CATEGORY_DOMAINS


def build_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    """Build deterministic preprocessing whose statistics are fitted on train only."""

    feature_names = tuple(features)
    categorical = tuple(
        feature for feature in feature_names if feature in CATEGORY_DOMAINS
    )
    numeric = tuple(feature for feature in feature_names if feature not in categorical)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("features must be a non-empty sequence without duplicates")

    transformers: list[tuple[str, Pipeline, tuple[str, ...]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=(
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    )
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=(
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                categories=[CATEGORY_DOMAINS[name] for name in categorical],
                                handle_unknown="error",
                                sparse_output=True,
                            ),
                        ),
                    )
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
