"""Factories for fold-fitted nominal, ordinal and numeric preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from obesity_risk_pipeline.data.modeling import CATEGORY_DOMAINS


def build_preprocessor(
    features: Sequence[str],
    *,
    scale_numeric: bool = True,
    categorical_encoding: str = "nominal",
) -> ColumnTransformer:
    """Build preprocessing whose learned state is isolated inside each fold."""

    feature_names = tuple(features)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("features must be non-empty and unique")
    if categorical_encoding not in {"nominal", "ordinal"}:
        raise ValueError("categorical_encoding must be nominal or ordinal")

    categorical = tuple(name for name in feature_names if name in CATEGORY_DOMAINS)
    ordinal = (
        tuple(name for name in categorical if name in {"CAEC", "CALC"})
        if categorical_encoding == "ordinal"
        else ()
    )
    nominal = tuple(name for name in categorical if name not in ordinal)
    numeric = tuple(name for name in feature_names if name not in categorical)
    transformers: list[tuple[str, Pipeline, tuple[str, ...]]] = []

    if numeric:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    (
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    )
                ),
                nominal,
            )
        )
    if ordinal:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    (
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[CATEGORY_DOMAINS[name] for name in ordinal],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    )
                ),
                ordinal,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")
