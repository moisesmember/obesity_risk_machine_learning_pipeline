"""Configurable validation and drift profiles for canonical modeling data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from obesity_risk_pipeline.config.modeling import (
    DEFAULT_NUMERIC_BOUNDS,
    ModelingSettings,
    NumericBounds,
)
from obesity_risk_pipeline.data.modeling import (
    CATEGORY_DOMAINS,
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_CLASSES,
    TARGET_COLUMN,
    ModelingDataError,
)

EXPECTED_TARGET_PROPORTIONS = {
    "Insufficient_Weight": 0.1215,
    "Normal_Weight": 0.1485,
    "Overweight_Level_I": 0.1169,
    "Overweight_Level_II": 0.1215,
    "Obesity_Type_I": 0.1402,
    "Obesity_Type_II": 0.1565,
    "Obesity_Type_III": 0.1949,
}


@dataclass(frozen=True, slots=True)
class DistributionProfile:
    """Sanitized reference distributions persisted beside the fitted model."""

    numeric_quantiles: dict[str, list[float]]
    categorical_proportions: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_canonical_frame(
    frame: pd.DataFrame,
    settings: ModelingSettings | None,
    *,
    require_target: bool = True,
    allow_unknown_categories: bool = False,
    numeric_bounds: dict[str, NumericBounds] | None = None,
) -> None:
    """Validate schema, finite ranges, domains, identifiers and target distribution."""

    required = {ID_COLUMN, *MODEL_FEATURES}
    if require_target:
        if settings is None:
            raise ValueError("settings are required when target validation is enabled")
        required.add(TARGET_COLUMN)
    missing = sorted(required - set(frame.columns))
    unexpected = sorted(set(frame.columns) - required)
    if missing or unexpected:
        raise ModelingDataError(
            f"canonical schema mismatch; missing={missing!r}, unexpected={unexpected!r}"
        )
    if frame.empty:
        raise ModelingDataError("canonical dataset must not be empty")
    if frame[list(required)].isnull().any().any():
        raise ModelingDataError("canonical dataset contains null values")
    if frame[ID_COLUMN].duplicated().any():
        raise ModelingDataError("canonical dataset contains duplicate identifiers")

    active_bounds = numeric_bounds or (
        dict(settings.numeric_bounds)
        if settings is not None
        else DEFAULT_NUMERIC_BOUNDS
    )
    for column, bounds in active_bounds.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelingDataError(f"feature {column!r} contains non-finite values")
        invalid = (values < bounds.minimum) | (values > bounds.maximum)
        if invalid.any():
            raise ModelingDataError(
                f"feature {column!r} must be between {bounds.minimum} and "
                f"{bounds.maximum}; invalid_rows={int(invalid.sum())}"
            )

    for column, allowed in CATEGORY_DOMAINS.items():
        if column not in frame:
            continue
        unknown = set(frame[column].astype(str)) - set(allowed)
        if unknown and not allow_unknown_categories:
            raise ModelingDataError(
                f"feature {column!r} contains unknown categories {sorted(unknown)!r}"
            )

    if require_target:
        observed = frozenset(frame[TARGET_COLUMN].astype(str).unique())
        if observed != frozenset(TARGET_CLASSES):
            raise ModelingDataError("canonical target does not cover the seven classes")
        proportions = frame[TARGET_COLUMN].value_counts(normalize=True)
        deviations = {
            label: abs(float(proportions.get(label, 0.0)) - expected)
            for label, expected in EXPECTED_TARGET_PROPORTIONS.items()
        }
        outside = {
            label: deviation
            for label, deviation in deviations.items()
            if deviation > settings.target_proportion_tolerance
        }
        if outside:
            raise ModelingDataError(
                "target distribution exceeds configured tolerance; "
                f"deviations={outside!r}"
            )


def build_distribution_profile(frame: pd.DataFrame) -> DistributionProfile:
    """Build non-row-level reference summaries for drift checks."""

    quantiles = np.linspace(0.0, 1.0, 11)
    numeric = {
        column: [float(value) for value in frame[column].quantile(quantiles).to_list()]
        for column in DEFAULT_NUMERIC_BOUNDS
    }
    categorical = {
        column: {
            str(label): float(value)
            for label, value in frame[column].value_counts(normalize=True).items()
        }
        for column in CATEGORY_DOMAINS
        if column in frame
    }
    return DistributionProfile(numeric, categorical)


def assess_distribution_shift(
    frame: pd.DataFrame,
    profile: DistributionProfile,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Calculate PSI-style drift indicators without exposing individual rows."""

    scores: dict[str, float] = {}
    epsilon = 1e-6
    for column, quantiles in profile.numeric_quantiles.items():
        internal = np.unique(np.asarray(quantiles[1:-1], dtype=float))
        edges = np.concatenate(([-np.inf], internal, [np.inf]))
        observed, _ = np.histogram(frame[column].to_numpy(dtype=float), bins=edges)
        observed_proportions = np.clip(observed / max(observed.sum(), 1), epsilon, None)
        expected = np.full(len(observed_proportions), 1.0 / len(observed_proportions))
        ratio = observed_proportions / expected
        scores[column] = float(
            np.sum((observed_proportions - expected) * np.log(ratio))
        )
    for column, expected_mapping in profile.categorical_proportions.items():
        observed_mapping = (
            frame[column].astype(str).value_counts(normalize=True).to_dict()
        )
        labels = set(expected_mapping) | set(observed_mapping)
        expected = np.asarray(
            [max(expected_mapping.get(label, 0.0), epsilon) for label in labels]
        )
        observed = np.asarray(
            [max(observed_mapping.get(label, 0.0), epsilon) for label in labels]
        )
        scores[column] = float(
            np.sum((observed - expected) * np.log(observed / expected))
        )
    return {
        "threshold": threshold,
        "scores": scores,
        "alerts": sorted(name for name, score in scores.items() if score > threshold),
    }
