"""Canonicalization and leakage-safe development/holdout partitioning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.raw_contract import validate_raw_dataset

TARGET_COLUMN: Final = "NObeyesdad"
ID_COLUMN: Final = "id"
TARGET_CLASSES: Final[tuple[str, ...]] = (
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
)
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
)
MODEL_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BEHAVIORAL_FEATURES: Final[tuple[str, ...]] = (
    "Age",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
)
HABITS_FEATURES: Final[tuple[str, ...]] = BEHAVIORAL_FEATURES
CATEGORY_DOMAINS: Final[dict[str, tuple[str, ...]]] = {
    "Gender": ("Female", "Male"),
    "family_history_with_overweight": ("No", "Yes"),
    "FAVC": ("No", "Yes"),
    "CAEC": ("No", "Sometimes", "Frequently", "Always"),
    "SMOKE": ("No", "Yes"),
    "SCC": ("No", "Yes"),
    "CALC": ("No", "Sometimes", "Frequently", "Always"),
    "MTRANS": (
        "Public_Transportation",
        "Automobile",
        "Walking",
        "Motorbike",
        "Bike",
    ),
    "Age_group": ("under_18", "18_24", "25_34", "35_44", "45_54", "55_plus"),
}


class ModelingDataError(ValueError):
    """Raised when a governed snapshot cannot be used for modeling."""


@dataclass(frozen=True, slots=True)
class DataPartition:
    """Features, target and trace-only identifiers for one partition."""

    features: pd.DataFrame
    target: pd.Series
    identifiers: pd.Series


@dataclass(frozen=True, slots=True)
class DataPartitions:
    """Development data and final holdout separated before experimentation."""

    development: DataPartition
    holdout: DataPartition


def load_canonical_dataset(settings: ModelingSettings) -> pd.DataFrame:
    """Validate lineage and return a canonical frame without mutating raw data."""

    dataset_path = settings.dataset_path
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ModelingDataError(f"governed manifest does not exist: {manifest_path}")

    profile = validate_raw_dataset(dataset_path)
    if profile.sha256 != settings.expected_sha256:
        raise ModelingDataError(
            "dataset hash differs from modeling settings; "
            f"expected={settings.expected_sha256}, actual={profile.sha256}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelingDataError(f"invalid governed manifest: {manifest_path}") from exc

    file_metadata = manifest.get("file")
    quality_metadata = manifest.get("quality")
    if not isinstance(file_metadata, dict) or not isinstance(quality_metadata, dict):
        raise ModelingDataError("governed manifest is missing file or quality metadata")
    if (
        manifest.get("dataset_version") != profile.sha256
        or file_metadata.get("sha256") != profile.sha256
        or file_metadata.get("byte_size") != profile.byte_size
        or quality_metadata.get("row_count") != profile.row_count
    ):
        raise ModelingDataError(
            "governed manifest does not match the validated dataset"
        )

    frame = pd.read_csv(dataset_path).rename(columns={"0be1dad": TARGET_COLUMN})
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].replace(
        {"0rmal_Weight": "Normal_Weight"}
    )
    for column in ("family_history_with_overweight", "FAVC", "SMOKE", "SCC"):
        frame[column] = frame[column].astype(str).replace({"0": "No", "1": "Yes"})
    for column in ("CAEC", "CALC"):
        frame[column] = frame[column].astype(str).replace({"0": "No"})
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[ID_COLUMN] = pd.to_numeric(frame[ID_COLUMN], errors="raise").astype("int64")
    return frame.loc[:, (ID_COLUMN,) + MODEL_FEATURES + (TARGET_COLUMN,)].copy()


def split_dataset(frame: pd.DataFrame, settings: ModelingSettings) -> DataPartitions:
    """Create the only final holdout before any model or transformer is fitted."""

    development, holdout = train_test_split(
        frame,
        test_size=settings.holdout_size,
        random_state=settings.random_state,
        shuffle=True,
        stratify=frame[TARGET_COLUMN],
    )
    partitions = DataPartitions(
        development=_to_partition(development),
        holdout=_to_partition(holdout),
    )
    _audit_partitions(partitions)
    return partitions


def build_cross_validator(settings: ModelingSettings) -> StratifiedKFold:
    """Return the governed cross-validator used only within development data."""

    return StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )


def _to_partition(frame: pd.DataFrame) -> DataPartition:
    return DataPartition(
        features=frame.loc[:, MODEL_FEATURES].reset_index(drop=True),
        target=frame[TARGET_COLUMN].reset_index(drop=True),
        identifiers=frame[ID_COLUMN].reset_index(drop=True),
    )


def _audit_partitions(partitions: DataPartitions) -> None:
    development_ids = frozenset(partitions.development.identifiers)
    holdout_ids = frozenset(partitions.holdout.identifiers)
    if development_ids & holdout_ids:
        raise ModelingDataError("development and holdout identifiers overlap")
    expected_targets = frozenset(TARGET_CLASSES)
    for name, partition in (
        ("development", partitions.development),
        ("holdout", partitions.holdout),
    ):
        if frozenset(partition.target.unique()) != expected_targets:
            raise ModelingDataError(f"partition {name!r} lacks target-class coverage")
        if ID_COLUMN in partition.features or TARGET_COLUMN in partition.features:
            raise ModelingDataError(
                f"partition {name!r} exposes trace or target columns"
            )
