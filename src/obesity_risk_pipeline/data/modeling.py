"""Canonicalization and leakage-safe partitions for baseline modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

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
HABITS_FEATURES: Final[tuple[str, ...]] = tuple(
    feature for feature in MODEL_FEATURES if feature not in {"Height", "Weight"}
)
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
}


class ModelingDataError(ValueError):
    """Raised when a governed snapshot cannot be used for modeling."""


@dataclass(frozen=True, slots=True)
class DataPartition:
    """Features, target and trace-only identifiers for one untouched partition."""

    features: pd.DataFrame
    target: pd.Series
    identifiers: pd.Series


@dataclass(frozen=True, slots=True)
class DataPartitions:
    """Train, validation and test partitions produced before any fitted transform."""

    train: DataPartition
    validation: DataPartition
    test: DataPartition


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
        raise ModelingDataError("governed manifest does not match the validated dataset")

    frame = pd.read_csv(dataset_path)
    frame = frame.rename(columns={"0be1dad": TARGET_COLUMN})
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

    observed_targets = frozenset(frame[TARGET_COLUMN].unique())
    if observed_targets != frozenset(TARGET_CLASSES):
        raise ModelingDataError(
            "canonical target coverage mismatch; "
            f"observed={sorted(observed_targets)!r}"
        )
    return frame.loc[:, (ID_COLUMN,) + MODEL_FEATURES + (TARGET_COLUMN,)].copy()


def split_dataset(frame: pd.DataFrame, settings: ModelingSettings) -> DataPartitions:
    """Split by target before any transformer fit and audit partition isolation."""

    development, test = train_test_split(
        frame,
        test_size=settings.test_size,
        random_state=settings.random_state,
        shuffle=True,
        stratify=frame[TARGET_COLUMN],
    )
    relative_validation_size = settings.validation_size / (1.0 - settings.test_size)
    train, validation = train_test_split(
        development,
        test_size=relative_validation_size,
        random_state=settings.random_state,
        shuffle=True,
        stratify=development[TARGET_COLUMN],
    )

    partitions = DataPartitions(
        train=_to_partition(train),
        validation=_to_partition(validation),
        test=_to_partition(test),
    )
    _audit_partitions(partitions)
    return partitions


def _to_partition(frame: pd.DataFrame) -> DataPartition:
    return DataPartition(
        features=frame.loc[:, MODEL_FEATURES].reset_index(drop=True),
        target=frame[TARGET_COLUMN].reset_index(drop=True),
        identifiers=frame[ID_COLUMN].reset_index(drop=True),
    )


def _audit_partitions(partitions: DataPartitions) -> None:
    expected_targets = frozenset(TARGET_CLASSES)
    identifier_sets = {
        name: frozenset(partition.identifiers.tolist())
        for name, partition in (
            ("train", partitions.train),
            ("validation", partitions.validation),
            ("test", partitions.test),
        )
    }
    if identifier_sets["train"] & identifier_sets["validation"]:
        raise ModelingDataError("train and validation identifiers overlap")
    if identifier_sets["train"] & identifier_sets["test"]:
        raise ModelingDataError("train and test identifiers overlap")
    if identifier_sets["validation"] & identifier_sets["test"]:
        raise ModelingDataError("validation and test identifiers overlap")

    for name, partition in (
        ("train", partitions.train),
        ("validation", partitions.validation),
        ("test", partitions.test),
    ):
        observed_targets = frozenset(partition.target.unique())
        if observed_targets != expected_targets:
            raise ModelingDataError(
                f"partition {name!r} does not cover every target class"
            )
        for column, declared_domain in CATEGORY_DOMAINS.items():
            observed = frozenset(partition.features[column].unique())
            full_observed = frozenset(
                pd.concat(
                    [
                        partitions.train.features[column],
                        partitions.validation.features[column],
                        partitions.test.features[column],
                    ],
                    ignore_index=True,
                ).unique()
            )
            missing = full_observed - observed
            if missing:
                raise ModelingDataError(
                    f"partition {name!r} lacks observed categories {sorted(missing)!r} "
                    f"for feature {column!r}; choose a reviewed seed or split policy"
                )
            unknown = observed - frozenset(declared_domain)
            if unknown:
                raise ModelingDataError(
                    f"partition {name!r} contains undeclared categories "
                    f"{sorted(unknown)!r} for feature {column!r}"
                )
