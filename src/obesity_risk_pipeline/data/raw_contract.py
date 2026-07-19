"""Executable contract for the immutable obesity dataset downloaded from Kaggle."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

RAW_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
    "0be1dad",
)

RAW_TARGET_VALUES: Final[frozenset[str]] = frozenset(
    {
        "Insufficient_Weight",
        "0rmal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    }
)


class DatasetValidationError(ValueError):
    """Raised when a downloaded dataset violates its executable contract."""


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Inclusive valid range for a finite numeric field."""

    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class RawDatasetContract:
    """Schema and domain constraints expected from the Kaggle raw file."""

    columns: tuple[str, ...] = RAW_COLUMNS
    target_column: str = "0be1dad"
    expected_target_values: frozenset[str] = RAW_TARGET_VALUES


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Non-sensitive lineage and quality metadata derived from a validated CSV."""

    sha256: str
    byte_size: int
    row_count: int
    columns: tuple[str, ...]
    target_counts: dict[str, int]


_ALLOWED_CATEGORIES: Final[dict[str, frozenset[str]]] = {
    "Gender": frozenset({"Female", "Male"}),
    "family_history_with_overweight": frozenset({"0", "1"}),
    "FAVC": frozenset({"0", "1"}),
    "CAEC": frozenset({"0", "Sometimes", "Frequently", "Always"}),
    "SMOKE": frozenset({"0", "1"}),
    "SCC": frozenset({"0", "1"}),
    "CALC": frozenset({"0", "Sometimes", "Frequently", "Always"}),
    "MTRANS": frozenset(
        {"Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"}
    ),
}

_NUMERIC_RANGES: Final[dict[str, NumericRange]] = {
    "Age": NumericRange(0.0, 120.0),
    "Height": NumericRange(0.5, 2.5),
    "Weight": NumericRange(10.0, 500.0),
    "FCVC": NumericRange(1.0, 3.0),
    "NCP": NumericRange(1.0, 4.0),
    "CH2O": NumericRange(1.0, 3.0),
    "FAF": NumericRange(0.0, 3.0),
    "TUE": NumericRange(0.0, 2.0),
}


def calculate_sha256(path: Path) -> str:
    """Calculate a file SHA-256 without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_numeric(field: str, value: str, line_number: int) -> None:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise DatasetValidationError(
            f"line {line_number}: field '{field}' must be numeric, received {value!r}"
        ) from exc

    bounds = _NUMERIC_RANGES[field]
    if not math.isfinite(parsed) or not bounds.minimum <= parsed <= bounds.maximum:
        raise DatasetValidationError(
            f"line {line_number}: field '{field}' must be finite and between "
            f"{bounds.minimum} and {bounds.maximum}, received {value!r}"
        )


def validate_raw_dataset(
    path: Path,
    contract: RawDatasetContract | None = None,
) -> DatasetProfile:
    """Validate schema, domains, identifiers, duplicates and target coverage."""

    active_contract = contract or RawDatasetContract()
    if not path.is_file():
        raise DatasetValidationError(f"dataset file does not exist: {path}")

    seen_ids: set[int] = set()
    seen_records: set[tuple[str, ...]] = set()
    target_counts: Counter[str] = Counter()
    feature_columns = tuple(column for column in active_contract.columns if column != "id")
    row_count = 0

    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetValidationError(f"unable to open dataset file: {path}") from exc

    with stream:
        reader = csv.DictReader(stream)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != active_contract.columns:
            missing = sorted(set(active_contract.columns) - set(actual_columns))
            unexpected = sorted(set(actual_columns) - set(active_contract.columns))
            raise DatasetValidationError(
                "raw schema mismatch; "
                f"expected_order={list(active_contract.columns)!r}, "
                f"actual_order={list(actual_columns)!r}, missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            missing_values = [
                field
                for field in active_contract.columns
                if row.get(field) is None or row[field].strip() == ""
            ]
            if missing_values:
                raise DatasetValidationError(
                    f"line {line_number}: empty required fields {missing_values!r}"
                )

            try:
                record_id = int(row["id"])
            except ValueError as exc:
                raise DatasetValidationError(
                    f"line {line_number}: field 'id' must be an integer"
                ) from exc
            if record_id in seen_ids:
                raise DatasetValidationError(
                    f"line {line_number}: duplicate id {record_id}"
                )
            seen_ids.add(record_id)

            record_signature = tuple(row[column] for column in feature_columns)
            if record_signature in seen_records:
                raise DatasetValidationError(
                    f"line {line_number}: duplicate record excluding id"
                )
            seen_records.add(record_signature)

            for field, allowed_values in _ALLOWED_CATEGORIES.items():
                if row[field] not in allowed_values:
                    raise DatasetValidationError(
                        f"line {line_number}: field '{field}' has unknown category "
                        f"{row[field]!r}; expected one of {sorted(allowed_values)!r}"
                    )

            for field in _NUMERIC_RANGES:
                _validate_numeric(field, row[field], line_number)

            target = row[active_contract.target_column]
            if target not in active_contract.expected_target_values:
                raise DatasetValidationError(
                    f"line {line_number}: unexpected target {target!r}"
                )
            target_counts[target] += 1

    if row_count == 0:
        raise DatasetValidationError("dataset must contain at least one record")

    missing_classes = active_contract.expected_target_values - target_counts.keys()
    if missing_classes:
        raise DatasetValidationError(
            f"dataset is missing required target classes: {sorted(missing_classes)!r}"
        )

    return DatasetProfile(
        sha256=calculate_sha256(path),
        byte_size=path.stat().st_size,
        row_count=row_count,
        columns=active_contract.columns,
        target_counts=dict(sorted(target_counts.items())),
    )
