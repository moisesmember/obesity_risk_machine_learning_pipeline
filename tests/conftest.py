"""Test bootstrap for the local src-layout package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from obesity_risk_pipeline.data.raw_contract import (  # noqa: E402
    RAW_COLUMNS,
    RAW_TARGET_VALUES,
    validate_raw_dataset,
)


@pytest.fixture
def governed_modeling_dataset(tmp_path: Path) -> tuple[Path, str]:
    """Build a deterministic, representative governed snapshot without network I/O."""

    dataset_path = tmp_path / "snapshot" / "obesity_level.csv"
    dataset_path.parent.mkdir(parents=True)
    ordered_targets = sorted(RAW_TARGET_VALUES)
    transport = (
        "Public_Transportation",
        "Automobile",
        "Walking",
        "Motorbike",
        "Bike",
    )
    eating = ("0", "Sometimes", "Frequently", "Always")
    alcohol = ("0", "Sometimes", "Frequently")
    bmi_by_target = {
        "Insufficient_Weight": 17.0,
        "0rmal_Weight": 22.0,
        "Overweight_Level_I": 26.0,
        "Overweight_Level_II": 28.5,
        "Obesity_Type_I": 32.0,
        "Obesity_Type_II": 36.5,
        "Obesity_Type_III": 42.0,
    }

    with dataset_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        record_id = 0
        for target in ordered_targets:
            for repetition in range(20):
                height = 1.55 + (repetition % 8) * 0.04
                bmi = bmi_by_target[target]
                writer.writerow(
                    {
                        "id": record_id,
                        "Gender": "Female" if repetition % 2 == 0 else "Male",
                        "Age": f"{18.0 + record_id / 100:.2f}",
                        "Height": f"{height:.3f}",
                        "Weight": f"{bmi * height**2 + repetition / 100:.3f}",
                        "family_history_with_overweight": str(repetition % 2),
                        "FAVC": str((repetition // 2) % 2),
                        "FCVC": f"{1.0 + repetition % 3:.1f}",
                        "NCP": f"{1.0 + repetition % 4:.1f}",
                        "CAEC": eating[repetition % len(eating)],
                        "SMOKE": str((repetition // 3) % 2),
                        "CH2O": f"{1.0 + repetition % 3:.1f}",
                        "SCC": str((repetition // 4) % 2),
                        "FAF": f"{repetition % 4:.1f}",
                        "TUE": f"{repetition % 3:.1f}",
                        "CALC": alcohol[repetition % len(alcohol)],
                        "MTRANS": transport[repetition % len(transport)],
                        "0be1dad": target,
                    }
                )
                record_id += 1

    profile = validate_raw_dataset(dataset_path)
    manifest = {
        "manifest_schema_version": 1,
        "dataset_version": profile.sha256,
        "file": {
            "name": dataset_path.name,
            "sha256": profile.sha256,
            "byte_size": profile.byte_size,
        },
        "quality": {"row_count": profile.row_count},
    }
    (dataset_path.parent / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dataset_path, profile.sha256
