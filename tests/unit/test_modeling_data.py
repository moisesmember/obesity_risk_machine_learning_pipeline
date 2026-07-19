from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_CLASSES,
    TARGET_COLUMN,
    load_canonical_dataset,
    split_dataset,
)
from obesity_risk_pipeline.features.preprocessing import build_preprocessor


def _settings(dataset_path: Path, sha256: str, output_root: Path) -> ModelingSettings:
    return ModelingSettings(
        dataset_path=dataset_path,
        expected_sha256=sha256,
        output_root=output_root,
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
    )


def test_canonicalization_preserves_raw_and_normalizes_contract(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    raw_header = dataset_path.read_text(encoding="utf-8").splitlines()[0]

    frame = load_canonical_dataset(_settings(dataset_path, sha256, tmp_path / "runs"))

    assert raw_header.endswith(",0be1dad")
    assert TARGET_COLUMN in frame.columns
    assert "0be1dad" not in frame.columns
    assert "0rmal_Weight" not in set(frame[TARGET_COLUMN])
    assert frozenset(frame[TARGET_COLUMN]) == frozenset(TARGET_CLASSES)
    assert set(frame["CAEC"]) == {"No", "Sometimes", "Frequently", "Always"}
    assert set(frame["family_history_with_overweight"]) == {"No", "Yes"}


def test_split_is_stratified_disjoint_and_excludes_trace_columns(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path / "runs")
    frame = load_canonical_dataset(settings)

    partitions = split_dataset(frame, settings)

    assert len(partitions.train.target) == 84
    assert len(partitions.validation.target) == 28
    assert len(partitions.test.target) == 28
    assert tuple(partitions.train.features.columns) == MODEL_FEATURES
    assert ID_COLUMN not in partitions.train.features
    assert TARGET_COLUMN not in partitions.train.features
    assert set(partitions.train.identifiers).isdisjoint(partitions.test.identifiers)
    for partition in (partitions.train, partitions.validation, partitions.test):
        assert frozenset(partition.target) == frozenset(TARGET_CLASSES)
        for feature in CATEGORICAL_FEATURES:
            assert set(partition.features[feature]) == set(frame[feature])


def test_preprocessor_statistics_are_fitted_only_from_train(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path / "runs")
    frame = load_canonical_dataset(settings)
    partitions = split_dataset(frame, settings)
    preprocessor = build_preprocessor(MODEL_FEATURES)

    preprocessor.fit(partitions.train.features)

    fitted_mean = preprocessor.named_transformers_["numeric"].named_steps[
        "scaler"
    ].mean_[0]
    assert fitted_mean == pytest.approx(partitions.train.features["Age"].mean())
    assert not np.isclose(fitted_mean, pd.to_numeric(frame["Age"]).mean())


@pytest.mark.parametrize(
    ("test_size", "validation_size"),
    ((0.0, 0.2), (0.2, 0.0), (0.6, 0.4)),
)
def test_modeling_settings_reject_invalid_partition_sizes(
    tmp_path: Path, test_size: float, validation_size: float
) -> None:
    with pytest.raises(ValueError):
        ModelingSettings(
            dataset_path=tmp_path / "dataset.csv",
            expected_sha256="0" * 64,
            output_root=tmp_path / "runs",
            test_size=test_size,
            validation_size=validation_size,
        )


def test_modeling_settings_reject_invalid_dataset_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        ModelingSettings(
            dataset_path=tmp_path / "dataset.csv",
            expected_sha256="not-a-sha256",
            output_root=tmp_path / "runs",
        )
