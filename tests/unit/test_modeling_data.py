from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import (
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_CLASSES,
    TARGET_COLUMN,
    load_canonical_dataset,
    split_dataset,
)
from obesity_risk_pipeline.data.validation import validate_canonical_frame
from obesity_risk_pipeline.features.engineering import FeatureEngineer
from obesity_risk_pipeline.features.preprocessing import build_preprocessor


def _settings(dataset_path: Path, sha256: str, output_root: Path) -> ModelingSettings:
    return ModelingSettings(
        dataset_path=dataset_path,
        expected_sha256=sha256,
        output_root=output_root,
        holdout_size=0.20,
        cv_folds=2,
        random_state=42,
        target_proportion_tolerance=1.0,
    )


def test_canonicalization_and_split_preserve_trace_without_model_leakage(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    raw_header = dataset_path.read_text(encoding="utf-8").splitlines()[0]
    settings = _settings(dataset_path, sha256, tmp_path / "runs")

    frame = load_canonical_dataset(settings)
    validate_canonical_frame(frame, settings)
    partitions = split_dataset(frame, settings)

    assert raw_header.endswith(",0be1dad")
    assert TARGET_COLUMN in frame and "0be1dad" not in frame
    assert frozenset(frame[TARGET_COLUMN]) == frozenset(TARGET_CLASSES)
    assert len(partitions.development.target) == 112
    assert len(partitions.holdout.target) == 28
    assert tuple(partitions.development.features.columns) == MODEL_FEATURES
    assert ID_COLUMN not in partitions.development.features
    assert TARGET_COLUMN not in partitions.development.features
    assert set(partitions.development.identifiers).isdisjoint(
        partitions.holdout.identifiers
    )


def test_bmi_and_age_representations_are_correct_and_mutually_exclusive() -> None:
    import pandas as pd

    source = pd.DataFrame({"Age": [24.9], "Height": [2.0], "Weight": [100.0]})
    bmi = FeatureEngineer(
        ("Age", "Height", "Weight"), include_bmi=True
    ).fit_transform(source)
    completed = FeatureEngineer(("Age",), age_mode="completed").fit_transform(source)
    grouped = FeatureEngineer(("Age",), age_mode="grouped").fit_transform(source)

    assert bmi.loc[0, "BMI"] == pytest.approx(25.0)
    assert completed.loc[0, "Age_completed"] == 24
    assert "Age" not in completed
    assert grouped.loc[0, "Age_group"] == "18_24"
    assert "Age" not in grouped


def test_unknown_nominal_category_is_ignored_by_sklearn_preprocessor(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    frame = load_canonical_dataset(_settings(dataset_path, sha256, tmp_path))
    preprocessor = build_preprocessor(MODEL_FEATURES)
    preprocessor.fit(frame.loc[:99, MODEL_FEATURES])
    unknown = frame.loc[[100], MODEL_FEATURES].copy()
    unknown["Gender"] = "Unknown_at_inference"

    transformed = preprocessor.transform(unknown)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_preprocessor_statistics_do_not_use_holdout(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path)
    frame = load_canonical_dataset(settings)
    partitions = split_dataset(frame, settings)
    preprocessor = build_preprocessor(MODEL_FEATURES)

    preprocessor.fit(partitions.development.features)

    fitted_mean = preprocessor.named_transformers_["numeric"].named_steps[
        "scaler"
    ].mean_[0]
    assert fitted_mean == pytest.approx(partitions.development.features["Age"].mean())
    assert not np.isclose(fitted_mean, frame["Age"].mean())


def test_age_validation_rejects_non_finite_and_negative_values(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path)
    frame = load_canonical_dataset(settings)
    frame.loc[0, "Age"] = np.inf
    with pytest.raises(ValueError, match="Age.*non-finite"):
        validate_canonical_frame(frame, settings)
    frame.loc[0, "Age"] = -1
    with pytest.raises(ValueError, match="Age.*between"):
        validate_canonical_frame(frame, settings)


@pytest.mark.parametrize("holdout_size", (0.0, 0.5, 0.8))
def test_modeling_settings_reject_invalid_holdout_sizes(
    tmp_path: Path, holdout_size: float
) -> None:
    with pytest.raises(ValueError):
        ModelingSettings(
            dataset_path=tmp_path / "dataset.csv",
            expected_sha256="0" * 64,
            output_root=tmp_path / "runs",
            holdout_size=holdout_size,
        )


def test_modeling_settings_reject_invalid_dataset_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        ModelingSettings(
            dataset_path=tmp_path / "dataset.csv",
            expected_sha256="not-a-sha256",
            output_root=tmp_path / "runs",
        )
