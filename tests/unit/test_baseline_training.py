from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import (
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
    load_canonical_dataset,
    split_dataset,
)
from obesity_risk_pipeline.features.experiments import build_experiment_catalog
from obesity_risk_pipeline.inference.service import PredictionService
from obesity_risk_pipeline.pipelines.training import run_experiments


def _settings(dataset_path: Path, sha256: str, output_root: Path) -> ModelingSettings:
    return ModelingSettings(
        dataset_path=dataset_path,
        expected_sha256=sha256,
        output_root=output_root,
        holdout_size=0.20,
        cv_folds=2,
        random_state=42,
        feature_set_version="test-v2",
        target_proportion_tolerance=1.0,
    )


def test_run_selects_by_cv_and_persists_traceable_inference_artifacts(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path / "runs")

    result = run_experiments(
        settings,
        experiment_names=("A_full", "D_behavioral", "E_body_bmi"),
        model_names=("dummy", "logistic_regression"),
    )

    assert len(result.cv_reports) == 6
    assert result.holdout_report.row_count == 28
    for path in (
        result.model_path,
        result.evaluation_path,
        result.leaderboard_path,
        result.predictions_path,
        result.manifest_path,
        result.training_events_path,
    ):
        assert path.is_file()
    assert not list(settings.output_root.glob(".*.publishing"))

    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["selection_partition"] == "development_cross_validation"
    assert evaluation["selection_metric"] == "macro_f1_mean"
    assert evaluation["selected_candidate"] == result.selected_candidate
    assert "holdout" in evaluation

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["partitions"] == {"development_rows": 112, "holdout_rows": 28}
    assert manifest["serialization_parity_validated"] is True
    assert manifest["promotion_status"] == "not_requested"
    for filename, metadata in manifest["artifacts"].items():
        artifact_path = result.run_directory / filename
        assert metadata["sha256"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

    events = [
        json.loads(line)
        for line in result.training_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(event["run_id"] == result.run_id for event in events)
    completed_stages = {
        event["stage"] for event in events if event["status"] == "completed"
    }
    assert {
        "dataset_loading",
        "data_contract_validation",
        "dataset_split",
        "cross_validation",
        "final_fit",
        "holdout_evaluation",
        "training_run",
    }.issubset(completed_stages)
    completed_folds = [
        event
        for event in events
        if event["stage"] == "cross_validation_fold"
        and event["status"] == "completed"
    ]
    assert len(completed_folds) == 12
    assert all("macro_f1" in event for event in completed_folds)

    predictions = pd.read_csv(result.predictions_path)
    probability_columns = [
        name for name in predictions if name.startswith("probability_")
    ]
    assert predictions[ID_COLUMN].is_unique
    assert len(predictions) == 28
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0)


def test_loaded_service_preserves_id_schema_unknown_categories_and_reproducibility(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = _settings(dataset_path, sha256, tmp_path / "runs")
    kwargs = {
        "experiment_names": ("A_full",),
        "model_names": ("logistic_regression",),
    }
    first = run_experiments(settings, **kwargs)
    second = run_experiments(settings, **kwargs)
    assert first.selected_candidate == second.selected_candidate
    assert first.cv_reports[first.selected_candidate].metric_mean == second.cv_reports[
        second.selected_candidate
    ].metric_mean

    frame = load_canonical_dataset(settings)
    holdout = split_dataset(frame, settings).holdout
    inference_input = pd.concat(
        [holdout.identifiers.rename(ID_COLUMN), holdout.features], axis=1
    )
    inference_input.loc[0, "Gender"] = "new_category"
    service = PredictionService.load(first.run_directory)

    output = service.predict(inference_input)

    assert tuple(output[ID_COLUMN]) == tuple(holdout.identifiers)
    assert TARGET_COLUMN not in output
    assert len(output) == len(holdout.target)
    probability_columns = [name for name in output if name.startswith("probability_")]
    assert np.allclose(output[probability_columns].sum(axis=1), 1.0)

    first.model_path.write_bytes(first.model_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="integrity mismatch"):
        PredictionService.load(first.run_directory)


def test_experiment_contracts_never_expose_identifier_or_target() -> None:
    for experiment in build_experiment_catalog().values():
        assert ID_COLUMN not in experiment.source_features
        assert TARGET_COLUMN not in experiment.source_features
        assert set(experiment.source_features).issubset(MODEL_FEATURES)
