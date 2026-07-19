from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.data.modeling import (
    HABITS_FEATURES,
    ID_COLUMN,
    TARGET_COLUMN,
    load_canonical_dataset,
    split_dataset,
)
from obesity_risk_pipeline.models.baselines import build_baseline_candidates
from obesity_risk_pipeline.pipelines.training import train_baselines


def test_baseline_run_selects_on_validation_and_persists_evaluated_pipeline(
    governed_modeling_dataset: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_path, sha256 = governed_modeling_dataset
    settings = ModelingSettings(
        dataset_path=dataset_path,
        expected_sha256=sha256,
        output_root=tmp_path / "runs",
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
        feature_set_version="test-v1",
    )

    result = train_baselines(settings)

    assert set(result.validation_reports) == {
        "bmi_rule",
        "dummy_stratified",
        "logistic_full",
        "tree_full",
        "logistic_without_anthropometrics",
    }
    assert result.test_report.row_count == 28
    assert result.model_path.is_file()
    assert result.evaluation_path.is_file()
    assert result.manifest_path.is_file()
    assert not list(settings.output_root.glob(".*.publishing"))

    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["selection_partition"] == "validation"
    assert evaluation["selection_metric"] == "macro_f1"
    assert evaluation["selected_candidate"] == result.selected_candidate
    assert "test" in evaluation

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["sha256"] == sha256
    assert manifest["partitions"] == {
        "train_rows": 84,
        "validation_rows": 28,
        "test_rows": 28,
    }
    assert manifest["mlflow_status"] == "not_logged_policy_pending"
    assert manifest["promotion_status"] == "not_requested"
    assert manifest["serialization_parity_validated"] is True
    for filename in ("model.joblib", "evaluation.json"):
        artifact_path = result.run_directory / filename
        assert manifest["artifacts"][filename]["sha256"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

    candidates = build_baseline_candidates(settings.random_state)
    selected_features = candidates[result.selected_candidate].features
    frame = load_canonical_dataset(settings)
    partitions = split_dataset(frame, settings)
    loaded_pipeline = joblib.load(result.model_path)
    predictions = loaded_pipeline.predict(
        partitions.test.features.loc[:, selected_features]
    )
    assert len(predictions) == result.test_report.row_count


def test_candidate_feature_contracts_exclude_identifiers_target_and_anthropometrics() -> None:
    candidates = build_baseline_candidates(random_state=42)

    for candidate in candidates.values():
        assert ID_COLUMN not in candidate.features
        assert TARGET_COLUMN not in candidate.features
    habits_candidate = candidates["logistic_without_anthropometrics"]
    assert habits_candidate.features == HABITS_FEATURES
    assert "Height" not in habits_candidate.features
    assert "Weight" not in habits_candidate.features
