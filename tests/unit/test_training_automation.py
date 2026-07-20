from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from obesity_risk_pipeline.config.experiments import ExperimentPlan
from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.config.settings import IngestionSettings
from obesity_risk_pipeline.ingestion.service import IngestionResult
from obesity_risk_pipeline.pipelines import automation


def _settings(tmp_path: Path) -> tuple[IngestionSettings, ModelingSettings]:
    sha256 = "a" * 64
    ingestion = IngestionSettings(
        dataset_slug="owner/dataset",
        expected_filename="obesity_level.csv",
        expected_sha256=sha256,
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
    )
    modeling = ModelingSettings(
        dataset_path=tmp_path / "placeholder.csv",
        expected_sha256=sha256,
        output_root=tmp_path / "runs",
        target_proportion_tolerance=1.0,
    )
    return ingestion, modeling


def test_quick_workflow_passes_governed_ingestion_path_to_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestion_settings, modeling_settings = _settings(tmp_path)
    dataset_path = tmp_path / "raw" / ("a" * 64) / "obesity_level.csv"
    ingestion_result = IngestionResult(
        dataset_path=dataset_path,
        manifest_path=dataset_path.parent / "manifest.json",
        sha256="a" * 64,
        row_count=140,
        reused_existing_snapshot=True,
    )
    captured: dict[str, Any] = {}
    training_result = object()

    class FakeIngestionService:
        def __init__(self, **kwargs: Any) -> None:
            captured["ingestion_kwargs"] = kwargs

        def run(self) -> IngestionResult:
            return ingestion_result

    def fake_train(settings: ModelingSettings) -> Any:
        captured["modeling_settings"] = settings
        return training_result

    monkeypatch.setattr(automation, "KaggleIngestionService", FakeIngestionService)
    monkeypatch.setattr(automation, "train_baselines", fake_train)

    result = automation.run_training_workflow(
        mode="quick",
        ingestion_settings=ingestion_settings,
        modeling_settings=modeling_settings,
    )

    assert result.ingestion is ingestion_result
    assert result.training is training_result
    assert captured["modeling_settings"].dataset_path == dataset_path.resolve()
    assert captured["modeling_settings"].expected_sha256 == "a" * 64


def test_full_workflow_requires_and_forwards_external_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ingestion_settings, modeling_settings = _settings(tmp_path)
    ingestion_result = IngestionResult(
        dataset_path=tmp_path / "dataset.csv",
        manifest_path=tmp_path / "manifest.json",
        sha256="a" * 64,
        row_count=140,
        reused_existing_snapshot=False,
    )
    captured: dict[str, Any] = {}

    class FakeIngestionService:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self) -> IngestionResult:
            return ingestion_result

    def fake_run(settings: ModelingSettings, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(automation, "KaggleIngestionService", FakeIngestionService)
    monkeypatch.setattr(automation, "run_experiments", fake_run)
    plan = ExperimentPlan(("A_full", "D_behavioral"), ("dummy",))

    automation.run_training_workflow(
        mode="full",
        ingestion_settings=ingestion_settings,
        modeling_settings=modeling_settings,
        experiment_plan=plan,
    )

    assert captured["experiment_names"] == plan.experiments
    assert captured["model_names"] == plan.models

    with pytest.raises(ValueError, match="requires an experiment plan"):
        automation.run_training_workflow(
            mode="full",
            ingestion_settings=ingestion_settings,
            modeling_settings=modeling_settings,
        )


def test_powershell_bootstrap_uses_explicit_venv_and_checked_commands() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_training_pipeline.ps1"
    ).read_text(encoding="utf-8")

    assert "Invoke-Checked" in script
    assert "Scripts\\python.exe" in script
    assert "obesity-training-pipeline.exe" in script
    assert "requirements-training.txt" in script
    assert "requirements-modeling.txt" in script
    assert "KAGGLE_API_TOKEN" not in script
