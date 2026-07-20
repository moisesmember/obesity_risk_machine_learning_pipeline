from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.models.tracking import (
    log_post_run_artifacts,
    log_training_run,
)


class _Run:
    def __init__(self, calls: list[tuple[str, Any]], kwargs: dict[str, Any]) -> None:
        self._calls = calls
        self._kwargs = kwargs
        self.info = SimpleNamespace(run_id=kwargs.get("run_id", "parent-run-id"))

    def __enter__(self) -> _Run:
        self._calls.append(("enter_run", self._kwargs))
        return self

    def __exit__(self, *_: object) -> None:
        self._calls.append(("exit_run", self._kwargs))


def _fake_mlflow(calls: list[tuple[str, Any]]) -> ModuleType:
    module = ModuleType("mlflow")
    module.set_tracking_uri = lambda value: calls.append(("tracking_uri", value))
    module.set_experiment = lambda value: calls.append(("experiment", value))
    module.start_run = lambda **kwargs: _Run(calls, kwargs)
    module.set_tags = lambda value: calls.append(("tags", value))
    module.log_params = lambda value: calls.append(("params", value))
    module.log_metrics = lambda value: calls.append(("metrics", value))
    module.log_metric = lambda *args, **kwargs: calls.append(
        ("metric", (args, kwargs))
    )
    module.log_artifact = lambda *args, **kwargs: calls.append(
        ("artifact", (args, kwargs))
    )
    return module


def test_mlflow_receives_parent_child_fold_stage_and_audit_artifacts(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(calls))
    artifact = tmp_path / "evaluation.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    fold = SimpleNamespace(
        macro_f1=0.7,
        weighted_f1=0.8,
        accuracy=0.8,
        balanced_accuracy=0.7,
        ordinal_mae=0.2,
        quadratic_weighted_kappa=0.9,
        log_loss=0.4,
    )
    report = SimpleNamespace(
        metric_mean={"macro_f1": 0.7},
        metric_std={"macro_f1": 0.02},
        per_class_mean={"f1_by_class": {"Normal_Weight": 0.8}},
        per_class_std={"f1_by_class": {"Normal_Weight": 0.01}},
        total_fit_seconds=1.2,
        fold_reports=(fold,),
    )
    settings = ModelingSettings(
        dataset_path=tmp_path / "dataset.csv",
        expected_sha256="0" * 64,
        output_root=tmp_path / "runs",
        mlflow_enabled=True,
    )

    tracking = log_training_run(
        settings=settings,
        run_name="run-name",
        selected_candidate="A_full__logistic_regression",
        cv_reports={"A_full__logistic_regression": report},
        candidate_metadata={
            "A_full__logistic_regression": {
                "model_name": "logistic_regression",
                "experiment_name": "A_full",
                "age_mode": "continuous",
                "categorical_encoding": "nominal",
                "features": ["Age", "Gender"],
                "parameters": {"classifier__C": 1.0},
            }
        },
        artifact_paths=[artifact],
        holdout_metrics={"macro_f1": 0.75, "f1_by_class": {}},
        selected_parameters={"classifier__C": 2.0},
        optimization_summary={
            "status": "completed",
            "best_value": 0.72,
            "best_params": {"C": 2.0},
            "trials": [
                {
                    "number": 0,
                    "state": "COMPLETE",
                    "value": 0.72,
                    "params": {"C": 2.0},
                    "user_attrs": {"macro_f1_std": 0.01},
                }
            ],
        },
        stage_durations={"final_fit": 0.5},
        partition_rows={"development_rows": 100, "holdout_rows": 25},
    )
    post = log_post_run_artifacts(
        settings=settings, tracking=tracking, artifact_paths=[manifest]
    )

    assert tracking == {"status": "logged", "run_id": "parent-run-id"}
    assert post["status"] == "logged"
    assert any(
        name == "enter_run" and value.get("nested") is True
        for name, value in calls
    )
    assert sum(
        name == "enter_run" and value.get("nested") is True
        for name, value in calls
    ) == 2
    assert any(
        name == "metrics" and value.get("stage_final_fit_seconds") == 0.5
        for name, value in calls
    )
    assert any(
        name == "params" and value.get("classifier__C") == 1.0
        for name, value in calls
    )
    assert any(
        name == "metric" and value[1].get("step") == 0
        for name, value in calls
    )
    logged_paths = [Path(value[0][0]).name for name, value in calls if name == "artifact"]
    assert logged_paths == ["evaluation.json", "manifest.json"]
