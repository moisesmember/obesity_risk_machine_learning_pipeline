from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from obesity_risk_pipeline.api.app import _validate_promotion_report, create_app
from obesity_risk_pipeline.api.cli import main as serving_main


class FakePredictionService:
    manifest = {
        "run_id": "approved-run",
        "selected_candidate": "A_full__logistic_regression",
        "configuration": {"feature_set_version": "v2"},
    }

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": frame["id"],
                "predicted_class": ["Normal_Weight"] * len(frame),
                "model_run_id": ["approved-run"] * len(frame),
            }
        )

    def assess_drift(self, frame: pd.DataFrame) -> dict[str, Any]:
        assert not frame.empty
        return {"alerts": ["Weight"]}


def _record() -> dict[str, Any]:
    return {
        "id": 1,
        "Gender": "Female",
        "Age": 25.0,
        "Height": 1.65,
        "Weight": 65.0,
        "family_history_with_overweight": "No",
        "FAVC": "No",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "No",
        "CH2O": 2.0,
        "SCC": "No",
        "FAF": 1.0,
        "TUE": 1.0,
        "CALC": "No",
        "MTRANS": "Walking",
    }


def test_api_serves_traceable_predictions_from_ready_service() -> None:
    app = create_app(
        prediction_service=FakePredictionService(),
        require_promotion=False,
    )

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        response = client.post("/v1/predict", json={"records": [_record()]})

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model_run_id": "approved-run"}
    assert response.status_code == 200
    assert response.json()["model_run_id"] == "approved-run"
    assert response.json()["predictions"][0]["predicted_class"] == "Normal_Weight"
    assert response.json()["telemetry"]["row_count"] == 1
    assert response.json()["telemetry"]["drift_alerts"] == ["Weight"]


def test_api_rejects_target_and_unknown_request_fields() -> None:
    app = create_app(
        prediction_service=FakePredictionService(),
        require_promotion=False,
    )
    record = _record()
    record["NObeyesdad"] = "Normal_Weight"

    with TestClient(app) as client:
        response = client.post("/v1/predict", json={"records": [record]})

    assert response.status_code == 422


def test_api_is_live_but_not_ready_without_approved_artifacts(
    tmp_path: Path,
) -> None:
    app = create_app(
        run_directory=tmp_path / "missing-run",
        promotion_report=tmp_path / "missing-promotion.json",
        require_promotion=True,
    )

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        prediction = client.post("/v1/predict", json={"records": [_record()]})

    assert live.status_code == 200
    assert ready.status_code == 503
    assert prediction.status_code == 503


def test_serving_cli_passes_explicit_network_configuration(
    monkeypatch: Any,
) -> None:
    calls: dict[str, Any] = {}

    def fake_run(application: str, **kwargs: Any) -> None:
        calls["application"] = application
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)

    exit_code = serving_main(
        ["--host", "127.0.0.1", "--port", "8765", "--log-level", "debug"]
    )

    assert exit_code == 0
    assert calls == {
        "application": "obesity_risk_pipeline.api.app:app",
        "host": "127.0.0.1",
        "port": 8765,
        "log_level": "debug",
        "access_log": True,
    }


def test_promotion_report_integrity_and_run_identity_are_enforced(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    report_directory = tmp_path / "promotion"
    run_directory.mkdir()
    report_directory.mkdir()
    (run_directory / "manifest.json").write_text(
        json.dumps({"run_id": "approved-run"}), encoding="utf-8"
    )
    report_path = report_directory / "promotion_report.json"
    report_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "decision": "approved",
                    "technically_approved": True,
                    "source_run_id": "approved-run",
                }
            }
        ),
        encoding="utf-8",
    )
    report_bytes = report_path.read_bytes()
    (report_directory / "manifest.json").write_text(
        json.dumps(
            {
                "promotion_report_sha256": hashlib.sha256(
                    report_bytes
                ).hexdigest(),
                "promotion_report_byte_size": len(report_bytes),
            }
        ),
        encoding="utf-8",
    )

    _validate_promotion_report(report_path, run_directory)
    report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        _validate_promotion_report(report_path, run_directory)
