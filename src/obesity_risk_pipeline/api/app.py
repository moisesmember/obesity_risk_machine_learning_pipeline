"""FastAPI application that serves only compatible, explicitly promoted artifacts."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status

from obesity_risk_pipeline.api.schemas import (
    BatchTelemetry,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from obesity_risk_pipeline.inference.service import PredictionService


LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    run_directory: Path | None = None,
    promotion_report: Path | None = None,
    require_promotion: bool | None = None,
    prediction_service: PredictionService | None = None,
) -> FastAPI:
    """Create an injectable app with fail-closed startup and readiness state."""

    configured_run = run_directory or Path(
        os.getenv("OBESITY_MODEL_RUN_DIRECTORY", "/models/current")
    )
    configured_report = promotion_report or Path(
        os.getenv(
            "OBESITY_PROMOTION_REPORT",
            "/governance/promotion_report.json",
        )
    )
    promotion_required = (
        _environment_boolean("OBESITY_REQUIRE_PROMOTION", True)
        if require_promotion is None
        else require_promotion
    )
    max_batch_size = _environment_integer(
        "OBESITY_API_MAX_BATCH_SIZE", 1_000, minimum=1, maximum=10_000
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.prediction_service = prediction_service
        app.state.readiness_error = None
        if prediction_service is None:
            try:
                if promotion_required:
                    _validate_promotion_report(configured_report, configured_run)
                app.state.prediction_service = PredictionService.load(configured_run)
            except (OSError, ValueError, RuntimeError) as exc:
                app.state.readiness_error = type(exc).__name__
                LOGGER.error("Model serving is not ready: %s", exc)
        yield

    app = FastAPI(
        title="Obesity Risk Model API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.max_batch_size = max_batch_size
    app.state.promotion_required = promotion_required

    @app.get("/health/live", response_model=HealthResponse)
    def liveness() -> HealthResponse:
        return HealthResponse(status="alive")

    @app.get("/health/ready", response_model=HealthResponse)
    def readiness(request: Request) -> HealthResponse:
        service = getattr(request.app.state, "prediction_service", None)
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="approved model artifacts are not ready",
            )
        return HealthResponse(
            status="ready", model_run_id=str(service.manifest["run_id"])
        )

    @app.get("/v1/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        service = _ready_service(request)
        manifest = service.manifest
        return ModelInfoResponse(
            model_run_id=str(manifest["run_id"]),
            selected_candidate=str(manifest["selected_candidate"]),
            feature_set_version=str(
                manifest["configuration"]["feature_set_version"]
            ),
            promotion_required=promotion_required,
        )

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        service = _ready_service(request)
        if len(payload.records) > request.app.state.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "prediction batch exceeds OBESITY_API_MAX_BATCH_SIZE="
                    f"{request.app.state.max_batch_size}"
                ),
            )
        frame = pd.DataFrame(
            [record.model_dump(mode="python") for record in payload.records]
        )
        started = time.perf_counter()
        try:
            predictions = service.predict(frame)
            drift = service.assess_drift(frame)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            LOGGER.error("Prediction runtime failure: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model prediction is temporarily unavailable",
            ) from exc
        duration_milliseconds = (time.perf_counter() - started) * 1_000.0
        probability_columns = [
            name for name in predictions if name.startswith("probability_")
        ]
        mean_max_probability = (
            float(predictions[probability_columns].max(axis=1).mean())
            if probability_columns
            else None
        )
        class_counts = {
            str(label): int(count)
            for label, count in predictions["predicted_class"]
            .value_counts()
            .items()
        }
        telemetry = BatchTelemetry(
            row_count=len(predictions),
            predicted_class_counts=class_counts,
            mean_max_probability=mean_max_probability,
            drift_alerts=list(drift.get("alerts", [])),
            duration_milliseconds=duration_milliseconds,
        )
        LOGGER.info(
            "prediction_batch model_run_id=%s rows=%s duration_ms=%.3f "
            "mean_max_probability=%s drift_alerts=%s class_counts=%s",
            service.manifest["run_id"],
            telemetry.row_count,
            telemetry.duration_milliseconds,
            telemetry.mean_max_probability,
            telemetry.drift_alerts,
            telemetry.predicted_class_counts,
        )
        records = predictions.to_dict(orient="records")
        return PredictionResponse(
            model_run_id=str(service.manifest["run_id"]),
            predictions=records,
            telemetry=telemetry,
        )

    return app


def _ready_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="approved model artifacts are not ready",
        )
    return service


def _validate_promotion_report(report_path: Path, run_directory: Path) -> None:
    import hashlib

    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    report_manifest_path = report_path.parent / "manifest.json"
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    if (
        hashlib.sha256(report_bytes).hexdigest()
        != report_manifest.get("promotion_report_sha256")
        or len(report_bytes) != report_manifest.get("promotion_report_byte_size")
    ):
        raise ValueError("promotion report integrity mismatch")
    manifest = json.loads(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )
    evaluation = report.get("evaluation", {})
    if evaluation.get("decision") != "approved":
        raise ValueError("promotion report does not approve the model")
    if evaluation.get("technically_approved") is not True:
        raise ValueError("promotion report technical gates are not approved")
    if evaluation.get("source_run_id") != manifest.get("run_id"):
        raise ValueError("promotion report belongs to a different model run")


def _environment_boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _environment_integer(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


app = create_app()
