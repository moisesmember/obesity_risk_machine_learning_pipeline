"""Strict public request and response schemas for model serving."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRecord(BaseModel):
    """One profile accepted by the current model contract."""

    model_config = ConfigDict(extra="forbid")

    id: int
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str


class PredictionRequest(BaseModel):
    """Bounded batch request; runtime configuration may impose a lower limit."""

    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRecord] = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    """Predictions with the immutable source run identifier."""

    model_run_id: str
    predictions: list[dict[str, Any]]
    telemetry: "BatchTelemetry"


class BatchTelemetry(BaseModel):
    """Non-row-level operational summaries safe for monitoring."""

    row_count: int
    predicted_class_counts: dict[str, int]
    mean_max_probability: float | None
    drift_alerts: list[str]
    duration_milliseconds: float


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None = None


class ModelInfoResponse(BaseModel):
    model_run_id: str
    selected_candidate: str
    feature_set_version: str
    promotion_required: bool
