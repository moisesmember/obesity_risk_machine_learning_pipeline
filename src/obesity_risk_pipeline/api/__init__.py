"""HTTP serving boundary for the governed prediction service."""

from obesity_risk_pipeline.api.app import create_app

__all__ = ["create_app"]
