"""External dataset ingestion boundary."""

from obesity_risk_pipeline.ingestion.kaggle import KaggleApiDownloader
from obesity_risk_pipeline.ingestion.service import (
    IngestionError,
    IngestionResult,
    KaggleIngestionService,
)

__all__ = [
    "IngestionError",
    "IngestionResult",
    "KaggleApiDownloader",
    "KaggleIngestionService",
]
