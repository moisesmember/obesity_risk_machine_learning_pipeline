"""Kaggle-specific adapter kept outside the domain and data contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DatasetDownloadError(RuntimeError):
    """Raised when an external dataset cannot be downloaded safely."""


class DatasetDownloader(Protocol):
    """Port implemented by external dataset providers."""

    def download(self, dataset_slug: str, destination: Path) -> None:
        """Download and extract a dataset into an isolated staging directory."""


class KaggleApiDownloader:
    """Download public Kaggle datasets through the official Python client."""

    def download(self, dataset_slug: str, destination: Path) -> None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise DatasetDownloadError(
                "the 'kaggle' dependency is unavailable; install requirements.txt"
            ) from exc

        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as exc:
            raise DatasetDownloadError(
                "Kaggle authentication failed; configure KAGGLE_API_TOKEN or the "
                "official kaggle.json credential file outside the repository"
            ) from exc

        try:
            api.dataset_download_files(
                dataset=dataset_slug,
                path=str(destination),
                unzip=True,
                quiet=False,
            )
        except Exception as exc:
            raise DatasetDownloadError(
                f"Kaggle download failed for dataset {dataset_slug!r}"
            ) from exc
