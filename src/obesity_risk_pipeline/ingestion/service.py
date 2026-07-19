"""Orchestration for validated, immutable and idempotent dataset ingestion."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from obesity_risk_pipeline.config.settings import IngestionSettings
from obesity_risk_pipeline.data.raw_contract import (
    DatasetProfile,
    DatasetValidationError,
    RawDatasetContract,
    validate_raw_dataset,
)
from obesity_risk_pipeline.ingestion.kaggle import (
    DatasetDownloadError,
    DatasetDownloader,
)

LOGGER = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when an ingestion run cannot produce a governed raw snapshot."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Published paths and lineage returned by a successful ingestion."""

    dataset_path: Path
    manifest_path: Path
    sha256: str
    row_count: int
    reused_existing_snapshot: bool


class KaggleIngestionService:
    """Ingest one hash-pinned Kaggle dataset without mutating previous snapshots."""

    def __init__(
        self,
        settings: IngestionSettings,
        downloader: DatasetDownloader,
        contract: RawDatasetContract | None = None,
    ) -> None:
        self._settings = settings
        self._downloader = downloader
        self._contract = contract or RawDatasetContract()

    def run(self) -> IngestionResult:
        """Download, validate, manifest and atomically publish the configured snapshot."""

        try:
            existing = self._reuse_expected_snapshot_if_present()
        except DatasetValidationError as exc:
            raise IngestionError(str(exc)) from exc
        if existing is not None:
            LOGGER.info(
                "Governed dataset already exists and is valid at %s; "
                "skipping Kaggle download",
                existing.dataset_path,
            )
            return existing

        LOGGER.info(
            "Governed dataset is absent; importing %s from Kaggle",
            self._settings.dataset_slug,
        )
        self._settings.staging_root.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(
            tempfile.mkdtemp(prefix="kaggle-ingest-", dir=self._settings.staging_root)
        )
        try:
            self._downloader.download(
                self._settings.dataset_slug,
                staging_directory,
            )
            downloaded_file = self._locate_downloaded_file(staging_directory)
            profile = validate_raw_dataset(downloaded_file, self._contract)
            self._validate_expected_hash(profile)
            return self._publish(downloaded_file, profile)
        except (DatasetDownloadError, DatasetValidationError) as exc:
            raise IngestionError(str(exc)) from exc
        finally:
            self._remove_staging_directory(staging_directory)

    def _reuse_expected_snapshot_if_present(self) -> IngestionResult | None:
        snapshot_directory = self._snapshot_directory(self._settings.expected_sha256)
        if not snapshot_directory.exists():
            return None
        return self._load_existing_snapshot(snapshot_directory)

    def _locate_downloaded_file(self, staging_directory: Path) -> Path:
        matches = list(staging_directory.rglob(self._settings.expected_filename))
        if len(matches) != 1:
            raise IngestionError(
                f"expected exactly one {self._settings.expected_filename!r} in staging, "
                f"found {len(matches)}"
            )
        return matches[0]

    def _validate_expected_hash(self, profile: DatasetProfile) -> None:
        if profile.sha256 != self._settings.expected_sha256:
            raise IngestionError(
                "downloaded dataset hash does not match the governed snapshot; "
                f"expected={self._settings.expected_sha256}, actual={profile.sha256}. "
                "Review the new source version and update the contract intentionally."
            )

    def _snapshot_directory(self, sha256: str) -> Path:
        return self._settings.raw_root / sha256

    def _publish(
        self,
        downloaded_file: Path,
        profile: DatasetProfile,
    ) -> IngestionResult:
        self._settings.raw_root.mkdir(parents=True, exist_ok=True)
        snapshot_directory = self._snapshot_directory(profile.sha256)
        if snapshot_directory.exists():
            return self._load_existing_snapshot(snapshot_directory)

        publication_directory = self._settings.raw_root / (
            f".{profile.sha256}.publishing-{uuid4().hex}"
        )
        publication_directory.mkdir(parents=False, exist_ok=False)
        try:
            dataset_path = publication_directory / self._settings.expected_filename
            shutil.copy2(downloaded_file, dataset_path)
            manifest_path = publication_directory / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    self._build_manifest(profile),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(publication_directory, snapshot_directory)
            except FileExistsError:
                LOGGER.info("Snapshot was published concurrently; reusing it")
            if publication_directory.exists():
                shutil.rmtree(publication_directory)
        except Exception:
            if publication_directory.exists():
                shutil.rmtree(publication_directory)
            raise

        return self._load_existing_snapshot(snapshot_directory, reused=False)

    def _build_manifest(self, profile: DatasetProfile) -> dict[str, object]:
        return {
            "manifest_schema_version": 1,
            "source": {
                "provider": "kaggle",
                "dataset_slug": self._settings.dataset_slug,
                "url": (
                    "https://www.kaggle.com/datasets/"
                    f"{self._settings.dataset_slug}"
                ),
            },
            "ingested_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "dataset_version": profile.sha256,
            "file": {
                "name": self._settings.expected_filename,
                "sha256": profile.sha256,
                "byte_size": profile.byte_size,
            },
            "schema": {
                "column_count": len(profile.columns),
                "columns": list(profile.columns),
                "target_column_raw": self._contract.target_column,
            },
            "quality": {
                "row_count": profile.row_count,
                "target_counts_raw": profile.target_counts,
            },
        }

    def _load_existing_snapshot(
        self,
        snapshot_directory: Path,
        *,
        reused: bool = True,
    ) -> IngestionResult:
        dataset_path = snapshot_directory / self._settings.expected_filename
        manifest_path = snapshot_directory / "manifest.json"
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise IngestionError(
                f"incomplete immutable snapshot found at {snapshot_directory}"
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestionError(
                f"invalid snapshot manifest at {manifest_path}"
            ) from exc

        profile = validate_raw_dataset(dataset_path, self._contract)
        file_metadata = manifest.get("file")
        schema_metadata = manifest.get("schema")
        quality_metadata = manifest.get("quality")
        if not all(
            isinstance(section, dict)
            for section in (file_metadata, schema_metadata, quality_metadata)
        ):
            raise IngestionError(
                f"snapshot manifest has invalid sections at {manifest_path}"
            )

        manifest_hash = file_metadata.get("sha256")
        if (
            profile.sha256 != self._settings.expected_sha256
            or manifest_hash != profile.sha256
            or snapshot_directory.name != profile.sha256
            or manifest.get("dataset_version") != profile.sha256
            or file_metadata.get("byte_size") != profile.byte_size
            or schema_metadata.get("columns") != list(profile.columns)
            or quality_metadata.get("row_count") != profile.row_count
            or quality_metadata.get("target_counts_raw") != profile.target_counts
        ):
            raise IngestionError(
                f"immutable snapshot integrity check failed at {snapshot_directory}"
            )

        return IngestionResult(
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            sha256=profile.sha256,
            row_count=profile.row_count,
            reused_existing_snapshot=reused,
        )

    def _remove_staging_directory(self, staging_directory: Path) -> None:
        resolved_staging = staging_directory.resolve()
        if self._settings.staging_root not in resolved_staging.parents:
            raise IngestionError(
                f"refusing to remove staging path outside configured root: {resolved_staging}"
            )
        shutil.rmtree(resolved_staging, ignore_errors=True)
