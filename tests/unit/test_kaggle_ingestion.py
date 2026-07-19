from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from obesity_risk_pipeline.config.settings import IngestionSettings
from obesity_risk_pipeline.data.raw_contract import RAW_COLUMNS, RAW_TARGET_VALUES
from obesity_risk_pipeline.ingestion.kaggle import DatasetDownloadError
from obesity_risk_pipeline.ingestion.service import IngestionError, KaggleIngestionService


def _valid_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, target in enumerate(sorted(RAW_TARGET_VALUES)):
        rows.append(
            {
                "id": str(index),
                "Gender": "Female" if index % 2 == 0 else "Male",
                "Age": str(18 + index),
                "Height": f"{1.55 + index * 0.03:.2f}",
                "Weight": str(50 + index * 10),
                "family_history_with_overweight": str(index % 2),
                "FAVC": "1",
                "FCVC": str(1 + index % 3),
                "NCP": str(1 + index % 4),
                "CAEC": ("0", "Sometimes", "Frequently", "Always")[index % 4],
                "SMOKE": "0",
                "CH2O": str(1 + index % 3),
                "SCC": "0",
                "FAF": str(index % 4),
                "TUE": str(index % 3),
                "CALC": ("0", "Sometimes", "Frequently")[index % 3],
                "MTRANS": (
                    "Public_Transportation",
                    "Automobile",
                    "Walking",
                    "Motorbike",
                    "Bike",
                )[index % 5],
                "0be1dad": target,
            }
        )
    return rows


def _csv_payload(
    rows: list[dict[str, str]],
    columns: tuple[str, ...] = RAW_COLUMNS,
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in columns})
    return stream.getvalue().encode("utf-8")


class FakeDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.call_count = 0

    def download(self, dataset_slug: str, destination: Path) -> None:
        assert dataset_slug == "jpkochar/obesity-risk-dataset"
        self.call_count += 1
        (destination / "obesity_level.csv").write_bytes(self.payload)


class FailingDownloader:
    def download(self, dataset_slug: str, destination: Path) -> None:
        (destination / "partial.zip").write_bytes(b"partial")
        raise DatasetDownloadError("controlled download failure")


class UnexpectedDownloaderCall:
    def download(self, dataset_slug: str, destination: Path) -> None:
        raise AssertionError("idempotent ingestion must not download an existing snapshot")


def _settings(tmp_path: Path, expected_sha256: str) -> IngestionSettings:
    return IngestionSettings(
        dataset_slug="jpkochar/obesity-risk-dataset",
        expected_filename="obesity_level.csv",
        expected_sha256=expected_sha256,
        raw_root=tmp_path / "raw",
        staging_root=tmp_path / "staging",
    )


def test_ingestion_validates_and_atomically_publishes_snapshot(tmp_path: Path) -> None:
    payload = _csv_payload(_valid_rows())
    expected_hash = hashlib.sha256(payload).hexdigest()
    downloader = FakeDownloader(payload)

    result = KaggleIngestionService(
        settings=_settings(tmp_path, expected_hash),
        downloader=downloader,
    ).run()

    assert downloader.call_count == 1
    assert result.sha256 == expected_hash
    assert result.row_count == 7
    assert result.reused_existing_snapshot is False
    assert result.dataset_path.read_bytes() == payload
    assert result.dataset_path.parent.name == expected_hash
    assert list((tmp_path / "staging").iterdir()) == []

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["dataset_slug"] == "jpkochar/obesity-risk-dataset"
    assert manifest["file"]["sha256"] == expected_hash
    assert manifest["quality"]["row_count"] == 7
    assert set(manifest["quality"]["target_counts_raw"]) == RAW_TARGET_VALUES
    assert "ingested_at_utc" in manifest


def test_ingestion_reuses_existing_snapshot_without_downloading(tmp_path: Path) -> None:
    payload = _csv_payload(_valid_rows())
    expected_hash = hashlib.sha256(payload).hexdigest()
    settings = _settings(tmp_path, expected_hash)
    first = KaggleIngestionService(settings, FakeDownloader(payload)).run()
    original_bytes = first.dataset_path.read_bytes()

    second = KaggleIngestionService(settings, UnexpectedDownloaderCall()).run()

    assert second.reused_existing_snapshot is True
    assert second.dataset_path == first.dataset_path
    assert second.dataset_path.read_bytes() == original_bytes


def test_ingestion_rejects_tampered_existing_manifest(tmp_path: Path) -> None:
    payload = _csv_payload(_valid_rows())
    expected_hash = hashlib.sha256(payload).hexdigest()
    settings = _settings(tmp_path, expected_hash)
    first = KaggleIngestionService(settings, FakeDownloader(payload)).run()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["quality"]["row_count"] = 999
    first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IngestionError, match="integrity check failed"):
        KaggleIngestionService(settings, UnexpectedDownloaderCall()).run()


def test_hash_mismatch_does_not_publish_or_leave_staging(tmp_path: Path) -> None:
    payload = _csv_payload(_valid_rows())
    settings = _settings(tmp_path, "0" * 64)

    with pytest.raises(IngestionError, match="hash does not match"):
        KaggleIngestionService(settings, FakeDownloader(payload)).run()

    assert not settings.raw_root.exists()
    assert list(settings.staging_root.iterdir()) == []


def test_schema_mismatch_is_rejected_before_publication(tmp_path: Path) -> None:
    columns = tuple(column for column in RAW_COLUMNS if column != "Weight")
    payload = _csv_payload(_valid_rows(), columns)
    expected_hash = hashlib.sha256(payload).hexdigest()
    settings = _settings(tmp_path, expected_hash)

    with pytest.raises(IngestionError, match="raw schema mismatch"):
        KaggleIngestionService(settings, FakeDownloader(payload)).run()

    assert not settings.raw_root.exists()


def test_duplicate_identifier_is_rejected(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[1]["id"] = rows[0]["id"]
    payload = _csv_payload(rows)
    expected_hash = hashlib.sha256(payload).hexdigest()

    with pytest.raises(IngestionError, match="duplicate id"):
        KaggleIngestionService(
            _settings(tmp_path, expected_hash), FakeDownloader(payload)
        ).run()


def test_download_failure_cleans_isolated_staging(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "0" * 64)

    with pytest.raises(IngestionError, match="controlled download failure"):
        KaggleIngestionService(settings, FailingDownloader()).run()

    assert list(settings.staging_root.iterdir()) == []


def test_configuration_rejects_nested_raw_and_staging_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain one another"):
        IngestionSettings(
            dataset_slug="jpkochar/obesity-risk-dataset",
            expected_filename="obesity_level.csv",
            expected_sha256="0" * 64,
            raw_root=tmp_path / "data",
            staging_root=tmp_path / "data" / "staging",
        )
