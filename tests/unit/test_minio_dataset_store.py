from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from obesity_risk_pipeline.config.minio import MinioSettings, load_minio_settings
from obesity_risk_pipeline.storage.minio import MinioDatasetStore, ObjectStorageError


class MissingObjectError(Exception):
    code = "NoSuchKey"


class FakeResponse(io.BytesIO):
    def release_conn(self) -> None:
        return None


class FakeMinioClient:
    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.upload_count = 0

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.add(bucket_name)

    def stat_object(self, bucket_name: str, object_name: str) -> SimpleNamespace:
        try:
            stored = self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise MissingObjectError(object_name) from exc
        return SimpleNamespace(
            size=len(stored["payload"]),
            metadata=stored["metadata"],
        )

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        self.upload_count += 1
        self.objects[(bucket_name, object_name)] = {
            "payload": Path(file_path).read_bytes(),
            "content_type": content_type,
            "metadata": {
                f"x-amz-meta-{key}": value for key, value in metadata.items()
            },
        }

    def get_object(self, bucket_name: str, object_name: str) -> FakeResponse:
        try:
            payload = self.objects[(bucket_name, object_name)]["payload"]
        except KeyError as exc:
            raise MissingObjectError(object_name) from exc
        return FakeResponse(payload)


def _settings() -> MinioSettings:
    return MinioSettings(
        endpoint="localhost:9000",
        access_key="local-access",
        secret_key="local-secret",
        bucket="obesity-risk-datasets",
        prefix="datasets/obesity_risk_dataset",
        secure=False,
        auto_create_bucket=True,
    )


def _local_snapshot(tmp_path: Path) -> tuple[Path, Path, str]:
    dataset_path = tmp_path / "obesity_level.csv"
    manifest_path = tmp_path / "manifest.json"
    dataset_path.write_bytes(b"id,target\n1,Normal_Weight\n")
    manifest_path.write_text('{"manifest_schema_version": 1}\n', encoding="utf-8")
    sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return dataset_path, manifest_path, sha256


def test_store_uploads_missing_snapshot_and_reuses_matching_objects(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path, sha256 = _local_snapshot(tmp_path)
    client = FakeMinioClient()
    store = MinioDatasetStore(_settings(), client=client)

    first = store.ensure_snapshot(dataset_path, manifest_path, sha256)
    second = store.ensure_snapshot(dataset_path, manifest_path, sha256)

    assert first.reused_existing_objects is False
    assert second.reused_existing_objects is True
    assert client.upload_count == 2
    assert first.bucket == "obesity-risk-datasets"
    assert first.dataset_object.endswith(f"/{sha256}/obesity_level.csv")
    assert store.read_verified_dataset(first.dataset_object, sha256) == (
        dataset_path.read_bytes()
    )


def test_store_refuses_to_overwrite_conflicting_object(tmp_path: Path) -> None:
    dataset_path, manifest_path, sha256 = _local_snapshot(tmp_path)
    client = FakeMinioClient()
    store = MinioDatasetStore(_settings(), client=client)
    snapshot = store.ensure_snapshot(dataset_path, manifest_path, sha256)
    client.objects[(snapshot.bucket, snapshot.dataset_object)]["metadata"] = {
        "x-amz-meta-sha256": "0" * 64,
        "x-amz-meta-dataset-sha256": sha256,
    }

    with pytest.raises(ObjectStorageError, match="conflicts with immutable snapshot"):
        store.ensure_snapshot(dataset_path, manifest_path, sha256)

    assert client.upload_count == 2


def test_store_detects_corrupted_downloaded_bytes(tmp_path: Path) -> None:
    dataset_path, manifest_path, sha256 = _local_snapshot(tmp_path)
    client = FakeMinioClient()
    store = MinioDatasetStore(_settings(), client=client)
    snapshot = store.ensure_snapshot(dataset_path, manifest_path, sha256)
    client.objects[(snapshot.bucket, snapshot.dataset_object)]["payload"] = b"tampered"

    with pytest.raises(ObjectStorageError, match="integrity check failed"):
        store.read_verified_dataset(snapshot.dataset_object, sha256)


def test_store_requires_existing_bucket_when_auto_creation_is_disabled(
    tmp_path: Path,
) -> None:
    dataset_path, manifest_path, sha256 = _local_snapshot(tmp_path)
    settings = MinioSettings(
        endpoint="localhost:9000",
        access_key="local-access",
        secret_key="local-secret",
        bucket="obesity-risk-datasets",
        prefix="datasets/obesity_risk_dataset",
        auto_create_bucket=False,
    )

    with pytest.raises(ObjectStorageError, match="does not exist"):
        MinioDatasetStore(settings, client=FakeMinioClient()).ensure_snapshot(
            dataset_path, manifest_path, sha256
        )


def test_default_settings_match_the_local_compose_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable_name in (
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_ROOT_USER",
        "MINIO_SECRET_KEY",
        "MINIO_ROOT_PASSWORD",
        "MINIO_DATASET_BUCKET",
        "MINIO_DATASET_PREFIX",
        "MINIO_SECURE",
        "MINIO_AUTO_CREATE_BUCKET",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    settings = load_minio_settings()

    assert settings.endpoint == "localhost:9000"
    assert settings.bucket == "obesity-risk-datasets"
    assert settings.prefix == "datasets/obesity_risk_dataset"
