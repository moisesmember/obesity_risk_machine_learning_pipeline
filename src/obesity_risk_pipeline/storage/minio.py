"""MinIO adapter for immutable, hash-addressed dataset snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from obesity_risk_pipeline.config.minio import MinioSettings
from obesity_risk_pipeline.data.raw_contract import calculate_sha256


class ObjectStorageError(RuntimeError):
    """Raised when a governed object cannot be safely stored or read."""


class MinioClient(Protocol):
    """Subset of the MinIO client used by this adapter."""

    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def stat_object(self, bucket_name: str, object_name: str) -> Any: ...

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> Any: ...

    def get_object(self, bucket_name: str, object_name: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class RemoteDatasetSnapshot:
    """Immutable object names created or verified in MinIO."""

    bucket: str
    dataset_object: str
    manifest_object: str
    sha256: str
    reused_existing_objects: bool


class MinioDatasetStore:
    """Publish and read immutable dataset snapshots without silent overwrites."""

    def __init__(
        self,
        settings: MinioSettings,
        client: MinioClient | None = None,
    ) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
            return

        try:
            from minio import Minio
        except ImportError as exc:
            raise ObjectStorageError(
                "the 'minio' dependency is unavailable; install requirements.txt"
            ) from exc

        self._client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )

    def ensure_snapshot(
        self,
        dataset_path: Path,
        manifest_path: Path,
        expected_sha256: str,
    ) -> RemoteDatasetSnapshot:
        """Create missing objects or verify that existing objects match local files."""

        if not dataset_path.is_file() or not manifest_path.is_file():
            raise ObjectStorageError(
                "local governed dataset and manifest must exist before MinIO publication"
            )
        actual_sha256 = calculate_sha256(dataset_path)
        if actual_sha256 != expected_sha256:
            raise ObjectStorageError(
                "local dataset hash differs from the governed version; refusing upload"
            )

        self._ensure_bucket()
        version_prefix = f"{self._settings.prefix}/{expected_sha256}"
        dataset_object = f"{version_prefix}/{dataset_path.name}"
        manifest_object = f"{version_prefix}/{manifest_path.name}"
        dataset_reused = self._ensure_object(
            local_path=dataset_path,
            object_name=dataset_object,
            dataset_sha256=expected_sha256,
            content_type="text/csv",
        )
        manifest_reused = self._ensure_object(
            local_path=manifest_path,
            object_name=manifest_object,
            dataset_sha256=expected_sha256,
            content_type="application/json",
        )

        return RemoteDatasetSnapshot(
            bucket=self._settings.bucket,
            dataset_object=dataset_object,
            manifest_object=manifest_object,
            sha256=expected_sha256,
            reused_existing_objects=dataset_reused and manifest_reused,
        )

    def read_verified_dataset(
        self,
        object_name: str,
        expected_sha256: str,
    ) -> bytes:
        """Read a dataset object and verify its content hash before returning bytes."""

        return self.read_object(object_name, expected_sha256=expected_sha256)

    def read_object(
        self,
        object_name: str,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        """Read one object, optionally enforcing a caller-provided SHA-256."""

        normalized_name = object_name.strip("/")
        if not normalized_name or any(
            part in {"", ".", ".."} for part in normalized_name.split("/")
        ):
            raise ObjectStorageError("MinIO object name must be a safe non-empty path")

        try:
            response = self._client.get_object(
                self._settings.bucket, normalized_name
            )
            try:
                payload = response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:
            raise ObjectStorageError(
                f"unable to read governed MinIO object {normalized_name!r}"
            ) from exc

        if (
            expected_sha256 is not None
            and hashlib.sha256(payload).hexdigest() != expected_sha256.lower()
        ):
            raise ObjectStorageError(
                f"MinIO object integrity check failed for {normalized_name!r}"
            )
        return payload

    def _ensure_bucket(self) -> None:
        try:
            exists = self._client.bucket_exists(self._settings.bucket)
            if exists:
                return
            if not self._settings.auto_create_bucket:
                raise ObjectStorageError(
                    f"MinIO bucket {self._settings.bucket!r} does not exist"
                )
            self._client.make_bucket(self._settings.bucket)
        except ObjectStorageError:
            raise
        except Exception as exc:
            raise ObjectStorageError(
                f"unable to initialize MinIO bucket {self._settings.bucket!r}"
            ) from exc

    def _ensure_object(
        self,
        *,
        local_path: Path,
        object_name: str,
        dataset_sha256: str,
        content_type: str,
    ) -> bool:
        file_sha256 = calculate_sha256(local_path)
        try:
            stat = self._client.stat_object(self._settings.bucket, object_name)
        except Exception as exc:
            if not self._is_missing_object(exc):
                raise ObjectStorageError(
                    f"unable to inspect MinIO object {object_name!r}"
                ) from exc
        else:
            metadata = {
                str(key).lower(): str(value)
                for key, value in (stat.metadata or {}).items()
            }
            stored_sha256 = metadata.get("x-amz-meta-sha256") or metadata.get(
                "sha256"
            )
            stored_dataset_sha256 = metadata.get(
                "x-amz-meta-dataset-sha256"
            ) or metadata.get("dataset-sha256")
            if (
                stat.size != local_path.stat().st_size
                or stored_sha256 != file_sha256
                or stored_dataset_sha256 != dataset_sha256
            ):
                raise ObjectStorageError(
                    f"existing MinIO object conflicts with immutable snapshot: "
                    f"{object_name!r}"
                )
            return True

        try:
            self._client.fput_object(
                self._settings.bucket,
                object_name,
                str(local_path),
                content_type=content_type,
                metadata={
                    "sha256": file_sha256,
                    "dataset-sha256": dataset_sha256,
                },
            )
        except Exception as exc:
            raise ObjectStorageError(
                f"unable to publish governed MinIO object {object_name!r}"
            ) from exc
        return False

    @staticmethod
    def _is_missing_object(exc: Exception) -> bool:
        return getattr(exc, "code", None) in {
            "NoSuchKey",
            "NoSuchObject",
            "NoSuchBucket",
        }
