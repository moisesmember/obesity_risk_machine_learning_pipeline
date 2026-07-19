"""Validated MinIO settings shared by notebooks and storage adapters."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class MinioSettings:
    """Connection and namespace settings for governed dataset objects."""

    endpoint: str
    access_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    bucket: str = "fraud-detection"
    prefix: str = "datasets/obesity_risk_dataset"
    secure: bool = False
    auto_create_bucket: bool = True

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip().rstrip("/")
        if not endpoint or "://" in endpoint:
            raise ValueError(
                "MinIO endpoint must use 'host:port' format without a URL scheme"
            )
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO access_key and secret_key must not be empty")
        if not _BUCKET_PATTERN.fullmatch(self.bucket):
            raise ValueError("MinIO bucket must follow S3-compatible naming rules")

        prefix = self.prefix.strip("/")
        if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise ValueError("MinIO dataset prefix must be a safe non-empty object path")

        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "prefix", prefix)


def _read_boolean(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"environment variable {name} must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)!r}"
    )


def load_minio_settings() -> MinioSettings:
    """Load MinIO settings with defaults matching the local Docker Compose service."""

    access_key = (
        os.getenv("MINIO_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_USER")
        or "minioadmin"
    )
    secret_key = (
        os.getenv("MINIO_SECRET_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
        or "minioadmin"
    )
    return MinioSettings(
        endpoint=os.getenv("MINIO_ENDPOINT") or "localhost:9000",
        access_key=access_key,
        secret_key=secret_key,
        bucket=os.getenv("MINIO_DATASET_BUCKET") or "fraud-detection",
        prefix=(
            os.getenv("MINIO_DATASET_PREFIX") or "datasets/obesity_risk_dataset"
        ),
        secure=_read_boolean("MINIO_SECURE", False),
        auto_create_bucket=_read_boolean("MINIO_AUTO_CREATE_BUCKET", True),
    )
