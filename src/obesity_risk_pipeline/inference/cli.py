"""Batch prediction command for a governed run directory."""

from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlparse

import pandas as pd

from obesity_risk_pipeline.config import load_minio_settings
from obesity_risk_pipeline.inference.service import PredictionService
from obesity_risk_pipeline.storage import MinioDatasetStore


LOGGER = logging.getLogger(__name__)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_TARGET_COLUMNS = ("0be1dad", "NObeyesdad")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run governed batch inference.")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="Local CSV path or s3://bucket/object.csv stored in MinIO",
    )
    parser.add_argument(
        "--input-sha256",
        help="Optional expected SHA-256 for a MinIO input object",
    )
    parser.add_argument(
        "--drop-target",
        action="store_true",
        help="Remove 0be1dad/NObeyesdad from an explicitly labeled scoring batch",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        frame = _load_input_frame(args.input, args.input_sha256)
        if args.drop_target:
            frame = _drop_target(frame)
        service = PredictionService.load(args.run_directory)
        predictions = service.predict(frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Inference failed: %s", exc)
        return 1
    logging.info("Predictions written to %s", args.output.resolve())
    return 0


def _load_input_frame(location: str, expected_sha256: str | None) -> pd.DataFrame:
    """Load a CSV from the local filesystem or the configured MinIO endpoint."""

    if expected_sha256 is not None and not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(
            "--input-sha256 must contain exactly 64 hexadecimal characters"
        )
    if not location.lower().startswith("s3://"):
        if "://" in location:
            raise ValueError("--input supports only local paths or s3:// URIs")
        if expected_sha256 is not None:
            raise ValueError("--input-sha256 is supported only for s3:// inputs")
        return pd.read_csv(Path(location))

    parsed = urlparse(location)
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError("invalid MinIO URI; expected s3://bucket/object.csv")
    if parsed.query or parsed.fragment:
        raise ValueError("MinIO input URI must not contain query or fragment values")
    object_name = unquote(parsed.path).strip("/")
    if not object_name:
        raise ValueError("MinIO input URI must include an object name")

    settings = load_minio_settings()
    if parsed.netloc != settings.bucket:
        raise ValueError(
            f"MinIO URI bucket {parsed.netloc!r} differs from configured bucket "
            f"{settings.bucket!r}"
        )
    governed_sha256 = expected_sha256 or _sha256_from_object_path(object_name)
    payload = MinioDatasetStore(settings).read_object(
        object_name,
        expected_sha256=governed_sha256,
    )
    LOGGER.info(
        "Inference input loaded from MinIO bucket=%s object=%s sha256_verified=%s",
        settings.bucket,
        object_name,
        governed_sha256 is not None,
    )
    return pd.read_csv(io.BytesIO(payload))


def _sha256_from_object_path(object_name: str) -> str | None:
    """Use a hash-addressed path component as the governed expected digest."""

    for component in reversed(object_name.split("/")):
        if _SHA256_PATTERN.fullmatch(component):
            return component.lower()
    return None


def _drop_target(frame: pd.DataFrame) -> pd.DataFrame:
    target_columns = [name for name in _TARGET_COLUMNS if name in frame.columns]
    if not target_columns:
        raise ValueError(
            "--drop-target was requested, but no supported target column was found"
        )
    LOGGER.info("Dropping target columns from scoring batch: %s", target_columns)
    return frame.drop(columns=target_columns)


if __name__ == "__main__":
    raise SystemExit(main())
