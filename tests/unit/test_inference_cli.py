from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from obesity_risk_pipeline.config.minio import MinioSettings
from obesity_risk_pipeline.inference import cli


def _settings() -> MinioSettings:
    return MinioSettings(
        endpoint="localhost:9000",
        access_key="access",
        secret_key="secret",
        bucket="obesity-risk-datasets",
    )


def test_load_input_frame_preserves_local_csv_compatibility(tmp_path: Path) -> None:
    input_path = tmp_path / "inference.csv"
    input_path.write_text("id,Age\n1,25\n", encoding="utf-8")

    frame = cli._load_input_frame(str(input_path), None)

    assert frame.to_dict(orient="records") == [{"id": 1, "Age": 25}]


def test_load_input_frame_reads_hash_addressed_minio_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"id,Age,0be1dad\n1,25,Normal_Weight\n"
    sha256 = hashlib.sha256(payload).hexdigest()
    observed: dict[str, str | None] = {}

    class FakeStore:
        def __init__(self, settings: MinioSettings) -> None:
            assert settings.bucket == "obesity-risk-datasets"

        def read_object(
            self, object_name: str, *, expected_sha256: str | None = None
        ) -> bytes:
            observed["object_name"] = object_name
            observed["expected_sha256"] = expected_sha256
            return payload

    monkeypatch.setattr(cli, "load_minio_settings", _settings)
    monkeypatch.setattr(cli, "MinioDatasetStore", FakeStore)
    object_name = f"datasets/obesity_risk_dataset/{sha256}/obesity_level.csv"

    frame = cli._load_input_frame(
        f"s3://obesity-risk-datasets/{object_name}", None
    )

    assert observed == {
        "object_name": object_name,
        "expected_sha256": sha256,
    }
    assert list(frame.columns) == ["id", "Age", "0be1dad"]


def test_load_input_frame_rejects_bucket_outside_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_minio_settings", _settings)

    with pytest.raises(ValueError, match="differs from configured bucket"):
        cli._load_input_frame("s3://another-bucket/input.csv", None)


def test_drop_target_is_explicit_and_never_passes_label_to_model() -> None:
    frame = pd.DataFrame(
        {"id": [1], "Age": [25], "0be1dad": ["Normal_Weight"]}
    )

    inference = cli._drop_target(frame)

    assert list(inference.columns) == ["id", "Age"]
    assert "0be1dad" in frame.columns


def test_drop_target_fails_when_labeled_column_is_absent() -> None:
    with pytest.raises(ValueError, match="no supported target column"):
        cli._drop_target(pd.DataFrame({"id": [1], "Age": [25]}))
