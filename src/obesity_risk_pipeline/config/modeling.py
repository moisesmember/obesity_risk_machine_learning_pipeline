"""Typed configuration for the governed baseline-training slice."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from obesity_risk_pipeline.config.settings import load_ingestion_settings


@dataclass(frozen=True, slots=True)
class ModelingSettings:
    """Validated split and artifact settings for one immutable dataset snapshot."""

    dataset_path: Path
    expected_sha256: str
    output_root: Path
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42
    feature_set_version: str = "v1"

    def __post_init__(self) -> None:
        dataset_path = self.dataset_path.expanduser().resolve()
        output_root = self.output_root.expanduser().resolve()
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be lower than 1")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        if re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256) is None:
            raise ValueError(
                "expected_sha256 must contain exactly 64 hexadecimal characters"
            )

        object.__setattr__(self, "dataset_path", dataset_path)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "expected_sha256", self.expected_sha256.lower())


def load_modeling_settings(project_root: Path | None = None) -> ModelingSettings:
    """Load provisional, centralized modeling defaults with environment overrides."""

    root = (project_root or Path.cwd()).expanduser().resolve()
    ingestion = load_ingestion_settings(root)
    default_dataset_path = (
        ingestion.raw_root / ingestion.expected_sha256 / ingestion.expected_filename
    )
    return ModelingSettings(
        dataset_path=Path(
            os.getenv("OBESITY_MODELING_DATASET_PATH", str(default_dataset_path))
        ),
        expected_sha256=ingestion.expected_sha256,
        output_root=Path(
            os.getenv("OBESITY_MODELING_OUTPUT_ROOT", str(root / "artifacts" / "runs"))
        ),
        test_size=float(os.getenv("OBESITY_TEST_SIZE", "0.20")),
        validation_size=float(os.getenv("OBESITY_VALIDATION_SIZE", "0.20")),
        random_state=int(os.getenv("OBESITY_RANDOM_STATE", "42")),
        feature_set_version=os.getenv("OBESITY_FEATURE_SET_VERSION", "v1"),
    )
