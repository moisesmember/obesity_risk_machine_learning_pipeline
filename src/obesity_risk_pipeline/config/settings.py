"""Centralized and validated settings for dataset ingestion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_SLUG = "jpkochar/obesity-risk-dataset"
DEFAULT_DATASET_FILENAME = "obesity_level.csv"
DEFAULT_DATASET_SHA256 = (
    "04549179841220E7537EE9065FAC9CF9446C6368133882B7199A5618EA541EE6"
)

_DATASET_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    """Validated runtime settings for a single immutable Kaggle snapshot."""

    dataset_slug: str
    expected_filename: str
    expected_sha256: str
    raw_root: Path
    staging_root: Path

    def __post_init__(self) -> None:
        if not _DATASET_SLUG_PATTERN.fullmatch(self.dataset_slug):
            raise ValueError(
                "dataset_slug must use the '<owner>/<dataset>' Kaggle format"
            )

        filename = Path(self.expected_filename)
        if filename.name != self.expected_filename or filename.suffix.lower() != ".csv":
            raise ValueError("expected_filename must be a CSV basename")

        if not _SHA256_PATTERN.fullmatch(self.expected_sha256):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")

        raw_root = self.raw_root.expanduser().resolve()
        staging_root = self.staging_root.expanduser().resolve()
        if raw_root == staging_root:
            raise ValueError("raw_root and staging_root must be different directories")
        if raw_root in staging_root.parents or staging_root in raw_root.parents:
            raise ValueError("raw_root and staging_root must not contain one another")

        object.__setattr__(self, "expected_sha256", self.expected_sha256.lower())
        object.__setattr__(self, "raw_root", raw_root)
        object.__setattr__(self, "staging_root", staging_root)


def load_ingestion_settings(project_root: Path | None = None) -> IngestionSettings:
    """Load ingestion settings from safe defaults and optional environment overrides."""

    root = (project_root or Path.cwd()).expanduser().resolve()
    raw_root = Path(
        os.getenv(
            "OBESITY_DATASET_RAW_ROOT",
            str(root / "data" / "raw" / "obesity_risk_dataset"),
        )
    )
    staging_root = Path(
        os.getenv(
            "OBESITY_DATASET_STAGING_ROOT",
            str(root / "data" / "staging" / "obesity_risk_dataset"),
        )
    )

    return IngestionSettings(
        dataset_slug=os.getenv("OBESITY_DATASET_SLUG", DEFAULT_DATASET_SLUG),
        expected_filename=os.getenv(
            "OBESITY_DATASET_FILENAME", DEFAULT_DATASET_FILENAME
        ),
        expected_sha256=os.getenv(
            "OBESITY_DATASET_EXPECTED_SHA256", DEFAULT_DATASET_SHA256
        ),
        raw_root=raw_root,
        staging_root=staging_root,
    )
