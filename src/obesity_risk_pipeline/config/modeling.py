"""Typed configuration for governed obesity-model experimentation."""

from __future__ import annotations

import os
import re
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from obesity_risk_pipeline.config.settings import load_ingestion_settings


@dataclass(frozen=True, slots=True)
class NumericBounds:
    """Inclusive numeric contract used by training and inference validation."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum >= self.maximum:
            raise ValueError("numeric bound minimum must be lower than maximum")


DEFAULT_NUMERIC_BOUNDS: dict[str, NumericBounds] = {
    "Age": NumericBounds(0.0, 120.0),
    "Height": NumericBounds(0.5, 2.5),
    "Weight": NumericBounds(10.0, 500.0),
    "FCVC": NumericBounds(1.0, 3.0),
    "NCP": NumericBounds(1.0, 4.0),
    "CH2O": NumericBounds(1.0, 3.0),
    "FAF": NumericBounds(0.0, 3.0),
    "TUE": NumericBounds(0.0, 2.0),
}


@dataclass(frozen=True, slots=True)
class ModelingSettings:
    """Validated split, CV, tracking and artifact settings."""

    dataset_path: Path
    expected_sha256: str
    output_root: Path
    holdout_size: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    feature_set_version: str = "v2"
    target_proportion_tolerance: float = 0.03
    distribution_psi_threshold: float = 0.25
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "obesity-risk-multiclass"
    optuna_trials: int = 0
    numeric_bounds: Mapping[str, NumericBounds] = field(
        default_factory=lambda: dict(DEFAULT_NUMERIC_BOUNDS)
    )

    def __post_init__(self) -> None:
        dataset_path = self.dataset_path.expanduser().resolve()
        output_root = self.output_root.expanduser().resolve()
        if not 0.0 < self.holdout_size < 0.5:
            raise ValueError("holdout_size must be between 0 and 0.5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version must not be empty")
        if re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256) is None:
            raise ValueError(
                "expected_sha256 must contain exactly 64 hexadecimal characters"
            )
        if not 0.0 <= self.target_proportion_tolerance <= 1.0:
            raise ValueError("target_proportion_tolerance must be between 0 and 1")
        if self.distribution_psi_threshold <= 0.0:
            raise ValueError("distribution_psi_threshold must be positive")
        if self.optuna_trials < 0:
            raise ValueError("optuna_trials must be non-negative")
        if set(self.numeric_bounds) != set(DEFAULT_NUMERIC_BOUNDS):
            raise ValueError(
                "numeric_bounds must define every governed numeric feature"
            )

        object.__setattr__(self, "dataset_path", dataset_path)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "expected_sha256", self.expected_sha256.lower())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return normalized in {"true", "1", "yes"}


def load_modeling_settings(project_root: Path | None = None) -> ModelingSettings:
    """Load centralized defaults with explicit environment overrides."""

    root = (project_root or Path.cwd()).expanduser().resolve()
    ingestion = load_ingestion_settings(root)
    defaults = _load_file_defaults(root / "configs" / "modeling.json")
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
        holdout_size=float(os.getenv("OBESITY_HOLDOUT_SIZE", defaults["holdout_size"])),
        cv_folds=int(os.getenv("OBESITY_CV_FOLDS", defaults["cv_folds"])),
        random_state=int(os.getenv("OBESITY_RANDOM_STATE", defaults["random_state"])),
        feature_set_version=os.getenv(
            "OBESITY_FEATURE_SET_VERSION", defaults["feature_set_version"]
        ),
        target_proportion_tolerance=float(
            os.getenv(
                "OBESITY_TARGET_PROPORTION_TOLERANCE",
                defaults["target_proportion_tolerance"],
            )
        ),
        distribution_psi_threshold=float(
            os.getenv(
                "OBESITY_DISTRIBUTION_PSI_THRESHOLD",
                defaults["distribution_psi_threshold"],
            )
        ),
        mlflow_enabled=_env_bool("OBESITY_MLFLOW_ENABLED", False),
        mlflow_tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        ),
        mlflow_experiment_name=os.getenv(
            "OBESITY_MLFLOW_EXPERIMENT", "obesity-risk-multiclass"
        ),
        optuna_trials=int(
            os.getenv("OBESITY_OPTUNA_TRIALS", defaults["optuna_trials"])
        ),
        numeric_bounds={
            name: NumericBounds(float(bounds["minimum"]), float(bounds["maximum"]))
            for name, bounds in defaults["numeric_bounds"].items()
        },
    )


def _load_file_defaults(path: Path) -> dict[str, object]:
    defaults: dict[str, object] = {
        "holdout_size": 0.20,
        "cv_folds": 5,
        "random_state": 42,
        "feature_set_version": "v2",
        "target_proportion_tolerance": 0.03,
        "distribution_psi_threshold": 0.25,
        "optuna_trials": 0,
        "numeric_bounds": {
            name: {"minimum": bounds.minimum, "maximum": bounds.maximum}
            for name, bounds in DEFAULT_NUMERIC_BOUNDS.items()
        },
    }
    if not path.is_file():
        return defaults
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"modeling config must be a JSON object: {path}")
    defaults.update(payload)
    return defaults
