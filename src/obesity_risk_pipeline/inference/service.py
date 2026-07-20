"""Load a governed model and produce traceable, probability-safe predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from obesity_risk_pipeline.config.modeling import NumericBounds
from obesity_risk_pipeline.data.modeling import (
    ID_COLUMN,
    MODEL_FEATURES,
    TARGET_CLASSES,
)
from obesity_risk_pipeline.data.validation import (
    DistributionProfile,
    assess_distribution_shift,
    validate_canonical_frame,
)
from obesity_risk_pipeline.models.evaluation import predict_probabilities


class PredictionService:
    """Validated inference facade around a complete serialized pipeline."""

    def __init__(
        self,
        estimator: BaseEstimator,
        manifest: dict[str, Any],
        profile: DistributionProfile,
    ) -> None:
        self.estimator = estimator
        self.manifest = manifest
        self.profile = profile

    @classmethod
    def load(cls, run_directory: Path) -> PredictionService:
        """Load and validate the required run artifacts."""

        root = run_directory.expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("manifest_schema_version") != 2:
            raise ValueError("unsupported model manifest schema")
        if tuple(manifest.get("input_features", ())) != MODEL_FEATURES:
            raise ValueError("model input schema is incompatible")
        if tuple(manifest.get("target_classes", ())) != TARGET_CLASSES:
            raise ValueError("model target schema is incompatible")
        _verify_artifact(root / "model.joblib", manifest)
        _verify_artifact(root / "distribution_profile.json", manifest)
        profile_payload = json.loads(
            (root / "distribution_profile.json").read_text(encoding="utf-8")
        )
        profile = DistributionProfile(**profile_payload)
        estimator = joblib.load(root / "model.joblib")
        return cls(estimator, manifest, profile)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict classes and ordered probabilities while preserving `id`."""

        canonical = _normalize_inference_frame(frame)
        validate_canonical_frame(
            canonical,
            None,
            require_target=False,
            allow_unknown_categories=True,
            numeric_bounds={
                name: NumericBounds(**bounds)
                for name, bounds in self.manifest["configuration"][
                    "numeric_bounds"
                ].items()
            },
        )
        features = canonical.loc[:, MODEL_FEATURES]
        predictions = np.asarray(self.estimator.predict(features), dtype=object)
        probabilities = predict_probabilities(self.estimator, features)
        output = pd.DataFrame(
            {ID_COLUMN: canonical[ID_COLUMN].to_numpy(), "predicted_class": predictions}
        )
        if probabilities is not None:
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                raise RuntimeError("predicted probabilities do not sum to one")
            for index, label in enumerate(TARGET_CLASSES):
                output[f"probability_{label}"] = probabilities[:, index]
        output["model_run_id"] = str(self.manifest["run_id"])
        return output

    def assess_drift(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Compare a prediction batch with the development distribution profile."""

        canonical = _normalize_inference_frame(frame)
        threshold = float(
            self.manifest["configuration"]["distribution_psi_threshold"]
        )
        return assess_distribution_shift(canonical, self.profile, threshold=threshold)


def _normalize_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    for column in ("family_history_with_overweight", "FAVC", "SMOKE", "SCC"):
        if column in canonical:
            canonical[column] = canonical[column].astype(str).replace(
                {"0": "No", "1": "Yes"}
            )
    for column in ("CAEC", "CALC"):
        if column in canonical:
            canonical[column] = canonical[column].astype(str).replace({"0": "No"})
    return canonical


def _verify_artifact(path: Path, manifest: dict[str, Any]) -> None:
    metadata = manifest.get("artifacts", {}).get(path.name)
    if not isinstance(metadata, dict):
        raise ValueError(f"artifact is absent from manifest: {path.name}")
    if not path.is_file():
        raise ValueError(f"required artifact does not exist: {path.name}")
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if (
        actual_hash != metadata.get("sha256")
        or len(payload) != metadata.get("byte_size")
    ):
        raise ValueError(f"artifact integrity mismatch: {path.name}")
