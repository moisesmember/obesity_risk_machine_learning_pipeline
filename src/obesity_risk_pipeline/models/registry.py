"""MLflow Model Registry adapter used only after governed approval."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from obesity_risk_pipeline.config.promotion import PromotionPolicy
from obesity_risk_pipeline.data.modeling import CATEGORY_DOMAINS, MODEL_FEATURES
from obesity_risk_pipeline.models.governance import (
    PromotionEvaluation,
    evaluate_run_for_promotion,
)


SKOPS_TRUSTED_TYPES = (
    "numpy.dtype",
    "obesity_risk_pipeline.features.engineering.FeatureEngineer",
    "obesity_risk_pipeline.models.baselines.BmiRuleClassifier",
    "obesity_risk_pipeline.models.catalog.OrderedTargetClassifier",
)


@dataclass(frozen=True, slots=True)
class RegistryResult:
    """Traceable outcome of logging and aliasing an approved model version."""

    status: str
    model_name: str
    version: str
    alias: str
    registry_run_id: str
    model_uri: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def register_approved_model(
    *,
    run_directory: Path,
    policy: PromotionPolicy,
    evaluation: PromotionEvaluation,
    alias: str,
    tracking_uri: str | None = None,
    experiment_name: str = "obesity-risk-promotions",
) -> RegistryResult:
    """Log, reload, register and alias a model only after every gate passes."""

    if evaluation.decision != "approved":
        raise ValueError("only an approved promotion evaluation can mutate the registry")
    if evaluation.approval is None:
        raise ValueError("registry mutation requires traceable human approval")
    if alias not in policy.allowed_aliases:
        raise ValueError(
            f"registry alias {alias!r} is not allowed by policy; "
            f"allowed={list(policy.allowed_aliases)!r}"
        )
    fresh_evaluation = evaluate_run_for_promotion(
        run_directory,
        policy,
        evaluation.approval,
    )
    if fresh_evaluation != evaluation or fresh_evaluation.decision != "approved":
        raise RuntimeError(
            "promotion source or policy changed after gate evaluation; "
            "evaluate the run again"
        )
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.exceptions import MlflowException
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "MLflow registry support is unavailable; install requirements-modeling.txt"
        ) from exc

    root = run_directory.expanduser().resolve()
    estimator = joblib.load(root / "model.joblib")
    manifest = _load_manifest(root / "manifest.json")
    input_example = _synthetic_input_example(manifest)
    expected = np.asarray(estimator.predict(input_example), dtype=object)
    signature = infer_signature(input_example, expected)
    active_tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    mlflow.set_tracking_uri(active_tracking_uri)
    mlflow.set_experiment(experiment_name)
    try:
        with mlflow.start_run(
            run_name=f"promote-{evaluation.source_run_id}",
            tags={
                "run_type": "model_promotion",
                "source_run_id": evaluation.source_run_id,
                "source_mlflow_run_id": evaluation.mlflow_run_id or "unavailable",
                "policy_version": evaluation.policy_version,
                "policy_sha256": evaluation.policy_sha256,
                "approved_by": evaluation.approval.approved_by,
                "approval_ticket": evaluation.approval.approval_ticket,
                "target_alias": alias,
            },
        ) as registry_run:
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator,
                name="model",
                serialization_format="skops",
                skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
                code_paths=[str(Path(__file__).resolve().parents[2])],
                signature=signature,
                input_example=input_example,
                metadata={
                    "source_run_id": evaluation.source_run_id,
                    "policy_sha256": evaluation.policy_sha256,
                },
            )
            loaded = mlflow.sklearn.load_model(model_info.model_uri)
            actual = np.asarray(loaded.predict(input_example), dtype=object)
            if not np.array_equal(expected, actual):
                raise RuntimeError(
                    "MLflow-loaded model predictions differ from the approved artifact"
                )
            version = mlflow.register_model(model_info.model_uri, policy.model_name)
            client = MlflowClient(tracking_uri=active_tracking_uri)
            client.set_model_version_tag(
                policy.model_name,
                version.version,
                "source_run_id",
                evaluation.source_run_id,
            )
            client.set_model_version_tag(
                policy.model_name,
                version.version,
                "approval_ticket",
                evaluation.approval.approval_ticket,
            )
            client.set_registered_model_alias(
                policy.model_name,
                alias,
                version.version,
            )
            result = RegistryResult(
                status="registered",
                model_name=policy.model_name,
                version=str(version.version),
                alias=alias,
                registry_run_id=registry_run.info.run_id,
                model_uri=str(model_info.model_uri),
            )
    except MlflowException as exc:
        raise RuntimeError(f"MLflow registry operation failed: {exc}") from exc
    return result


def _synthetic_input_example(manifest: dict[str, Any]) -> pd.DataFrame:
    bounds = manifest.get("configuration", {}).get("numeric_bounds", {})
    row: dict[str, Any] = {}
    for feature in MODEL_FEATURES:
        if feature in CATEGORY_DOMAINS:
            row[feature] = CATEGORY_DOMAINS[feature][0]
            continue
        feature_bounds = bounds.get(feature)
        if not isinstance(feature_bounds, dict):
            raise ValueError(f"manifest has no numeric bounds for {feature!r}")
        minimum = float(feature_bounds["minimum"])
        maximum = float(feature_bounds["maximum"])
        row[feature] = (minimum + maximum) / 2.0
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def _load_manifest(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model manifest must be a JSON object")
    return payload
