from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from obesity_risk_pipeline.config.promotion import (
    PromotionPolicy,
    PromotionThresholds,
)
from obesity_risk_pipeline.data.modeling import (
    CATEGORY_DOMAINS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from obesity_risk_pipeline.models.governance import (
    HumanApproval,
    PromotionEvaluation,
)
from obesity_risk_pipeline.models.registry import (
    SKOPS_TRUSTED_TYPES,
    register_approved_model,
)


def _policy() -> PromotionPolicy:
    return PromotionPolicy(
        policy_version="test-v1",
        model_name="obesity-risk-test",
        thresholds=PromotionThresholds(0.8, 0.8, 0.7, 0.3, 0.05, 0.1),
    )


def _evaluation(*, decision: str = "approved") -> PromotionEvaluation:
    return PromotionEvaluation(
        decision=decision,
        source_run_id="run-1",
        selected_candidate="dummy",
        policy_version="test-v1",
        policy_sha256="policy-sha",
        model_name="obesity-risk-test",
        gates=(),
        approval=HumanApproval("owner@example.com", "CHANGE-123"),
        mlflow_run_id="training-run-1",
    )


def _run_directory(tmp_path: Path) -> Path:
    root = tmp_path / "run-1"
    root.mkdir()
    row = {
        feature: (
            CATEGORY_DOMAINS[feature][0]
            if feature in CATEGORY_DOMAINS
            else 1.0
        )
        for feature in MODEL_FEATURES
    }
    features = pd.DataFrame([row, row], columns=MODEL_FEATURES)
    estimator = DummyClassifier(strategy="most_frequent").fit(
        features, ["Normal_Weight", "Normal_Weight"]
    )
    joblib.dump(estimator, root / "model.joblib")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "numeric_bounds": {
                        name: {"minimum": 0.0, "maximum": 2.0}
                        for name in NUMERIC_FEATURES
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_registry_refuses_non_approved_evaluation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only an approved"):
        register_approved_model(
            run_directory=tmp_path,
            policy=_policy(),
            evaluation=_evaluation(decision="rejected"),
            alias="candidate",
        )


def test_registry_logs_reloads_registers_and_aliases_approved_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    evaluation = _evaluation()
    monkeypatch.setattr(
        "obesity_risk_pipeline.models.registry.evaluate_run_for_promotion",
        lambda *args, **kwargs: evaluation,
    )
    mlflow = ModuleType("mlflow")
    mlflow.__path__ = []  # type: ignore[attr-defined]
    sklearn_module = ModuleType("mlflow.sklearn")
    models_module = ModuleType("mlflow.models")
    tracking_module = ModuleType("mlflow.tracking")
    exceptions_module = ModuleType("mlflow.exceptions")
    loaded_estimator: object | None = None

    def set_tracking_uri(uri: str) -> None:
        calls["tracking_uri"] = uri

    def set_experiment(name: str) -> None:
        calls["experiment"] = name

    @contextmanager
    def start_run(**kwargs: object):
        calls["start_run"] = kwargs
        yield SimpleNamespace(info=SimpleNamespace(run_id="registry-run-1"))

    def log_model(**kwargs: object) -> SimpleNamespace:
        nonlocal loaded_estimator
        loaded_estimator = kwargs["sk_model"]
        calls["log_model"] = kwargs
        return SimpleNamespace(model_uri="runs:/registry-run-1/model")

    def load_model(uri: str) -> object:
        calls["load_model"] = uri
        return loaded_estimator

    def register_model(uri: str, name: str) -> SimpleNamespace:
        calls["register_model"] = (uri, name)
        return SimpleNamespace(version="7")

    class FakeClient:
        def __init__(self, *, tracking_uri: str) -> None:
            calls["client_tracking_uri"] = tracking_uri

        def set_model_version_tag(
            self, name: str, version: str, key: str, value: str
        ) -> None:
            calls.setdefault("tags", []).append((name, version, key, value))

        def set_registered_model_alias(
            self, name: str, alias: str, version: str
        ) -> None:
            calls["alias"] = (name, alias, version)

    class FakeMlflowException(Exception):
        pass

    mlflow.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    mlflow.set_experiment = set_experiment  # type: ignore[attr-defined]
    mlflow.start_run = start_run  # type: ignore[attr-defined]
    mlflow.register_model = register_model  # type: ignore[attr-defined]
    mlflow.sklearn = sklearn_module  # type: ignore[attr-defined]
    sklearn_module.log_model = log_model  # type: ignore[attr-defined]
    sklearn_module.load_model = load_model  # type: ignore[attr-defined]
    models_module.infer_signature = lambda inputs, outputs: (inputs, outputs)  # type: ignore[attr-defined]
    tracking_module.MlflowClient = FakeClient  # type: ignore[attr-defined]
    exceptions_module.MlflowException = FakeMlflowException  # type: ignore[attr-defined]
    for name, module in {
        "mlflow": mlflow,
        "mlflow.sklearn": sklearn_module,
        "mlflow.models": models_module,
        "mlflow.tracking": tracking_module,
        "mlflow.exceptions": exceptions_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = register_approved_model(
        run_directory=_run_directory(tmp_path),
        policy=_policy(),
        evaluation=evaluation,
        alias="candidate",
        tracking_uri="http://mlflow.test:5000",
    )

    assert result.status == "registered"
    assert result.version == "7"
    logged_model = calls["log_model"]
    assert isinstance(logged_model, dict)
    assert logged_model["name"] == "model"
    assert "artifact_path" not in logged_model
    assert logged_model["serialization_format"] == "skops"
    assert logged_model["skops_trusted_types"] == list(SKOPS_TRUSTED_TYPES)
    assert Path(logged_model["code_paths"][0]).name == "src"
    assert calls["register_model"] == (
        "runs:/registry-run-1/model",
        "obesity-risk-test",
    )
    assert calls["alias"] == ("obesity-risk-test", "candidate", "7")
    assert ("obesity-risk-test", "7", "approval_ticket", "CHANGE-123") in calls[
        "tags"
    ]
