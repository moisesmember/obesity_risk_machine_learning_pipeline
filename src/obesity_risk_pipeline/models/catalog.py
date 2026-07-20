"""Progressive model catalog with controlled optional dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from obesity_risk_pipeline.data.modeling import CATEGORY_DOMAINS, TARGET_CLASSES
from obesity_risk_pipeline.features.engineering import FeatureEngineer
from obesity_risk_pipeline.features.experiments import ExperimentSpec
from obesity_risk_pipeline.features.preprocessing import build_preprocessor

DEFAULT_MODEL_NAMES = (
    "dummy",
    "logistic_regression",
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "catboost",
    "lightgbm",
    "xgboost",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A fully specified model/experiment pair."""

    name: str
    model_name: str
    experiment: ExperimentSpec
    estimator: BaseEstimator


@dataclass(frozen=True, slots=True)
class CandidateCatalog:
    """Available candidates and actionable optional dependency failures."""

    available: Mapping[str, Candidate]
    unavailable: Mapping[str, str]


class OrderedTargetClassifier(ClassifierMixin, BaseEstimator):
    """Adapt estimators that require integer targets to the canonical labels."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, features: Any, target: Any) -> OrderedTargetClassifier:
        mapping = {label: index for index, label in enumerate(TARGET_CLASSES)}
        labels = np.asarray([mapping[str(value)] for value in target], dtype=int)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(features, labels)
        self.classes_ = np.asarray(TARGET_CLASSES, dtype=object)
        return self

    def predict(self, features: Any) -> np.ndarray:
        indices = np.asarray(self.estimator_.predict(features), dtype=int)
        return np.asarray(TARGET_CLASSES, dtype=object)[indices]

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.asarray(self.estimator_.predict_proba(features), dtype=float)


def build_candidate_catalog(
    experiments: Iterable[ExperimentSpec],
    random_state: int,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
) -> CandidateCatalog:
    """Build every requested model/ablation pair without importing unused backends."""

    available: dict[str, Candidate] = {}
    unavailable: dict[str, str] = {}
    for model_name in model_names:
        for experiment in experiments:
            candidate_name = f"{experiment.name}__{model_name}"
            try:
                estimator = _build_estimator(model_name, experiment, random_state)
            except ImportError as exc:
                unavailable[candidate_name] = str(exc)
                continue
            available[candidate_name] = Candidate(
                candidate_name, model_name, experiment, estimator
            )
    return CandidateCatalog(available, unavailable)


def _feature_engineer(experiment: ExperimentSpec) -> FeatureEngineer:
    return FeatureEngineer(
        experiment.source_features,
        include_bmi=experiment.include_bmi,
        age_mode=experiment.age_mode,
    )


def _sklearn_pipeline(
    experiment: ExperimentSpec,
    classifier: BaseEstimator,
    *,
    scale_numeric: bool,
) -> Pipeline:
    return Pipeline(
        (
            ("features", _feature_engineer(experiment)),
            (
                "preprocessor",
                build_preprocessor(
                    experiment.output_features,
                    scale_numeric=scale_numeric,
                    categorical_encoding=experiment.categorical_encoding,
                ),
            ),
            ("classifier", classifier),
        )
    )


def _build_estimator(
    model_name: str,
    experiment: ExperimentSpec,
    random_state: int,
) -> BaseEstimator:
    if model_name == "dummy":
        return _sklearn_pipeline(
            experiment,
            DummyClassifier(strategy="stratified", random_state=random_state),
            scale_numeric=False,
        )
    if model_name == "logistic_regression":
        return _sklearn_pipeline(
            experiment,
            LogisticRegression(max_iter=1_500, random_state=random_state),
            scale_numeric=True,
        )
    if model_name == "extra_trees":
        return _sklearn_pipeline(
            experiment,
            ExtraTreesClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            scale_numeric=False,
        )
    if model_name == "random_forest":
        return _sklearn_pipeline(
            experiment,
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            scale_numeric=False,
        )
    if model_name == "hist_gradient_boosting":
        return _sklearn_pipeline(
            experiment,
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=random_state,
            ),
            scale_numeric=False,
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "catboost is not installed; install requirements-modeling.txt"
            ) from exc
        categorical = [
            name for name in experiment.output_features if name in CATEGORY_DOMAINS
        ]
        classifier = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=250,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            cat_features=categorical,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        return Pipeline(
            (("features", _feature_engineer(experiment)), ("classifier", classifier))
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed; install requirements-modeling.txt"
            ) from exc
        classifier = OrderedTargetClassifier(
            LGBMClassifier(
                objective="multiclass",
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
        )
        return _sklearn_pipeline(
            experiment, classifier, scale_numeric=False
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed; install requirements-modeling.txt"
            ) from exc
        classifier = OrderedTargetClassifier(
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(TARGET_CLASSES),
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
            )
        )
        return _sklearn_pipeline(
            experiment, classifier, scale_numeric=False
        )
    raise ValueError(f"unknown model name: {model_name!r}")
