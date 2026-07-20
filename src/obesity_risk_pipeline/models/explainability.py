"""Model-agnostic explainability with optional SHAP enrichment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def build_explainability_report(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int,
    max_rows: int = 500,
) -> dict[str, Any]:
    """Calculate raw-feature permutation importance on a bounded final sample."""

    sample_size = min(max_rows, len(features))
    sample = features.sample(n=sample_size, random_state=random_state)
    sample_target = target.loc[sample.index]
    result = permutation_importance(
        estimator,
        sample,
        sample_target,
        scoring="f1_macro",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    ordering = np.argsort(result.importances_mean)[::-1]
    importance = [
        {
            "feature": str(sample.columns[index]),
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index in ordering
    ]
    shap_result = _try_shap(estimator, sample)
    return {
        "method": "permutation_importance",
        "sample_rows": sample_size,
        "scoring": "f1_macro",
        "importance": importance,
        "shap": shap_result,
    }


def write_explainability_plot(report: dict[str, Any], path: Path) -> None:
    """Persist a compact permutation-importance chart for tracking."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    top = list(reversed(report["importance"][:15]))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        [item["feature"] for item in top],
        [item["importance_mean"] for item in top],
        xerr=[item["importance_std"] for item in top],
    )
    axis.set_title("Permutation importance — Macro-F1")
    axis.set_xlabel("Redução média da métrica")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _try_shap(estimator: BaseEstimator, sample: pd.DataFrame) -> dict[str, Any]:
    try:
        import shap
    except ImportError:
        return {"status": "unavailable"}
    try:
        if not hasattr(estimator, "named_steps"):
            return {"status": "incompatible", "reason": "not_a_named_pipeline"}
        working: Any = sample.iloc[: min(200, len(sample))]
        steps = list(estimator.named_steps.items())
        for _, transformer in steps[:-1]:
            working = transformer.transform(working)
        classifier = steps[-1][1]
        if hasattr(classifier, "estimator_"):
            classifier = classifier.estimator_
        explainer = shap.TreeExplainer(classifier)
        raw_values = explainer.shap_values(working)
        values = np.asarray(raw_values)
        if isinstance(raw_values, list) and values.ndim == 3:
            importance = np.mean(np.abs(values), axis=(0, 1))
        elif values.ndim == 3:
            importance = np.mean(np.abs(values), axis=(0, 2))
        elif values.ndim == 2:
            importance = np.mean(np.abs(values), axis=0)
        else:
            return {"status": "incompatible", "reason": "unexpected_shap_shape"}
        return {
            "status": "completed",
            "mean_absolute_values": [float(value) for value in importance],
        }
    except Exception as exc:
        return {"status": "failed_optional", "reason": type(exc).__name__}
