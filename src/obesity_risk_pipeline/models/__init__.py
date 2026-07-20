"""Model catalog, evaluation, CV, optimization and governance helpers."""

from obesity_risk_pipeline.models.baselines import build_baseline_candidates
from obesity_risk_pipeline.models.catalog import Candidate, build_candidate_catalog
from obesity_risk_pipeline.models.evaluation import (
    EvaluationReport,
    evaluate_classifier,
    evaluate_predictions,
)
from obesity_risk_pipeline.models.governance import (
    GateResult,
    HumanApproval,
    PromotionEvaluation,
    evaluate_run_for_promotion,
)
from obesity_risk_pipeline.models.validation import CrossValidationReport

__all__ = [
    "Candidate",
    "CrossValidationReport",
    "EvaluationReport",
    "GateResult",
    "HumanApproval",
    "PromotionEvaluation",
    "build_baseline_candidates",
    "build_candidate_catalog",
    "evaluate_classifier",
    "evaluate_predictions",
    "evaluate_run_for_promotion",
]
