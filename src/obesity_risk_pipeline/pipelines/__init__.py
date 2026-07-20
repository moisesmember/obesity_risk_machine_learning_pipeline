"""Training and inference orchestration."""

from obesity_risk_pipeline.pipelines.training import (
    TrainingResult,
    run_experiments,
    train_baselines,
)
from obesity_risk_pipeline.pipelines.automation import (
    AutomatedTrainingResult,
    run_training_workflow,
)

__all__ = [
    "AutomatedTrainingResult",
    "TrainingResult",
    "run_experiments",
    "run_training_workflow",
    "train_baselines",
]
