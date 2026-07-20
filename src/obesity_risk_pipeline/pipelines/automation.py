"""End-to-end initialization and training workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace

from obesity_risk_pipeline.config.experiments import ExperimentPlan
from obesity_risk_pipeline.config.modeling import ModelingSettings
from obesity_risk_pipeline.config.settings import IngestionSettings
from obesity_risk_pipeline.ingestion.kaggle import (
    DatasetDownloader,
    KaggleApiDownloader,
)
from obesity_risk_pipeline.ingestion.service import (
    IngestionResult,
    KaggleIngestionService,
)
from obesity_risk_pipeline.pipelines.training import (
    TrainingResult,
    run_experiments,
    train_baselines,
)


@dataclass(frozen=True, slots=True)
class AutomatedTrainingResult:
    """Results from the idempotent ingestion and governed training stages."""

    ingestion: IngestionResult
    training: TrainingResult


def run_training_workflow(
    *,
    mode: str,
    ingestion_settings: IngestionSettings,
    modeling_settings: ModelingSettings,
    experiment_plan: ExperimentPlan | None = None,
    downloader: DatasetDownloader | None = None,
) -> AutomatedTrainingResult:
    """Initialize the immutable dataset and run quick or complete training."""

    if mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    if mode == "full" and experiment_plan is None:
        raise ValueError("full mode requires an experiment plan")

    ingestion = KaggleIngestionService(
        settings=ingestion_settings,
        downloader=downloader if downloader is not None else KaggleApiDownloader(),
    ).run()
    governed_modeling = replace(
        modeling_settings,
        dataset_path=ingestion.dataset_path,
        expected_sha256=ingestion.sha256,
    )
    if mode == "quick":
        training = train_baselines(governed_modeling)
    else:
        assert experiment_plan is not None
        training = run_experiments(
            governed_modeling,
            experiment_names=experiment_plan.experiments,
            model_names=experiment_plan.models,
        )
    return AutomatedTrainingResult(ingestion=ingestion, training=training)
