"""Single command for dataset initialization followed by governed training."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from obesity_risk_pipeline.config.experiments import load_experiment_plan
from obesity_risk_pipeline.config.modeling import load_modeling_settings
from obesity_risk_pipeline.config.settings import load_ingestion_settings
from obesity_risk_pipeline.ingestion.service import IngestionError
from obesity_risk_pipeline.pipelines.automation import run_training_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize the dataset and execute governed model training."
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.json"),
        help="Experiment plan used in full mode",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--optuna-trials", type=int)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run initialization and training with one process-compatible command."""

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        from dotenv import load_dotenv

        load_dotenv()
        ingestion_settings = load_ingestion_settings()
        modeling_settings = load_modeling_settings()
        overrides = {
            name: value
            for name, value in (
                ("output_root", args.output_root),
                ("cv_folds", args.cv_folds),
                ("random_state", args.random_state),
                ("optuna_trials", args.optuna_trials),
            )
            if value is not None
        }
        if args.enable_mlflow:
            overrides["mlflow_enabled"] = True
        if overrides:
            modeling_settings = replace(modeling_settings, **overrides)
        plan = load_experiment_plan(args.config) if args.mode == "full" else None
        result = run_training_workflow(
            mode=args.mode,
            ingestion_settings=ingestion_settings,
            modeling_settings=modeling_settings,
            experiment_plan=plan,
        )
    except (IngestionError, OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error("Automated training failed: %s", exc)
        return 1

    logging.getLogger(__name__).info(
        "Automated training completed: dataset_reused=%s run_id=%s selected=%s "
        "run_directory=%s",
        result.ingestion.reused_existing_snapshot,
        result.training.run_id,
        result.training.selected_candidate,
        result.training.run_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
