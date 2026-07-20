"""Commands for complete experiments and the fast compatibility baseline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from obesity_risk_pipeline.config.modeling import load_modeling_settings
from obesity_risk_pipeline.config.experiments import load_experiment_plan
from obesity_risk_pipeline.data.modeling import ModelingDataError
from obesity_risk_pipeline.pipelines.training import run_experiments, train_baselines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run governed obesity experiments.")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--holdout-size", type=float)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--optuna-trials", type=int)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--experiments", help="Comma-separated experiment names")
    parser.add_argument("--models", help="Comma-separated model names")
    parser.add_argument(
        "--config",
        type=Path,
        help="External experiment/model plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete requested catalog unless explicitly filtered."""

    return _execute(argv, fast_baselines=False)


def baseline_main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded baseline slice for fast local verification."""

    return _execute(argv, fast_baselines=True)


def _execute(argv: Sequence[str] | None, *, fast_baselines: bool) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()
        settings = load_modeling_settings()
        overrides = {
            name: value
            for name, value in (
                ("dataset_path", args.dataset_path),
                ("output_root", args.output_root),
                ("holdout_size", args.holdout_size),
                ("cv_folds", args.cv_folds),
                ("random_state", args.random_state),
                ("optuna_trials", args.optuna_trials),
            )
            if value is not None
        }
        if args.enable_mlflow:
            overrides["mlflow_enabled"] = True
        settings = replace(settings, **overrides) if overrides else settings
        if fast_baselines:
            result = train_baselines(settings)
        else:
            default_config = Path("configs/experiments.json")
            if args.config is not None:
                plan = load_experiment_plan(args.config)
            elif default_config.is_file():
                plan = load_experiment_plan(default_config)
            else:
                plan = None
            experiments = (
                tuple(args.experiments.split(","))
                if args.experiments
                else plan.experiments if plan else None
            )
            models = (
                tuple(args.models.split(","))
                if args.models
                else plan.models if plan else None
            )
            result = run_experiments(
                settings, experiment_names=experiments, model_names=models
            )
    except (ModelingDataError, OSError, ValueError, RuntimeError) as exc:
        logging.error("Training failed: %s", exc)
        return 1
    selected_cv = result.cv_reports[result.selected_candidate]
    logging.info(
        "run_id=%s selected=%s cv_macro_f1=%.6f holdout_macro_f1=%.6f directory=%s",
        result.run_id,
        result.selected_candidate,
        selected_cv.metric_mean["macro_f1"],
        result.holdout_report.macro_f1,
        result.run_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
