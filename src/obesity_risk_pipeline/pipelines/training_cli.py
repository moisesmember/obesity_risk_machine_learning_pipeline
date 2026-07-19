"""CLI for the governed baseline-training slice."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from obesity_risk_pipeline.config.modeling import load_modeling_settings
from obesity_risk_pipeline.data.modeling import ModelingDataError
from obesity_risk_pipeline.pipelines.training import train_baselines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train governed obesity baselines and evaluate the winner on test."
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--validation-size", type=float)
    parser.add_argument("--random-state", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline training and return a process-compatible status code."""

    args = _parser().parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        settings = load_modeling_settings()
        overrides = {
            name: value
            for name, value in (
                ("dataset_path", args.dataset_path),
                ("output_root", args.output_root),
                ("test_size", args.test_size),
                ("validation_size", args.validation_size),
                ("random_state", args.random_state),
            )
            if value is not None
        }
        if overrides:
            settings = replace(settings, **overrides)
        result = train_baselines(settings)
    except (ModelingDataError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("Baseline training failed: %s", exc)
        return 1

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "selected_candidate": result.selected_candidate,
                "validation_macro_f1": result.validation_reports[
                    result.selected_candidate
                ].macro_f1,
                "test_macro_f1": result.test_report.macro_f1,
                "run_directory": str(result.run_directory),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
