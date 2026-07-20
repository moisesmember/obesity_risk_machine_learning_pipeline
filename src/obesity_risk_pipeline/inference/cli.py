"""Batch prediction command for a governed run directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from obesity_risk_pipeline.inference.service import PredictionService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run governed batch inference.")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        service = PredictionService.load(args.run_directory)
        predictions = service.predict(pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Inference failed: %s", exc)
        return 1
    logging.info("Predictions written to %s", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
