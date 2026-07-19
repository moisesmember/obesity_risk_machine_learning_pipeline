"""Command-line entry point for the governed Kaggle ingestion."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from obesity_risk_pipeline.config.settings import load_ingestion_settings
from obesity_risk_pipeline.ingestion.kaggle import KaggleApiDownloader
from obesity_risk_pipeline.ingestion.service import IngestionError, KaggleIngestionService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and publish the governed obesity dataset from Kaggle."
    )
    parser.add_argument("--dataset-slug", help="Kaggle slug in owner/dataset format")
    parser.add_argument("--expected-sha256", help="Governed SHA-256 of the CSV")
    parser.add_argument("--raw-root", type=Path, help="Immutable publication root")
    parser.add_argument("--staging-root", type=Path, help="Temporary download root")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion CLI and return a process-compatible status code."""

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
        settings = load_ingestion_settings()
        overrides: dict[str, object] = {}
        if args.dataset_slug:
            overrides["dataset_slug"] = args.dataset_slug
        if args.expected_sha256:
            overrides["expected_sha256"] = args.expected_sha256
        if args.raw_root:
            overrides["raw_root"] = args.raw_root
        if args.staging_root:
            overrides["staging_root"] = args.staging_root
        if overrides:
            settings = replace(settings, **overrides)

        result = KaggleIngestionService(
            settings=settings,
            downloader=KaggleApiDownloader(),
        ).run()
    except (IngestionError, ValueError) as exc:
        logging.getLogger(__name__).error("Ingestion failed: %s", exc)
        return 1

    logging.getLogger(__name__).info(
        "Ingestion completed: path=%s rows=%d sha256=%s reused=%s",
        result.dataset_path,
        result.row_count,
        result.sha256,
        result.reused_existing_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
