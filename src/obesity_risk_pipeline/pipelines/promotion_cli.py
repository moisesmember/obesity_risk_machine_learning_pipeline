"""CLI for fail-closed evaluation and explicit MLflow model promotion."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from obesity_risk_pipeline.config.promotion import load_promotion_policy
from obesity_risk_pipeline.models.governance import HumanApproval
from obesity_risk_pipeline.pipelines.promotion import promote_run


LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a run and register it only when policy and approval both pass."""

    parser = argparse.ArgumentParser(
        description="Evaluate deterministic model promotion gates."
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/promotions")
    )
    parser.add_argument("--approved-by")
    parser.add_argument("--approval-ticket")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--alias", default="candidate")
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if bool(args.approved_by) != bool(args.approval_ticket):
            raise ValueError(
                "--approved-by and --approval-ticket must be provided together"
            )
        approval = (
            HumanApproval(args.approved_by, args.approval_ticket)
            if args.approved_by
            else None
        )
        policy = load_promotion_policy(args.policy)
        result = promote_run(
            run_directory=args.run_directory,
            policy=policy,
            output_root=args.output_root,
            approval=approval,
            register=args.register,
            alias=args.alias,
            tracking_uri=args.tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Promotion failed: %s", exc)
        return 1

    for gate in result.evaluation.gates:
        log = LOGGER.info if gate.passed else LOGGER.warning
        log(
            "gate=%s passed=%s actual=%s operator=%s required=%s",
            gate.name,
            gate.passed,
            gate.actual,
            gate.operator,
            gate.required,
        )
    LOGGER.info(
        "Promotion decision=%s report=%s",
        result.evaluation.decision,
        result.report_path.resolve(),
    )
    if result.registry:
        LOGGER.info(
            "MLflow model registered name=%s version=%s alias=%s",
            result.registry.model_name,
            result.registry.version,
            result.registry.alias,
        )
    if result.evaluation.decision == "approved":
        return 0
    return 2 if result.evaluation.decision == "pending_approval" else 3


if __name__ == "__main__":
    raise SystemExit(main())
