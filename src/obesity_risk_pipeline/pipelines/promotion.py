"""Promotion orchestration separated from model training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from obesity_risk_pipeline.config.promotion import PromotionPolicy
from obesity_risk_pipeline.models.governance import (
    HumanApproval,
    PromotionEvaluation,
    evaluate_run_for_promotion,
    utc_now,
)
from obesity_risk_pipeline.models.registry import (
    RegistryResult,
    register_approved_model,
)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Evaluation, optional registry mutation and immutable local report."""

    evaluation: PromotionEvaluation
    registry: RegistryResult | None
    report_directory: Path
    report_path: Path


def promote_run(
    *,
    run_directory: Path,
    policy: PromotionPolicy,
    output_root: Path,
    approval: HumanApproval | None = None,
    register: bool = False,
    alias: str = "candidate",
    tracking_uri: str | None = None,
) -> PromotionResult:
    """Evaluate gates, optionally mutate MLflow, and publish an immutable report."""

    evaluation = evaluate_run_for_promotion(run_directory, policy, approval)
    decision_id = _decision_id(evaluation, register, alias)
    report_directory = output_root / evaluation.source_run_id / decision_id
    report_path = report_directory / "promotion_report.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        registry_payload = existing.get("registry", {})
        existing_registry = (
            RegistryResult(**registry_payload)
            if registry_payload.get("status") == "registered"
            else None
        )
        return PromotionResult(
            evaluation, existing_registry, report_directory, report_path
        )
    registry: RegistryResult | None = None
    if register and evaluation.decision == "approved":
        registry = register_approved_model(
            run_directory=run_directory,
            policy=policy,
            evaluation=evaluation,
            alias=alias,
            tracking_uri=tracking_uri,
        )
    registry_payload: dict[str, Any]
    if registry:
        registry_payload = registry.to_dict()
    elif register:
        registry_payload = {
            "status": "blocked_by_gates",
            "decision": evaluation.decision,
        }
    else:
        registry_payload = {"status": "not_requested"}
    report_payload = {
        "report_schema_version": 1,
        "created_at_utc": utc_now(),
        "evaluation": evaluation.to_dict(),
        "registry": registry_payload,
    }
    publishing = output_root / evaluation.source_run_id / f".{decision_id}.publishing"
    publishing.parent.mkdir(parents=True, exist_ok=True)
    publishing.mkdir(exist_ok=False)
    try:
        publication_report = publishing / "promotion_report.json"
        _write_json(publication_report, report_payload)
        _write_json(
            publishing / "manifest.json",
            {
                "decision_id": decision_id,
                "source_run_id": evaluation.source_run_id,
                "promotion_report_sha256": hashlib.sha256(
                    publication_report.read_bytes()
                ).hexdigest(),
                "promotion_report_byte_size": publication_report.stat().st_size,
            },
        )
        os.replace(publishing, report_directory)
    except Exception:
        shutil.rmtree(publishing, ignore_errors=True)
        raise
    return PromotionResult(evaluation, registry, report_directory, report_path)


def _decision_id(
    evaluation: PromotionEvaluation,
    register: bool,
    alias: str,
) -> str:
    payload = {
        "source_run_id": evaluation.source_run_id,
        "policy_sha256": evaluation.policy_sha256,
        "approval": asdict(evaluation.approval) if evaluation.approval else None,
        "register": register,
        "alias": alias,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
