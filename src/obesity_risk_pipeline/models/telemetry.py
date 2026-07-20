"""Structured, privacy-safe telemetry for governed training runs."""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class TrainingEventRecorder:
    """Emit readable logs and retain equivalent JSONL audit events."""

    def __init__(self, run_id: str, logger: logging.Logger) -> None:
        self.run_id = run_id
        self._logger = logger
        self._events: list[dict[str, Any]] = []
        self._stage_durations: dict[str, float] = {}
        self._run_started = time.perf_counter()

    @property
    def stage_durations(self) -> dict[str, float]:
        """Return completed top-level stage durations in seconds."""

        return dict(self._stage_durations)

    @property
    def elapsed_seconds(self) -> float:
        """Return monotonic elapsed time since recorder creation."""

        return time.perf_counter() - self._run_started

    def record(self, stage: str, status: str, **details: Any) -> None:
        """Record a safe event and mirror it to the configured Python logger."""

        event = {
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "run_id": self.run_id,
            "stage": stage,
            "status": status,
            **{key: _json_safe(value) for key, value in details.items()},
        }
        self._events.append(event)
        self._logger.info(
            "training_event=%s",
            json.dumps(event, ensure_ascii=False, sort_keys=True),
        )

    @contextmanager
    def stage(
        self, stage: str, *, track_duration: bool = True, **details: Any
    ) -> Iterator[dict[str, Any]]:
        """Record the start, completion or failure of a measured stage."""

        started = time.perf_counter()
        outcome: dict[str, Any] = {}
        self.record(stage, "started", **details)
        try:
            yield outcome
        except Exception as exc:
            duration = time.perf_counter() - started
            self.record(
                stage,
                "failed",
                duration_seconds=duration,
                error_type=type(exc).__name__,
                **details,
            )
            raise
        else:
            duration = time.perf_counter() - started
            if track_duration:
                self._stage_durations[stage] = duration
            self.record(
                stage,
                "completed",
                duration_seconds=duration,
                **details,
                **outcome,
            )

    def write_jsonl(self, path: Path) -> None:
        """Persist all events accumulated so far as UTF-8 JSON Lines."""

        content = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in self._events
        )
        path.write_text(content, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)
