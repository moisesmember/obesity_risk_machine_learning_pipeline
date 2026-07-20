"""Production-oriented Uvicorn launcher with environment-only configuration."""

from __future__ import annotations

import argparse
import os
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve an explicitly promoted obesity-risk model."
    )
    parser.add_argument(
        "--host", default=os.getenv("OBESITY_API_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OBESITY_API_PORT", "8000")),
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.getenv("OBESITY_API_LOG_LEVEL", "info").lower(),
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "API dependencies are unavailable; install requirements-serving.txt"
        ) from exc
    uvicorn.run(
        "obesity_risk_pipeline.api.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
