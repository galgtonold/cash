"""Structured JSON logging for cache operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if present
        for key in ("event", "duration_ms", "cache_key", "status", "variable", "module", "cell_id", "backend"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry)


def setup_logging(level: int = logging.INFO, json_output: bool = False, log_file: str | None = None) -> None:
    """Configure the ``cash`` logger hierarchy.

    Args:
        level: Logging level (DEBUG, INFO, etc.)
        json_output: If True, use JSON formatter for console output
        log_file: Optional path for JSON file logging
    """
    cash_logger = logging.getLogger("cash")
    cash_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    cash_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    if json_output:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    cash_logger.addHandler(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonFormatter())
        cash_logger.addHandler(fh)
