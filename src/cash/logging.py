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
        for key in ("event", "duration_ms", "cache_key", "status",
                     "variable", "module", "cell_id", "backend"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry)


class CashLogHandler(logging.Handler):
    """Handler that stores structured log events for inspection.

    Accumulates log records so they can be retrieved via
    ``%cash_stats log`` or the CLI.
    """

    MAX_ENTRIES = 1000

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "time": record.created,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for key in ("event", "duration_ms", "cache_key", "status",
                     "variable", "module"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        self.records.append(entry)
        if len(self.records) > self.MAX_ENTRIES:
            self.records = self.records[-self.MAX_ENTRIES:]

    def get_events(self, event_type: str | None = None, limit: int = 50) -> list[dict]:
        """Retrieve recent log events, optionally filtered."""
        filtered = [r for r in self.records if r.get("event") == event_type] if event_type else self.records
        return filtered[-limit:]

    def clear(self) -> None:
        self.records.clear()


def setup_logging(level: int = logging.INFO,
                  json_output: bool = False,
                  log_file: str | None = None) -> CashLogHandler:
    """Configure the ``cash`` logger hierarchy.

    Args:
        level: Logging level (DEBUG, INFO, etc.)
        json_output: If True, use JSON formatter for console output
        log_file: Optional path for JSON file logging

    Returns:
        The CashLogHandler for programmatic access to events.
    """
    cash_logger = logging.getLogger("cash")
    cash_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    cash_logger.handlers.clear()

    # In-memory structured handler (always active)
    mem_handler = CashLogHandler()
    mem_handler.setLevel(level)
    cash_logger.addHandler(mem_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    if json_output:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(logging.Formatter(
            "[%(name)s] %(message)s"
        ))
    cash_logger.addHandler(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonFormatter())
        cash_logger.addHandler(fh)

    return mem_handler
