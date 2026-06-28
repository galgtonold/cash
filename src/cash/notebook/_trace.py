"""Bulletproof, opt-in decision tracing for the upstream checker / simulator.

Why a file and not ``logging``: the integration tests run cash inside a Jupyter
kernel **subprocess**. Some module loggers' output is reliably surfaced through
the kernel's IOPub stream into ``nb_runner.get_raw_output`` and some is not
(e.g. ``cash.notebook.upstream.simulator`` lines never appear), which makes
log-string debugging of the simulation engine unreliable. A file append from
inside the kernel is immune to all stdout/IOPub/logging capture quirks: the
kernel inherits ``CASH_TRACE_FILE`` from the parent process's environment and
both sides read the same path.

Usage (from a test): set ``CASH_TRACE_FILE`` *before* ``start_kernel`` (a cold
kernel inherits it; the opt-in warm kernel, booted once per worker, will not —
so trace with the default cold-kernel path). Then read/parse the JSONL file.
The :func:`tests...conftest.capture_upstream_trace` helper wraps this.

Zero-cost when off: a single ``os.environ.get`` per call, and the decision
points instrumented are not hot loops. Never raises — tracing must never change
program behaviour.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

__all__ = ["trace_event", "is_tracing"]

_LOCK = threading.Lock()
_ENV = "CASH_TRACE_FILE"


def is_tracing() -> bool:
    return bool(os.environ.get(_ENV))


def trace_event(event: str, **fields: Any) -> None:
    """Append one JSON record ``{"event": event, **fields}`` to the trace file.

    No-op unless ``CASH_TRACE_FILE`` is set. Sets are sorted to lists for stable,
    diffable output. Never raises.
    """
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        norm = {k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                for k, v in fields.items()}
        line = json.dumps({"event": event, **norm}, default=str)
    except Exception:  # noqa: BLE001 — tracing must never break the run
        try:
            line = json.dumps({"event": event, "_trace_error": True})
        except Exception:  # noqa: BLE001
            return
    try:
        with _LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
