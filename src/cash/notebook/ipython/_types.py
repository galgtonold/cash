"""Value types for the IPython adapter.

Pure data classes used to ferry per-cell / per-statement metrics across
the adapter (`CashMagics`, `CellExecutor`, `%cash_status`). Extracted from
``magics.py`` so the orchestrator file is just the orchestrator.

Mirrors the ``_types.py`` pattern used by :mod:`cash.notebook.upstream`
(simulator-internal IR) — leading underscore marks the module as internal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ..statement import ProcessResult


class TimingBreakdown(TypedDict, total=False):
    """Phase-level timing accumulated during ``_execute_cell``."""

    badge_init: float
    total_restore_time: float
    total_execution_time: float
    upstream_check: float
    upstream_check_raw: float
    badge_progress: float


class StatementSummary(TypedDict):
    """Per-statement summary stored in ``CellMetrics.statements``."""

    code: str
    status: str | None
    execution_time: float
    saved_time: float
    outputs: list[str]
    is_upstream: bool


class CellMetrics(TypedDict):
    """Structure of ``_last_cell_metrics`` exposed by ``%cash_status``."""

    statements: list[StatementSummary]
    total_time: float
    total_restored_time: float
    total_computed_time: float
    upstream_metrics: list[ProcessResult]
    status: str | None
