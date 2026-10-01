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
    from ..cache_status import CacheStatus
    from ..statement import ProcessResult


class TimingBreakdown(TypedDict, total=False):
    """Phase-level timing accumulated during ``_execute_cell``."""

    badge_init: float
    total_restore_time: float
    total_execution_time: float
    upstream_check: float
    upstream_check_raw: float
    badge_progress: float
    # Writing what the cell deferred to its end (``end_cell_persistence``).
    persist_final: float
    # Asking object storage whether tracked remote data changed, and how often
    # (``remote_source``'s validation sink).
    remote_validate: float
    remote_validate_count: int


class StatementSummary(TypedDict):
    """Per-statement summary stored in ``CellMetrics.statements``."""

    code: str
    status: CacheStatus | None
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


class RunInstead:
    """The executor stepped aside: the hook runs *source* through IPython's
    original ``run_cell`` (or ``run_cell_async``) and returns its result.

    *source* is the cell itself, run uncached, or a one-line ``raise`` that
    surfaces an error cash found in the notebook as the cell's own. *error*
    is what the upstream check raised.
    """

    __slots__ = ("source", "error")

    def __init__(self, source: str, error: Exception) -> None:
        self.source = source
        self.error = error


class PipelineSyntaxError:
    """Sentinel returned by :meth:`CellExecutor.execute_cell` when the cell's
    own AST fails to parse.  Caller decides how to react."""

    __slots__ = ()


class PipelineCompleted:
    """Successful pipeline run: carries everything the finaliser needs."""

    __slots__ = (
        "all_metrics",
        "buffered_outputs",
        "badge_display_id",
        "hook_start",
        "timing_breakdown",
        "badge_render_time",
    )

    def __init__(
        self,
        all_metrics: list,
        buffered_outputs: list,
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: "TimingBreakdown",
        badge_render_time: float,
    ) -> None:
        self.all_metrics = all_metrics
        self.buffered_outputs = buffered_outputs
        self.badge_display_id = badge_display_id
        self.hook_start = hook_start
        self.timing_breakdown = timing_breakdown
        self.badge_render_time = badge_render_time
