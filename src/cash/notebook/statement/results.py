"""The result a processed statement reports, as the badge and the cell read it."""

from __future__ import annotations

from typing import Any, TypedDict

from cash.notebook.cache_status import CacheStatus

__all__ = ["DecoratorCallMetric", "ProcessResult", "ProcessResultRequired"]


class ProcessResultRequired(TypedDict):
    """Keys that are always present in a :class:`ProcessResult`."""

    status: CacheStatus
    execution_time: float
    total_time: float
    saved_time: float
    error: Exception | None
    restored_vars: list[str]
    code: str
    uncacheable_reasons: list[str]


class DecoratorCallMetric(TypedDict, total=False):
    """Metrics for a single ``@cash.cache`` decorated function call."""

    func_name: str
    cache_hit: bool
    execution_time: float


class ProcessResult(ProcessResultRequired, total=False):
    """Typed dictionary for the return value of ``StatementProcessor.process_statement()``.

    Required keys (inherited from ``ProcessResultRequired``) are always
    present.  Optional keys are added during execution depending on cache
    status.
    """

    # --- Set when available ---
    outputs: list[str]
    storage: list[str]
    _output_flushed: bool
    control_type: str
    body_statements: list[str]
    # --- Added by process_statement() at runtime ---
    source: str
    stdout: str
    stderr: str
    decorator_calls: list[DecoratorCallMetric]
    evaluated_vars: list[str]
    skipped_reason: str
    is_upstream: bool
    inputs: list[str]
    output_vars: list[str]
    loop_vars: dict[str, Any]
    control_context: str
    branch_label: str
    changed_functions: list[str]
    changed_modules: dict[str, str]
    # A statement's RNG role, surfaced on the badge: 'seed' sets a global seed,
    # 'draw' consumes randomness. ``random_unseeded`` marks a draw/fit with no
    # frozen seed, whose cached value is a frozen replay (advisory, still cached).
    random_effect: str
    random_unseeded: bool
    cost_model_size_bytes: int
    cost_model_restore_seconds: float
    cost_model_type_name: str
    cost_model_family: str
