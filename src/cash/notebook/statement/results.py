"""The result a processed statement reports, as the badge and the cell read it."""

from __future__ import annotations

from typing import Any, TypedDict

from cash.notebook.cache_status import CacheStatus

__all__ = ["COST_MODEL_KEYS", "DecoratorCallMetric", "ProcessResult", "ProcessResultRequired"]

#: The cost-model prediction fields a stored entry carries, copied into the
#: statement's metrics whether it was computed or served from the cache.
COST_MODEL_KEYS = (
    "cost_model_size_bytes",
    "cost_model_restore_seconds",
    "cost_model_type_name",
    "cost_model_family",
)


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
    """One call event in a statement's ``decorator_calls``.

    Written by ``@cash.cache`` (``Cash.drain_decorator_calls``) and, in the
    same shape, by an intercepted call (``CallUnit._record``). The loop
    handler stamps the ``loop_header``/``body_index`` keys onto an event whose
    statement ran inside a loop.
    """

    func_name: str
    cache_hit: bool
    execution_time: float
    time_saved: float
    args_hash: str
    cache_key: str | None
    timestamp: float
    # A decorator miss carries a ``MissReason``; an intercepted one a string.
    miss_reason: Any
    # --- @cash.cache only ---
    body_seconds: float | None
    cash_seconds: float
    not_persisted: Any
    not_stored: Any
    sampled_files: tuple[str, ...]
    # --- intercepted calls only ---
    call_source: str
    occurrence_index: int
    intercepted: bool
    ran_plain: bool
    stored: bool
    # --- stamped inside a loop ---
    loop_header: str
    loop_header_chain: list[str]
    body_index_chain: list[int]


class ProcessResult(ProcessResultRequired, total=False):
    """Typed dictionary for the return value of ``StatementProcessor.process_statement()``.

    Required keys (inherited from ``ProcessResultRequired``) are always
    present.  Optional keys are added during execution depending on cache
    status.
    """

    # --- Set when available ---
    # The statement as the user laid it out, for the badge; never hashed.
    display_code: str | None
    cache_key: str | None
    # What moved since the statement last ran, when it missed.
    miss_reason: str
    # Why the miss guard stopped storing it (see ``MissGuard.cause``).
    guard_cause: str
    # What the statement cost to run, and what cash spent around it.
    compute_cost: float
    cash_tax: float
    # Rich display outputs captured on a miss or replayed on a hit.
    rich_outputs: list[Any]
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
    # Stamped by the loop handler on a statement that ran inside a loop.
    loop_header: str
    loop_header_chain: list[str]
    body_index: int
    body_index_chain: list[int]
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
