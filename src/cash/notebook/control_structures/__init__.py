from __future__ import annotations

"""Control-structure caching subsystem.

A strategy pattern: :class:`ControlStructureProcessor` (the orchestrator)
dispatches per-iteration / per-statement work to the three handler classes
(`ForLoopHandler`, `IfHandler`, `TryHandler`) and falls back to single-unit
processing for `while` / `with` / break-continue loops. Shared lineage,
mutation, badge, and error helpers live in :mod:`.helpers` and are imported
by all handlers — handlers never call back into each other.

Public surface:
    - :class:`ControlStructureProcessor` — the orchestrator.
    - :class:`ControlStructureResult` — the return type.
    - Pure AST helpers used by callers across the notebook subsystem:
      :func:`is_control_structure`, :func:`get_control_structure_type`,
      :func:`extract_target_names`, :func:`contains_break_or_continue`,
      :func:`bind_target_values`, :func:`build_iteration_context`,
      :func:`compute_context_hash`.

Handler classes (`ForLoopHandler`, `IfHandler`, `TryHandler`) and the
contents of :mod:`.helpers` are internal to this package. See ADR-012
for the package-extraction rationale.
"""

from .processor import (
    ControlStructureProcessor,
    ControlStructureResult,
    bind_target_values,
    build_iteration_context,
    compute_context_hash,
    contains_break_or_continue,
    contains_top_level_await,
    extract_target_names,
    get_control_structure_type,
    is_control_structure,
)

__all__ = [
    "ControlStructureProcessor",
    "ControlStructureResult",
    "bind_target_values",
    "build_iteration_context",
    "compute_context_hash",
    "contains_break_or_continue",
    "contains_top_level_await",
    "extract_target_names",
    "get_control_structure_type",
    "is_control_structure",
]
