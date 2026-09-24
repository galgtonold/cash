"""Upstream change-detection subsystem.

Public surface (production):
    - :class:`UpstreamChecker` — orchestrator. Detects stale upstream
      statements and re-executes them.
    - :class:`UpstreamResult` — value returned by ``check_and_reexecute``.

Public surface (tests):
    - :class:`NotebookSimulator` — pure-AST + cache-probing replay of upstream
      cells; constructible with just a ``TrackingState`` + ``SimpleNamespace``.

Everything else is internal to this package: the simulation's phases
(:class:`VirtualLineage`, :class:`MismatchClassifier`,
:class:`ReexecutionPlanner`), the parts they and the checker delegate to
(``cache_restore``, ``loop_rules``, ``unsaved_edits``, ``stale_values``,
``read_scope``, ``file_writers``, ``notebook_vetting``, ``rng_rewind``,
``replay``), and the value types in ``_types``.
"""

from __future__ import annotations

from .checker import UpstreamChecker, UpstreamResult
from .simulator import NotebookSimulator

__all__ = [
    "NotebookSimulator",
    "UpstreamChecker",
    "UpstreamResult",
]
