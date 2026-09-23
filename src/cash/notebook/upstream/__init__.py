"""Upstream change-detection subsystem.

Public surface (production):
    - :class:`UpstreamChecker` — orchestrator. Detects stale upstream
      statements and re-executes them.
    - :class:`UpstreamResult` — value returned by ``check_and_reexecute``.

Public surface (tests):
    - :class:`NotebookSimulator` — pure-AST + cache-probing replay of upstream
      cells; constructible with just a ``TrackingState`` + ``SimpleNamespace``.
      See ADR-009 for the test-seam rationale.

Everything else (:class:`VirtualLineage`, :class:`MismatchClassifier`,
:class:`ReexecutionPlanner`, the value-type IR in ``_types``) is internal to
this package. See ADR-010 for the package-extraction rationale.
"""

from __future__ import annotations

from .checker import UpstreamChecker, UpstreamResult
from .simulator import NotebookSimulator

__all__ = [
    "NotebookSimulator",
    "UpstreamChecker",
    "UpstreamResult",
]
