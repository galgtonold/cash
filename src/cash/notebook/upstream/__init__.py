from __future__ import annotations

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

from .checker import UpstreamChecker, UpstreamResult
from .simulator import NotebookSimulator

# Private re-exports kept for test files that import them via the package
# path (e.g. ``from cash.notebook.upstream import _SimulationCacheEntry``).
# These are not part of the public surface — the leading underscore is the
# signal — but co-locating the re-export here keeps test paths stable.
from .checker import _FORWARD_PROBE_PLACEHOLDER, _SimulationCacheEntry  # noqa: F401

__all__ = [
    "NotebookSimulator",
    "UpstreamChecker",
    "UpstreamResult",
]
