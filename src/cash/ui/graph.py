"""Variable dependency graph for visualising lineage relationships.

Re-exports :class:`~cash.graph.DependencyGraph` for backwards compatibility.
The class now lives in ``cash.graph``; this module is kept as a shim so that
existing code importing from ``cash.ui.graph`` continues to work unchanged.
"""

from __future__ import annotations

from cash.graph import DependencyGraph  # noqa: F401  (re-export)

__all__ = ["DependencyGraph"]

# ---------------------------------------------------------------------------
# DEPRECATED: the class definition below has been moved to cash.graph.
# This file is kept only for backward-compatible imports.
# ---------------------------------------------------------------------------


