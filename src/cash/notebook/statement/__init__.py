"""Statement-level caching subsystem.

`StatementProcessor` plus its four sibling classes (`CacheFreshnessChecker`,
`StatementFileDeps`, `StatementLineageBuilder`, `StatementRestorer`) share
one `TrackingState` and one `ProcessResult` schema. They are owned by
`StatementProcessor` via composition.

Public surface:
    - :class:`StatementProcessor` — orchestrator. Processes one statement:
      cacheability decision, key computation, cache lookup, execute-or-restore,
      lineage capture.
    - :class:`ProcessResult` — TypedDict returned from `StatementProcessor.process()`.
    - :class:`StatementCacheMetadata` — TypedDict stored alongside cached values.
    - :class:`DecoratorCallMetric` — TypedDict for tracking decorated function
      calls observed during statement execution.

Everything else (`CacheFreshnessChecker`, `StatementFileDeps`,
`StatementLineageBuilder`, `StatementRestorer`) is internal to this package.
See ADR-011 for the package-extraction rationale.

The file-snapshot helper (`snapshot_file_deps`) lives in
:mod:`cash.tracking.file_dep_snapshot`, not here: it has cross-subsystem
callers (the decorator path in ``src/cash/core.py``, ``Restorer``, and
``upstream/virtual_lineage.py``).
"""

from __future__ import annotations

# Re-exports kept for test files that patch / import via the package path
# (e.g. ``from cash.notebook.statement import TeeWriter``). The ones missing
# from ``__all__`` are not part of the public surface, but co-locating the
# re-export here keeps test paths stable.
from ._metadata import StatementCacheMetadata
from .capture import TeeWriter, tee_output  # noqa: F401
from .processor import StatementProcessor
from .results import DecoratorCallMetric, ProcessResult, ProcessResultRequired  # noqa: F401

__all__ = [
    "DecoratorCallMetric",
    "ProcessResult",
    "StatementCacheMetadata",
    "StatementProcessor",
]
