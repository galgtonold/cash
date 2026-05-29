from __future__ import annotations

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

The two file-snapshot helpers (`snapshot_file_deps`, `split_file_dep_value`)
that historically lived in `cache_freshness.py` were extracted to
:mod:`cash.notebook.file_dep_snapshot` before this package was formed —
they have cross-subsystem callers (the decorator path in
``src/cash/core.py``, ``Restorer``, and ``upstream/virtual_lineage.py``)
and don't belong inside ``statement/``.
"""

from .processor import (
    DecoratorCallMetric,
    ProcessResult,
    StatementCacheMetadata,
    StatementProcessor,
)

# Private re-exports kept for test files that patch / import via the package
# path (e.g. ``from cash.notebook.statement import _TeeWriter``). These are
# not part of the public surface — the leading underscore is the signal —
# but co-locating the re-export here keeps test paths stable.
from .processor import _ProcessResultRequired, _TeeWriter, _tee_output  # noqa: F401

__all__ = [
    "DecoratorCallMetric",
    "ProcessResult",
    "StatementCacheMetadata",
    "StatementProcessor",
]
