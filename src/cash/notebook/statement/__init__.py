"""Statement-level caching subsystem.

:class:`StatementProcessor` runs one statement at a time -- cacheability
decision, key computation, cache lookup, execute-or-restore, lineage capture
-- through collaborators it composes (the store, the hit server, the call
router, the randomness, mutation, rebuild-cost and provenance helpers). They
share one ``TrackingState`` and report through one :class:`ProcessResult`.

Public surface:
    - :class:`StatementProcessor` -- the orchestrator, and the hooks the cell
      executor, the control-structure handlers and the magics call on it.
    - :class:`ProcessResult` / :class:`DecoratorCallMetric` -- what a
      processed statement reports.
    - :class:`StatementCacheMetadata` -- the metadata stored with an entry.
    - :func:`is_control_body` -- whether a statement is one statement of a
      loop or branch body rather than a cell-level statement.

Everything else in this package is internal to it. See ADR-011 for the
package-extraction rationale.

The file-snapshot helper (`snapshot_file_deps`) lives in
:mod:`cash.tracking.file_dep_snapshot`, not here: it has cross-subsystem
callers (the decorator path in ``src/cash/core.py``, ``Restorer``, and
``upstream/virtual_lineage.py``).
"""

from __future__ import annotations

from ._metadata import StatementCacheMetadata
from .processor import StatementProcessor, is_control_body
from .results import DecoratorCallMetric, ProcessResult

__all__ = [
    "DecoratorCallMetric",
    "ProcessResult",
    "StatementCacheMetadata",
    "StatementProcessor",
    "is_control_body",
]
