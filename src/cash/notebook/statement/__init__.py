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

Everything else in this package is internal to it: code outside it imports
only the names above.

The file-snapshot helper (`snapshot_file_deps`) lives in
:mod:`cash.tracking.file_dep_snapshot`, not here: it has cross-subsystem
callers (the decorator path in ``src/cash/core.py`` and
``upstream/virtual_lineage.py``).

The names are imported when first asked for (`__getattr__`), so a module of
this package the decorator path reads (`identity_holders`) is imported without
the processor and everything it imports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._metadata import StatementCacheMetadata
    from .control_body import is_control_body
    from .processor import StatementProcessor
    from .results import DecoratorCallMetric, ProcessResult

#: Each public name, and the module of this package it is defined in.
_HOMES = {
    "DecoratorCallMetric": ".results",
    "ProcessResult": ".results",
    "StatementCacheMetadata": "._metadata",
    "StatementProcessor": ".processor",
    "is_control_body": ".control_body",
}

__all__ = [
    "DecoratorCallMetric",
    "ProcessResult",
    "StatementCacheMetadata",
    "StatementProcessor",
    "is_control_body",
]


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(home, __name__), name)
    globals()[name] = value
    return value
