"""IPython/Jupyter notebook integration for statement-level caching.

The magics live in ``cash.notebook.ipython``, the statement processor in
``cash.notebook.statement`` and the upstream checker in
``cash.notebook.upstream``; import them from there. This package imports
only the small result types, so importing any ``cash.notebook`` module does
not load the whole notebook path.
"""

from __future__ import annotations

from . import memo_lifetimes  # noqa: F401 - lists this package's memos with `cash._memo_registry`
from .cache_status import CacheStatus, ExecutionResult

__all__ = [
    "CacheStatus",
    "ExecutionResult",
]
