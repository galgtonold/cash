"""IPython/Jupyter notebook integration for statement-level caching.

The magics live in ``cash.notebook.ipython`` and the statement processor in
``cash.notebook.statement``; import them from there.
"""

from __future__ import annotations

from .cache_status import CacheStatus, ExecutionResult
from .upstream import UpstreamChecker

__all__ = [
    "CacheStatus",
    "ExecutionResult",
    "UpstreamChecker",
]
