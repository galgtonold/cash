"""Custom exception hierarchy for Cash.

All Cash-specific exceptions derive from :class:`CashError`, enabling
callers to handle any Cash failure with a single ``except CashError``
while still distinguishing between error categories when needed.
"""

from __future__ import annotations

__all__ = [
    "CashError",
    "CacheBackendError",
    "CacheSerializationError",
    "CacheExpiredError",
    "DependencyNotFoundError",
    "AmbiguousCellError",
    "UpstreamStateError",
    "CacheKeyComputationError",
]

class CashError(Exception):
    """Base exception for all Cash errors."""

# ---------------------------------------------------------------------------
# Backend / storage errors
# ---------------------------------------------------------------------------

class CacheBackendError(CashError):
    """Raised on backend I/O failures (disk, S3, Redis, SQLite)."""

class CacheSerializationError(CashError):
    """Raised when a value cannot be serialized or deserialized for caching."""

class CacheExpiredError(CashError):
    """Raised when a cache entry has exceeded its TTL."""

# ---------------------------------------------------------------------------
# Dependency errors
# ---------------------------------------------------------------------------

class DependencyNotFoundError(CashError, ImportError):
    """Raised when an optional backend dependency is missing.

    Inherits from both :class:`CashError` and :class:`ImportError` so
    that existing ``except ImportError`` handlers continue to work.
    """

# ---------------------------------------------------------------------------
# Notebook-specific errors
# ---------------------------------------------------------------------------

class AmbiguousCellError(CashError):
    """Raised when a notebook cell cannot be uniquely identified."""

class UpstreamStateError(CashError):
    """Raised when upstream cell state cannot be restored or simulated."""

class CacheKeyComputationError(CashError):
    """Raised when a cache key cannot be computed for a statement."""
