"""Custom exception hierarchy for Cash.

All Cash-specific exceptions derive from `CashError`, enabling
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
    "CashImpureFunctionError",
    "CashWarning",
    "CashCacheIneffectiveWarning",
    "CashCacheStoreFailedWarning",
    "CashImpurityWarning",
    "CashUpstreamSyntaxWarning",
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

    Inherits from both `CashError` and `ImportError` so
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

class CashImpureFunctionError(CashError):
    """Raised by ``@cash.cache(strict=True)`` on first call when the
    decorated function (or one of its module-bounded helpers) has
    side effects, mutates external state, or uses explicit dynamism.

    The exception body lists each detected reason with line numbers
    so the user can fix the function or relax the analysis via
    ``assume_safe=True`` / ``@cash.mark_pure(callee)`` / refactoring.
    """

# ---------------------------------------------------------------------------
# Warnings (not errors — runtime advisories surfaced via warnings.warn)
# ---------------------------------------------------------------------------

class CashWarning(UserWarning):
    """Base class for all Cash-emitted warnings.

    Filter via:
        import warnings
        import cash
        warnings.filterwarnings("error", category=cash.CashWarning)
    """

class CashCacheIneffectiveWarning(CashWarning):
    """The cache is not doing anything useful for this call.

    Typical causes: unpicklable args with no registered hasher;
    dynamic dependency resolver raised; @cash.cache on an async
    generator; use_locking=True on an async function. The user's
    function ran (or will run) but its result is not being cached
    or re-used.
    """

class CashUpstreamSyntaxWarning(CashWarning):
    """An upstream notebook cell could not be parsed (a half-written cell the
    user has saved but not run).

    The unparseable cell is skipped so downstream cells that do not depend on
    it keep caching, but a cell that DID depend on it can no longer have its
    dependency tracked. Emitted by the notebook upstream checker naming the
    offending cell (1-based), so caching never silently stops mid-edit without
    telling the user why (CAS-173).
    """

class CashCacheStoreFailedWarning(CashWarning):
    """Compute succeeded but the backend rejected the write.

    Typical causes: serializer cannot handle the return type, disk
    full, Redis disconnected mid-set, S3 credential expiry.
    """

class CashImpurityWarning(CashCacheIneffectiveWarning):
    """The decorated function (or a module-bounded helper) has
    detected side effects, scope mutations, or explicit dynamism.

    Caching may still produce the "right" return value on a hit,
    but the side effect runs only on the first call. Common causes:
    network/file writes, ``logging``/``print``, mutation of globals,
    ``eval``/``exec``, calling a parameter as a function.

    Subclasses `CashCacheIneffectiveWarning` so existing
    ``warnings.filterwarnings('ignore', category=CashCacheIneffective\
Warning)`` filters continue to catch it. Filter more precisely with
    ``CashImpurityWarning`` directly.

    Promote to ``error`` in CI to fail the build when an impure
    function is cached without ``assume_safe=True``.
    """
