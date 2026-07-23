"""Custom exception hierarchy for Cash.

All Cash-specific exceptions derive from `CashError`, enabling
callers to handle any Cash failure with a single ``except CashError``
while still distinguishing between error categories when needed.
"""

from __future__ import annotations

from tokenize import TokenError

__all__ = [
    "SOURCE_RETRIEVAL_ERRORS",
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

# ---------------------------------------------------------------------------
# Source-retrieval failure modes
# ---------------------------------------------------------------------------

#: Every way ``inspect.getsource``/``getsourcelines`` can fail to hand back a
#: function's source.
#:
#: Wider than the documented ``OSError``/``TypeError`` because ``inspect`` does
#: not just read the file — it runs ``tokenize`` over it to find where the
#: function's block ends. So a file that is *not valid Python end-to-end* raises
#: ``tokenize.TokenError``, which derives straight from ``Exception`` and is
#: therefore caught by neither an ``OSError`` nor a ``SyntaxError`` handler.
#:
#: That is not exotic: it is any function whose ``co_filename`` names something
#: only partly parseable as Python — a docs page exec'd under its own path, a
#: Jupytext/Quarto literate source, a saved REPL transcript. Cash reaches for
#: source at decoration time, so an uncaught raise here aborts ``@cash.cache``
#: itself rather than degrading.
#:
#: Every call site guarded by this tuple already has a fallback (bytecode hash,
#: opaque identity, or a conservative "assume impure"), so widening it only
#: converts a crash into the degradation those sites were written to perform.
SOURCE_RETRIEVAL_ERRORS: tuple[type[BaseException], ...] = (
    OSError,            # file missing / not on disk (builtins, C extensions)
    TypeError,          # object has no source to speak of
    TokenError,         # file does not tokenize as Python (see above)
    SyntaxError,        # file tokenizes but does not parse
    IndentationError,   # SyntaxError subclass; listed for the reader
    UnicodeDecodeError, # ValueError subclass; file is not decodable text
    ValueError,         # e.g. inspect given a built-in with a bogus lineno
)


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
    """Raised on first call when caching cannot be guaranteed correct.

    Two triggers:

    * **By default** (plain ``@cash.cache``), when the function -- or a
      module-bounded helper it calls -- resolves a dependency from a runtime
      value that cash cannot track: ``getattr(obj, name)()`` dynamic dispatch,
      ``importlib.import_module(...)``, ``eval`` / ``exec`` / ``compile``
      (including a dynamic result stored in a local and then called). A cached
      result could go silently stale, so cash refuses rather than mis-cache.
    * **Under ``strict=True``**, additionally on any purity issue -- side
      effects, external-state mutation, opaque callees.

    The body lists each reason with line numbers. Cache it anyway (accepting the
    staleness risk) with ``@cash.cache(assume_safe=True)``, mark an audited
    callee with ``@cash.mark_pure(callee)``, or refactor to a static call.
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
