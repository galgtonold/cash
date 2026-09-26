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
    "DependencyNotFoundError",
    "AmbiguousCellError",
    "ForwardReferenceError",
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
    OSError,  # file missing / not on disk (builtins, C extensions)
    TypeError,  # object has no source to speak of
    TokenError,  # file does not tokenize as Python (see above)
    SyntaxError,  # file tokenizes but does not parse
    IndentationError,  # SyntaxError subclass; listed for the reader
    UnicodeDecodeError,  # ValueError subclass; file is not decodable text
    ValueError,  # e.g. inspect given a built-in with a bogus lineno
)


class CashError(Exception):
    """Base class of every exception cash raises; catch it to catch them all."""


# ---------------------------------------------------------------------------
# Backend / storage errors
# ---------------------------------------------------------------------------


class CacheBackendError(CashError):
    """A backend could not read or write its storage (disk, SQLite, Redis, S3).

    Raised by backend methods you call directly. A cached function does not
    raise it for a failed write: it returns the result and warns with
    `CashCacheStoreFailedWarning`.
    """


class CacheSerializationError(CashError):
    """A stored entry could not be turned back into a value.

    Fix: clear the entry (``cash clear --entry ID``) or the function
    (``f.cache_clear()``); the next call recomputes it.
    """


# ---------------------------------------------------------------------------
# Dependency errors
# ---------------------------------------------------------------------------


class DependencyNotFoundError(CashError, ImportError):
    """A backend needs a package that is not installed.

    The message names the ``pip install`` command. Also an `ImportError`,
    so ``except ImportError`` catches it.
    """


# ---------------------------------------------------------------------------
# Notebook-specific errors
# ---------------------------------------------------------------------------


class AmbiguousCellError(CashError):
    """Notebook: the running cell's code appears more than once in the
    notebook, and cash cannot tell which copy is running.

    Shown as the cell's error. Fix: save the notebook (so cells carry ids),
    or make the duplicated cells differ.
    """


class UpstreamStateError(CashError):
    """Notebook: an earlier statement this cell needs could not be re-run.

    Shown as the cell's error, with the statement and its own error. Fix
    that statement, or run the notebook from the top.
    """


class ForwardReferenceError(CashError):
    """Notebook: a cell reads a name that only a later cell defines.

    It works in this kernel because the later cell already ran, but a run
    from the top would raise ``NameError``. Shown as the cell's error. Fix:
    move the definition above the cell that reads it.
    """


class CacheKeyComputationError(CashError):
    """Notebook: no cache key could be built for a statement.

    Not shown as an error: the statement runs without caching and cash
    warns with ``NOTEBOOK-BAILOUT``.
    """


class CashImpureFunctionError(CashError):
    """Decorator: raised on the first call when cash cannot cache the
    function safely.

    By default, raised when the function (or a helper in the same module)
    picks what to call or import at run time, in a way cash cannot track:
    ``getattr(obj, name)()``, ``importlib.import_module(...)``, ``eval``,
    ``exec``. With ``strict=True``, also raised for any other purity
    finding. The message lists each reason.

    Fix: call the code directly, mark an audited helper with `pure`, or
    accept the risk: for some lines with ``with cash.assume_safe():``, for
    the whole function with ``@cash.cache(assume_safe=True)``.
    """


# ---------------------------------------------------------------------------
# Warnings (not errors — runtime advisories surfaced via warnings.warn)
# ---------------------------------------------------------------------------


class CashWarning(UserWarning):
    """Base class of every warning cash emits.

    Each warning has a ``code`` attribute, and its message starts with
    ``[CODE]``; the Warnings page explains every code.
    """


class CashCacheIneffectiveWarning(CashWarning):
    """Caching is not working here, or not paying off.

    Covers results that cannot be cached (an unhashable argument, a value
    too large for any tier), caching that costs more than it saves, and
    configuration or annotation mistakes (``CONFIG-*`` and ``ANNOT-*``
    codes). The call still returns its result.
    """


class CashUpstreamSyntaxWarning(CashWarning):
    """Notebook: an earlier cell has a syntax error.

    Cash skips that cell. Cells that do not depend on it keep caching; the
    warning names the cell. Fix or run the cell to clear it.
    """


class CashCacheStoreFailedWarning(CashWarning):
    """The result was computed and returned, but could not be stored.

    Typical causes: a value that cannot be pickled, a full disk, a lost
    Redis connection, expired S3 credentials. The next call computes again.
    """


class CashImpurityWarning(CashCacheIneffectiveWarning):
    """Decorator: the function does something a cache hit will not repeat.

    For example it writes a file, prints, changes a global, reads the
    environment or the network, or calls ``eval``. On a hit the stored
    result is returned and none of that happens. Log calls are not
    reported. A subclass of `CashCacheIneffectiveWarning`.

    Fix: move the side effect out, mark an audited helper with `pure`, wrap
    the audited lines in ``with cash.assume_safe():``, or pass
    ``assume_safe=True``. Turn it into an error to fail CI.
    """
