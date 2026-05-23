"""cash — smart caching library.

**Canonical import paths:**

- **Public stable API** (decorator caching, backends, configuration):
  Import from ``cash`` directly, e.g. ``from cash import Cash, pure``.

- **Notebook/Jupyter API** (IPython magics, cache status, statement processor):
  Import from ``cash.notebook``, e.g. ``from cash.notebook import CacheStatus``.

Purity decorators (``pure``, ``stateful``, ``is_pure``, ``is_stateful``) are
part of the public API and are re-exported here from ``cash.notebook.purity``
for convenience.  Either import path is valid for those symbols; prefer
``from cash import pure`` in application code.
"""
from __future__ import annotations

from .backends import CascadingBackend, FileBackend, InMemoryBackend
from .backends.sqlite_backend import SQLiteBackend
from .config import CashConfig, create_default_config, get_config
from .core import Cash
from .data_source import FileDataSource
from .exceptions import (
    AmbiguousCellError,
    CacheBackendError,
    CacheExpiredError,
    CacheKeyComputationError,
    CacheSerializationError,
    CashError,
    DependencyNotFoundError,
    UpstreamStateError,
)
from .notebook.purity import analyze_function_purity, is_pure, is_stateful, pure, stateful

__version__ = "0.5.0b1"

# Lazy-initialized global instance (created on first access)
_global_cash = None

def _get_global_cash():
    """Return the global ``Cash`` singleton, creating it on first call.

    The singleton avoids import-time side effects and supports the
    convenience API (``cash.cache``, ``cash.show_stats``,
    ``cash.register_hasher``) without requiring users to instantiate
    ``Cash`` themselves.  For custom configuration, create your own
    ``Cash(...)`` instance instead.
    """
    global _global_cash
    if _global_cash is None:
        _global_cash = Cash()
    return _global_cash


def reset_session() -> None:
    """Drop the global ``Cash`` singleton so the next access starts fresh.

    Use cases:

    * **Testing fixtures** that need cash to start over without
      restarting the Python interpreter.
    * **Benchmark harnesses** doing repeated measurements in one process
      (without this, cash's in-memory tracking dicts and FileAccess-
      Tracker monkey-patches survive ``shell.reset()`` and contaminate
      successive runs).
    * **Advanced users** who want to discard accumulated lineage state
      mid-session (e.g. before re-running a notebook against new inputs).

    What this does:

    * Sets ``_global_cash = None`` so the next ``cash.cache`` /
      ``cash.show_stats`` / ``%load_ext cash`` creates a fresh ``Cash``
      with empty tracking state.
    * If an IPython session is active, re-runs the auto-load so the
      ``%cash_on`` / ``%cash_off`` / ``%cash_stats`` magics rebind to
      the new singleton.

    What this does NOT do:

    * It doesn't clear the on-disk cache directory — that's a separate
      operation (``Cash().clear_cache()`` or ``%cash_clear``).
    * It doesn't unwind any ``FileAccessTracker`` monkey-patches that
      are currently in flight — those self-heal on the next tracker
      ``__enter__`` (see ``cash.notebook.file_tracker._unwrap_to_real``).
    """
    global _global_cash
    _global_cash = None
    # If IPython is active, re-register magics on a fresh instance so
    # existing ``%cash_*`` references resolve to the new singleton.
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
        if get_ipython() is not None:
            _get_global_cash().register_magic()
    except ImportError:
        pass


def __getattr__(name):
    """Proxy module-level attribute access to the global ``Cash`` singleton.

    Supported attributes: ``cache``, ``show_stats``, ``register_hasher``.
    These are created lazily on first access via :func:`_get_global_cash`.
    """
    if name == 'cache':
        return _get_global_cash().cache
    if name == 'show_stats':
        return _get_global_cash().show_stats
    if name == 'register_hasher':
        return _get_global_cash().register_hasher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core API (stable)
    "Cash",
    "cache",
    "show_stats",
    "register_hasher",
    "reset_session",
    # Purity declarations (stable)
    "pure",
    "stateful",
    "is_pure",
    "is_stateful",
    "analyze_function_purity",
    # Configuration (stable)
    "get_config",
    "CashConfig",
    "create_default_config",
    # Backends (stable)
    "InMemoryBackend",
    "FileBackend",
    "SQLiteBackend",
    "CascadingBackend",
    # Data sources (stable)
    "FileDataSource",
    # Exceptions (stable)
    "CashError",
    "CacheBackendError",
    "CacheExpiredError",
    "CacheSerializationError",
    "DependencyNotFoundError",
    "AmbiguousCellError",
    "UpstreamStateError",
    "CacheKeyComputationError",
]

# Experimental features are available via:
#   from cash.experimental import CacheExplorer, CacheDebugger, etc.

def load_ipython_extension(ipython):
    """Register the ``%%cash`` IPython magic on behalf of the global singleton.

    Called automatically by IPython/Jupyter when the user runs
    ``%load_ext cash`` or when cash is listed in ``ipython_config``.  Most
    users do not need to call this directly — plain ``import cash`` already
    registers the magics via :func:`_auto_load_in_ipython`, and is the
    recommended entry point because it also exposes ``@cash.cache`` for
    decorator-style caching.

    Delegates to :meth:`Cash.register_magic` on the global instance so
    that the magic shares the same backend and tracking state as any
    ``cash.cache`` calls in the same session.
    """
    _get_global_cash().register_magic()


def _auto_load_in_ipython() -> None:
    """Auto-register magics when cash is imported inside an IPython session.

    This means ``import cash`` is sufficient in a Jupyter notebook — no
    explicit ``%load_ext cash`` required, and the same single import enables
    both the ``%cash_on`` magic and ``@cash.cache`` decorator API.  The
    function is a no-op when running outside IPython (e.g. plain Python
    scripts).
    """
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        if ip is not None:
            load_ipython_extension(ip)
    except NameError:
        pass  # Not in an IPython session


_auto_load_in_ipython()
