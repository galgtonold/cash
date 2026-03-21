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

from .backends import AsyncBackendWrapper, CascadingBackend, FileBackend, InMemoryBackend
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
    "AsyncBackendWrapper",
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
    ``%load_ext cash`` or when cash is listed in ``ipython_config``.
    Delegates to :meth:`Cash.register_magic` on the global instance so
    that the magic shares the same backend and tracking state as any
    ``cash.cache`` calls in the same session.
    """
    _get_global_cash().register_magic()


def _auto_load_in_ipython() -> None:
    """Auto-register magics when cash is imported inside an IPython session.

    This means ``import cash`` is sufficient in a Jupyter notebook — no
    explicit ``%load_ext cash`` required.  The function is a no-op when
    running outside IPython (e.g. plain Python scripts).
    """
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        if ip is not None:
            load_ipython_extension(ip)
    except NameError:
        pass  # Not in an IPython session


_auto_load_in_ipython()
