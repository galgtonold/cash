"""cash — a Python cache that re-runs only what changed.

**Canonical import paths:**

- **Public stable API** (decorator caching, backends, configuration):
  Import from ``cash`` directly, e.g. ``from cash import Cash, pure``.

- **Notebook/Jupyter API** (cache status): import from
  ``cash.notebook``, e.g. ``from cash.notebook import CacheStatus``. The magics
  are in ``cash.notebook.ipython``, the statement processor in
  ``cash.notebook.statement``.

Purity decorators (``pure``, ``stateful``, ``is_pure``, ``is_stateful``) are
part of the public API; import them from ``cash``, e.g. ``from cash import pure``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from . import _active
from .backends import FileBackend, InMemoryBackend, TieredBackend
from .backends.sqlite_backend import SQLiteBackend
from .config import CashConfig, create_default_config, get_config
from .core import CacheExplanation, Cash
from .data_source import DataSource
from .exceptions import (
    AmbiguousCellError,
    CacheBackendError,
    CacheKeyComputationError,
    CacheSerializationError,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashError,
    CashImpureFunctionError,
    CashImpurityWarning,
    CashUpstreamSyntaxWarning,
    CashWarning,
    DependencyNotFoundError,
    ForwardReferenceError,
    UpstreamStateError,
)
from .file_source import FileDataSource
from .purity import is_pure, is_stateful, pure, stateful
from .remote_source import RemoteFileDataSource
from .tracking.file_tracker import install_read_watch
from .tracking.randomness import CashRandomnessWarning


def _watch_reads_from_import() -> None:
    """Credit file reads to the code that made them from ``import cash`` on.

    A config loader memoised with ``lru_cache`` is often first called while
    the app's modules are imported -- a banner, logging setup -- before any
    function is decorated, which is when the watch used to start. That read
    was nobody's, the cached function that later got the memoised config
    recorded no file, and an edit to the file was served the old result.
    Not with ``CASH_DISABLE`` set: caching off promises nothing is watched.
    """
    from .config import validate_value

    try:
        if validate_value("disable", os.environ.get("CASH_DISABLE", "").strip() or "0"):
            return
    except ValueError:
        pass  # the config load reports the bad value
    try:
        install_read_watch()
    except Exception:  # noqa: BLE001 - decorating a function installs it anyway
        pass


_watch_reads_from_import()


def opaque(cls: type) -> type:
    """Exclude this class's code from every cache key.

    Use it when a call takes an instance of the class, or the class itself,
    as an argument, but its methods' source should not take part in the key.
    Decorate a class you own with ``@cash.opaque``; for one you do not own
    (third-party, generated, vendored) call it on the class instead:
    ``cash.opaque(VendorWidget)``.

    Returns *cls* itself -- never a wrapper, so ``isinstance`` checks and
    identity comparisons against it keep working exactly as before. The class
    is recorded in a process-wide registry and is left unmodified, so this
    works on classes that refuse new attributes too.

    Example:

        import cash

        @cash.opaque
        class SlowToHashButStableConfig:
            ...

    Does NOT propagate to a subclass. A subclass may carry its own
    freshly-written methods the user actively edits, and inheriting opacity
    from an ancestor would silently exempt that new code from ever
    invalidating the cache. A subclass that wants the same treatment is
    marked itself.
    """
    Cash.mark_opaque(cls)
    return cls


__version__ = "0.11.0"


def _get_global_cash():
    """Return the global ``Cash`` singleton, creating it on first call.

    The singleton avoids import-time side effects and supports the
    convenience API (``cash.cache``, ``cash.show_stats``,
    ``cash.register_hasher``) without requiring users to instantiate
    ``Cash`` themselves.  For custom configuration, create your own
    ``Cash(...)`` instance instead. It is kept in ``cash._active``, where
    the modules below ``core`` read it.
    """
    instance = _active.default_cash()
    if instance is None:
        instance = Cash()
        _active.set_default_cash(instance)
    return instance


def reset_session() -> None:
    """Replace the default ``Cash`` with a fresh one, forgetting what the
    old one tracked in memory.

    For test fixtures and benchmarks that need cash to start over without
    restarting Python. Under IPython the magics are re-registered on the
    new instance. The cache on disk is kept; ``cash clear --all`` deletes
    it.
    """
    _active.set_default_cash(None)
    # If IPython is active, re-register magics on a fresh instance so
    # existing ``%cash_*`` references resolve to the new singleton.
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        if get_ipython() is not None:
            _get_global_cash().register_magic()
    except ImportError:
        pass


def configure(**overrides: Any) -> None:
    """Update the configuration of the default ``Cash`` singleton at runtime.

    ``cash.configure(debug=True)`` flips debug mode on every subsequent
    cache decision. ``cash.configure(backend="redis", redis_host="...")``
    swaps in a fresh Redis-backed backend, draining any pending writes
    on the previous one first.

    The kwargs are validated against the ``CashConfig`` fields (a bad key
    or value raises ``ValueError`` and changes nothing), then applied to
    the active config in place. When that changes the tiers the backend
    is built from (``cache_dir``, ``backend``, ``tiers``, the connection
    details of a tier in use, ...), the running backend drains its
    pending writes and a fresh one is built. A setting no tier uses --
    ``redis_host`` on a RAM + disk stack -- is stored for later and
    rebuilds nothing. ``min_cache_savings_pct`` reaches the running
    backend's persistence policy without a rebuild; every other setting
    is read by the next operation.

    Worker processes started afterwards (``multiprocessing``,
    ``concurrent.futures``, joblib) run with the same settings, whatever the
    start method; a worker already running keeps the ones it started with.

    This function never writes to disk. To persist changes across
    process invocations, edit ``pyproject.toml`` ``[tool.cash]`` or the
    XDG user config file directly.
    """
    if overrides:
        _get_global_cash().reconfigure(**overrides)


@contextlib.contextmanager
def disabled(on: bool = True) -> Iterator[None]:
    """Run the block with the default ``Cash`` switched off, then put it back.

    Inside ``with cash.disabled():`` every ``@cash.cache`` call runs its body
    and nothing is read from or written to the cache. On the way out the
    ``disable`` setting returns to what it was -- including ``True``, when the
    run was started with ``CASH_DISABLE=1``, which ``configure(disable=False)``
    on the way out would switch back on.

    ``disabled(False)`` is the reverse: force caching on for the block.

    Worker processes started inside the block run uncached too, and keep that
    setting for as long as they run.
    """
    c = _get_global_cash()
    previous = bool(c.config.disable)
    configure(disable=bool(on))
    try:
        yield
    finally:
        configure(disable=previous)


def cleanup(max_age: int | None = None) -> int:
    """Remove expired entries from the default ``Cash`` singleton's backend.

    Module-level convenience so the documented ``cash.cleanup()`` call works
    without constructing a ``Cash`` instance yourself — it proxies to
    :meth:`Cash.cleanup` on the global singleton.

    Args:
        max_age: If given, also remove entries older than *max_age* seconds,
            regardless of their stored TTL. ``None`` (default) removes only
            entries past their own TTL.

    Returns:
        Number of entries removed.
    """
    return _get_global_cash().cleanup(max_age)


# The coding-agent guide entry point: ``cash.help()`` prints/returns the compact
# reference (see ``_agent_guide.py``). Deliberately shadows the builtin at the
# ``cash.help`` path only, and is kept out of ``__all__`` so that a star import
# leaves the user's bare ``help()`` alone.
from ._agent_guide import help as help


def __getattr__(name):
    """Proxy module-level attribute access to the global ``Cash`` singleton.

    Supported attributes: ``cache``, ``show_stats``, ``register_hasher``.
    These are created lazily on first access via `_get_global_cash`.
    """
    if name == "cache":
        return _get_global_cash().cache
    if name == "show_stats":
        return _get_global_cash().show_stats
    if name == "register_hasher":
        return _get_global_cash().register_hasher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Served by the module ``__getattr__``, which builds the global ``Cash``.
_LAZY_NAMES = ("cache", "show_stats", "register_hasher")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_NAMES))


# Left out of ``__all__``: ``help``, so ``from cash import *`` does not replace
# the builtin, and the lazy names, so it does not build the global ``Cash``
# (load config, construct the backend). All of them are ``cash.<name>``.
__all__ = [
    # Core API (stable)
    "Cash",
    "CacheExplanation",
    "opaque",
    "reset_session",
    "configure",
    "disabled",
    "cleanup",
    # Purity declarations (stable)
    "pure",
    "stateful",
    "is_pure",
    "is_stateful",
    # Configuration (stable)
    "get_config",
    "CashConfig",
    "create_default_config",
    # Backends (stable)
    "InMemoryBackend",
    "FileBackend",
    "SQLiteBackend",
    "TieredBackend",
    # Data sources (stable)
    "DataSource",
    "FileDataSource",
    "RemoteFileDataSource",
    # Exceptions (stable)
    "CashError",
    "CacheBackendError",
    "CacheSerializationError",
    "DependencyNotFoundError",
    "AmbiguousCellError",
    "ForwardReferenceError",
    "UpstreamStateError",
    "CacheKeyComputationError",
    "CashImpureFunctionError",
    # Warnings (stable)
    "CashWarning",
    "CashCacheIneffectiveWarning",
    "CashCacheStoreFailedWarning",
    "CashImpurityWarning",
    "CashUpstreamSyntaxWarning",
    "CashRandomnessWarning",
]


def _jupyter_labextension_paths():
    """Where cash's prebuilt JupyterLab extension lives inside the package.

    The extension pushes the notebook's live (unsaved) cell sources over the
    ``cash_live_cells`` comm that `cash.notebook.live_cells` receives, which
    is the only way cash can see an edit that is not saved to the ``.ipynb``
    yet. Source and build instructions: ``labextension/`` in the repo.

    An ordinary ``pip install`` does **not** go through this hook -- the wheel
    drops the bundle straight into ``share/jupyter/labextensions/`` (see
    ``[tool.hatch.build.targets.wheel.shared-data]`` in ``pyproject.toml``),
    which is where JupyterLab looks. This exists for the *other* install
    route, ``jupyter labextension develop``, which symlinks the directory
    named here so a rebuild is picked up without reinstalling. Invoke it as
    ``cd src && jupyter labextension develop --overwrite cash`` -- from the
    repo root it would look for an importable module named after the
    *distribution* (``cash_lib``), which does not exist. The directory exists
    only in a checkout: the wheel ships the bundle once, through shared-data.

    ``src`` is relative to this package's directory; ``dest`` must equal the
    npm package name in ``labextension/package.json``, because federated
    module loading resolves the extension by that name.
    """
    return [{"src": "labextension", "dest": "cash-live-cells"}]


def load_ipython_extension(ipython):
    """Register cash's IPython magics on behalf of the global singleton.

    Called automatically by IPython/Jupyter when the user runs
    ``%load_ext cash`` or when cash is listed in ``ipython_config``.  Most
    users do not need to call this directly — plain ``import cash`` already
    registers the magics via `_auto_load_in_ipython`, and is the
    recommended entry point because it also exposes ``@cash.cache`` for
    decorator-style caching.

    Delegates to `Cash.register_magic` on the global instance so
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
        ip = get_ipython()  # type: ignore[name-defined]
        if ip is not None:
            load_ipython_extension(ip)
    except NameError:
        pass  # Not in an IPython session


_auto_load_in_ipython()
