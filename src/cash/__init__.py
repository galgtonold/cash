"""cash — a Python cache that re-runs only what changed.

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

from typing import Any

from .backends import CascadingBackend, FileBackend, InMemoryBackend
from .backends.sqlite_backend import SQLiteBackend
from .config import CashConfig, create_default_config, get_config
from .core import Cash, CacheExplanation
from .data_source import DataSource, FileDataSource
from .remote_source import RemoteFileDataSource
from .exceptions import (
    AmbiguousCellError,
    CacheBackendError,
    CacheExpiredError,
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
    UpstreamStateError,
)
from .notebook.purity import analyze_function_purity, is_pure, is_stateful, pure, stateful
from .notebook.randomness import CashRandomnessWarning

# ---------------------------------------------------------------------------
# Annotation helpers for third-party callables
# ---------------------------------------------------------------------------

def mark_pure(func: Any) -> Any:
    """Mark *func* as pure for cash's purity analyzer.

    Use this on library functions you've audited and want the
    analyzer to trust. Sets the same ``_cash_pure`` attribute the
    `pure` decorator sets, so it composes with the existing
    annotation system. Returns *func* unmodified (no wrapping).

    Unlike `pure`, this does not wrap the function — the
    attribute is set directly on the passed object. This matters for
    third-party callables that may not survive wrapping (C extensions,
    callable instances with strict signatures).

    Example:

        import pandas as pd
        import cash

        cash.mark_pure(pd.DataFrame.merge)

    Returns:
        The same callable, with ``_cash_pure = True`` attached if
        possible. Silently no-ops on objects that don't allow
        attribute setting.
    """
    try:
        setattr(func, '_cash_pure', True)
    except (AttributeError, TypeError):
        pass
    return func


def mark_stateful(func: Any) -> Any:
    """Mark *func* as stateful (side-effecting) for cash's purity analyzer.

    Tells the analyzer to flag any call to *func* as an impure call
    when analyzing a cached function. Sets the same
    ``_cash_stateful`` attribute the `stateful` decorator sets.

    Use on library functions whose impurity the analyzer can't see
    (C extensions, database drivers, etc.) so that calling them
    from a ``@cash.cache``-decorated function triggers the warning.

    Example:

        import pandas as pd
        import cash

        cash.mark_stateful(pd.DataFrame.to_sql)

    Returns:
        The same callable, with ``_cash_stateful = True`` attached
        if possible. Silently no-ops on objects that don't allow
        attribute setting.
    """
    try:
        setattr(func, '_cash_stateful', True)
    except (AttributeError, TypeError):
        pass
    return func


def opaque(cls: type) -> type:
    """Class decorator: exclude this class's code from every cache key.

    Use on a class you own -- when a call takes an instance of it, or the
    class itself, as an argument -- but don't want its methods' source
    participating in the cache key. For a class you do NOT own (third-party,
    generated, vendored) decorating it isn't an option; register it from
    outside with `mark_opaque` instead. The two are the same marker spelled
    two ways for the two kinds of owner.

    Returns *cls* itself, unchanged apart from one new attribute
    (``__cash_opaque__``) -- never a wrapper, so ``isinstance`` checks and
    identity comparisons against the decorated class keep working exactly
    as before.

    Example:

        import cash

        @cash.opaque
        class SlowToHashButStableConfig:
            ...

    Does NOT propagate to a subclass. A subclass may carry its own
    freshly-written methods the user actively edits, and inheriting opacity
    from an ancestor would silently exempt that new code from ever
    invalidating the cache. A subclass that wants the same treatment
    decorates itself.
    """
    Cash.mark_opaque(cls)
    cls.__cash_opaque__ = True
    return cls


mark_opaque = Cash.mark_opaque

__version__ = "0.9.0"

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
    * It doesn't touch the ``FileAccessTracker`` dispatcher wrappers
      installed on ``builtins.open``, ``pandas.read_csv``, and other
      tracked I/O entry points. Those are permanent for the process
      lifetime and tracker-agnostic — they no-op when no tracker is
      active (see ``cash.notebook.file_tracker._active_tracker``).
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


def configure(**overrides: Any) -> None:
    """Update the configuration of the default ``Cash`` singleton at runtime.

    ``cash.configure(debug=True)`` flips debug mode on every subsequent
    cache decision. ``cash.configure(backend="redis", redis_host="...")``
    swaps in a fresh Redis-backed backend, draining any pending writes
    on the previous one first.

    Resolution semantics — the kwargs are validated against the
    ``CashConfig`` field set, then applied to the active config in
    place. Fields that affect backend construction (cache_dir,
    max_cache_size, max_memory_entries, flush_interval, compress,
    backend, tiers, and all per-backend connection details) trigger:

        1. ``current_backend.shutdown()`` — drain in-flight writes.
        2. Build a fresh backend from the updated config.
        3. Swap it in atomically.

    Hot fields (debug, smart-persistence policy knobs) are applied to
    the dataclass in place and read by the next operation. They do not
    rebuild the backend.

    Stale fields (e.g. ``redis_host`` when there's no Redis tier
    currently active) are stored silently — the next switch to a Redis
    backend will pick them up.

    This function never writes to disk. To persist changes across
    process invocations, edit ``pyproject.toml`` ``[tool.cash]`` or the
    XDG user config file directly.
    """
    if not overrides:
        return
    from dataclasses import fields
    from .config import CashConfig

    valid_fields = {f.name for f in fields(CashConfig) if not f.name.startswith("_")}
    unknown = set(overrides) - valid_fields
    if unknown:
        raise ValueError(
            f"{sorted(unknown)!r} is not a configurable field. "
            f"Valid keys: {sorted(valid_fields)!r}"
        )

    c = _get_global_cash()

    # Hot vs backend-affecting fields. Anything that influences which
    # concrete backend(s) get constructed needs a rebuild.
    BACKEND_AFFECTING = {
        "cache_dir", "compress", "max_cache_size", "max_memory_entries",
        "flush_interval", "backend", "tiers",
        "redis_host", "redis_port", "redis_db", "redis_password", "redis_prefix",
        "s3_bucket", "s3_region", "s3_prefix",
    }

    needs_rebuild = bool(set(overrides) & BACKEND_AFFECTING)

    # If the user changed a backend-affecting field, decide WHETHER the
    # change actually affects the currently-active backend. If no active
    # tier uses the field (e.g. setting redis_host while running RAM+disk),
    # we skip the rebuild — the value is stored on the dataclass for later.
    if needs_rebuild and not _change_affects_active_backend(c, set(overrides)):
        needs_rebuild = False

    # Apply overrides to the config dataclass in place.
    for key, val in overrides.items():
        setattr(c.config, key, val)

    # Propagate the `debug` flag onto the Cash instance attribute too —
    # statement_processor reads it from there in some paths.
    if "debug" in overrides:
        c.debug = bool(overrides["debug"])

    if needs_rebuild:
        from .backends.factory import build_backend_from_config
        old_backend = c._backend
        if old_backend is not None:
            try:
                old_backend.shutdown()
            except Exception as e:  # noqa: BLE001 — best-effort drain
                import logging
                logging.getLogger(__name__).warning(
                    "Old backend shutdown failed during configure(): %s", e,
                )
        c._backend = build_backend_from_config(c.config)


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


def _change_affects_active_backend(c: Cash, changed: set[str]) -> bool:
    """Decide whether *changed* config fields would change the active stack.

    Conservatively returns True for any non-connection field (cache_dir,
    compress, etc.). For connection fields (redis_*, s3_*) it inspects
    the current backend stack — if no live tier of that type is present,
    the change is dormant.
    """
    # Fields that always force rebuild when changed.
    UNCONDITIONAL = {
        "cache_dir", "compress", "max_cache_size", "max_memory_entries",
        "flush_interval", "backend", "tiers",
    }
    if changed & UNCONDITIONAL:
        return True

    # Connection fields: only relevant if a tier of that backend is live.
    redis_fields = {"redis_host", "redis_port", "redis_db", "redis_password", "redis_prefix"}
    s3_fields = {"s3_bucket", "s3_region", "s3_prefix"}

    def _has_tier_type(backend, type_name: str) -> bool:
        if backend is None:
            return False
        class_name = type(backend).__name__.lower()
        if class_name.startswith(type_name):
            return True
        # TieredBackend / CascadingBackend: walk children.
        children = getattr(backend, "backends", None)
        if children:
            return any(_has_tier_type(child, type_name) for child in children)
        return False

    if changed & redis_fields and _has_tier_type(c._backend, "redis"):
        return True
    if changed & s3_fields and _has_tier_type(c._backend, "s3"):
        return True
    return False


# The coding-agent guide entry point: ``cash.help()`` prints/returns the compact
# reference (see ``_agent_guide.py``). Deliberately shadows the builtin at the
# ``cash.help`` path only; user code's bare ``help()`` is unaffected.
from ._agent_guide import help  # noqa: A004,E402


def __getattr__(name):
    """Proxy module-level attribute access to the global ``Cash`` singleton.

    Supported attributes: ``cache``, ``show_stats``, ``register_hasher``.
    These are created lazily on first access via `_get_global_cash`.
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
    "CacheExplanation",
    "cache",
    "show_stats",
    "register_hasher",
    "opaque",
    "mark_opaque",
    "help",
    "reset_session",
    "configure",
    "cleanup",
    # Purity declarations (stable)
    "pure",
    "stateful",
    "is_pure",
    "is_stateful",
    "analyze_function_purity",
    "mark_pure",
    "mark_stateful",
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
    "DataSource",
    "FileDataSource",
    "RemoteFileDataSource",
    # Exceptions (stable)
    "CashError",
    "CacheBackendError",
    "CacheExpiredError",
    "CacheSerializationError",
    "DependencyNotFoundError",
    "AmbiguousCellError",
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

# Experimental features are available via:
#   from cash.experimental import CacheExplorer, CacheDebugger, etc.


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
    *distribution* (``cash_lib``), which does not exist.

    ``src`` is relative to this package's directory; ``dest`` must equal the
    npm package name in ``labextension/package.json``, because federated
    module loading resolves the extension by that name.
    """
    return [{"src": "labextension", "dest": "cash-live-cells"}]


def load_ipython_extension(ipython):
    """Register the ``%%cash`` IPython magic on behalf of the global singleton.

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
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        if ip is not None:
            load_ipython_extension(ip)
    except NameError:
        pass  # Not in an IPython session


_auto_load_in_ipython()
