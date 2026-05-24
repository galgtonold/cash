"""Main Cash class — decorator-based caching with automatic dependency tracking.

Provides the :class:`Cash` entry point for ``@cash.cache`` function-level
caching and the :meth:`Cash.notebook` bridge for Jupyter integration.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import inspect
import logging
import pickle
import threading
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from .backends import CacheBackend, CacheMetadata, CascadingBackend, FileBackend, InMemoryBackend
from .backends.factory import build_backend_from_config

if TYPE_CHECKING:
    from .ui.explorer import CacheExplorer
from .backends.serialization import get_serializer
from .backends.tiered_backend import TieredBackend
from .config import CashConfig, get_config
from .data_source import DataSource
from .exceptions import (
    CacheExpiredError,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
)
from .graph import DependencyGraph
from .notebook.analysis import CodeAnalyzer

# Configure Logging
logger = logging.getLogger(__name__)

# Sentinel object used by wrapper helpers to signal a cache miss without
# conflicting with any legitimate cached value (including None).
_CACHE_MISS = object()

P = ParamSpec("P")
T = TypeVar("T")

__all__ = ["Cash"]

class Cash:
    """Smart caching framework for Python functions and Jupyter notebooks.

    Provides decorator-based caching with automatic dependency tracking,
    file dependency monitoring, and pluggable storage backends.

    By default (no ``backend`` or ``cache_dir`` specified), uses a
    TieredBackend with L1 in-memory + L2 file-based storage in a local
    ``.cash`` directory.  Pass ``backend=`` to override.

    Args:
        backend: A specific cache backend instance to use.
        cache_dir: Path to cache directory (creates TieredBackend automatically).
        backends: List of backends for cascading cache (L1/L2/L3).
        compress: Enable gzip compression for file-based caching.
        register_magic: Register IPython magic commands (default True).
        debug: Enable debug logging output.
        use_locking: Enable double-checked locking for thread-safe caching.
        config_path: Path to custom config TOML file.

    Example::

        from cash import Cash
        c = Cash()

        @c.cache
        def expensive(x):
            return x ** 2
    """

    graph: DependencyGraph
    functions: dict[str, Callable[..., Any]]
    data_sources: dict[str, DataSource]
    source_hashes: dict[str, str]
    debug: bool | None
    use_locking: bool
    config: CashConfig

    def __init__(
        self,
        backend: CacheBackend | None = None,
        cache_dir: str | None = None,
        backends: list[CacheBackend] | None = None,
        compress: bool | None = None,
        register_magic: bool = True,
        debug: bool | None = None,
        use_locking: bool = False,
        config_path: str | None = None,
        **config_overrides: Any,
    ) -> None:
        # Map the legacy explicit kwargs into the overrides dict so the
        # config layer treats them with the same priority as any other
        # constructor-supplied override (highest).
        for key, val in (("cache_dir", cache_dir), ("compress", compress), ("debug", debug)):
            if val is not None:
                config_overrides.setdefault(key, val)
        self.config = get_config(config_path=config_path, overrides=config_overrides or None)

        debug = self.config.debug

        # Store params for lazy backend construction. If an explicit
        # backend (or list of backends) was provided, that wins — those
        # are concrete objects, not config — and we skip the factory.
        self._backend: CacheBackend | None = None
        self._explicit_backends = backends  # remembered for repr / debugging only
        if backend is not None:
            self._backend = backend
        elif backends:
            if len(backends) > 1:
                self._backend = CascadingBackend(backends)
            else:
                self._backend = backends[0]

        self._backend_lock = threading.Lock()

        self.graph = DependencyGraph()
        self.functions: dict[str, Callable[..., Any]] = {} # Registry of cached functions
        self.data_sources: dict[str, DataSource] = {} # Registry of data sources
        self.source_hashes: dict[str, str] = {} # Current source hashes
        self._analyzed = set() # Track which functions we've analyzed for dependencies
        self._func_key_cache: dict[int, str] = {}  # id(func) -> module-qualified key
        self.debug = debug  # Debug mode flag
        self.use_locking = use_locking

        # Decorator call log for notebook integration.
        # Each entry is a dict with: func_name, cache_hit (bool), execution_time,
        # args_hash, cache_key, timestamp.  The notebook statement processor
        # drains this list after executing each statement so it can include
        # decorator metrics in the badge.
        self._decorator_call_log: list[dict[str, Any]] = []
        self._decorator_call_log_lock = threading.Lock()
        # Custom type hasher registry: maps type → callable(value) → str
        self._type_hashers: dict[type, Callable[[Any], str]] = {}

        # Dedup keys for _warn_once: (category, func_name, arg_type_name).
        # Guarded by _decorator_call_log_lock (already exists for thread safety).
        self._warning_keys_seen: set[tuple[type[Warning], str, str]] = set()

        atexit.register(self.shutdown)

        if register_magic:
            self.register_magic()

    @property
    def backend(self) -> CacheBackend:
        """Lazily build the cache backend from ``self.config`` on first access.

        This avoids filesystem I/O, thread creation, and directory
        scanning at ``Cash()`` construction time. The heavy lifting
        happens only when the cache is actually used.
        """
        if self._backend is not None:
            return self._backend
        with self._backend_lock:
            if self._backend is not None:
                return self._backend
            self._backend = build_backend_from_config(self.config)
            return self._backend

    @backend.setter
    def backend(self, value: CacheBackend) -> None:
        """Allow direct assignment (e.g. ``c.backend = MyBackend()``)."""
        self._backend = value

    def __repr__(self) -> str:
        backend_name = type(self._backend).__name__ if self._backend is not None else '<deferred>'
        n_funcs = len(self.functions)
        return f"Cash(backend={backend_name}, functions={n_funcs}, debug={self.debug})"

    @staticmethod
    def _get_func_key(func: Callable) -> str:
        """Return a module-qualified key for a function.

        Uses ``func.__module__ + '.' + func.__qualname__`` to avoid collisions
        when different modules define functions with the same ``__qualname__``
        (e.g. a notebook's ``dep()`` vs a library module's ``dep()``).
        """
        module = getattr(func, '__module__', None) or '__unknown__'
        qualname = getattr(func, '__qualname__', None) or func.__name__
        return f"{module}.{qualname}"


    @overload
    def cache(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def cache(
        self,
        func: None = None,
        *,
        depends_on: list[Callable[..., Any] | DataSource] | None = ...,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = ...,
        file_depends_on: str | list[str] | None = ...,
        ttl: int | None = ...,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def cache(
        self,
        func: Callable[P, T] | None = None,
        *,
        depends_on: list[Callable[..., Any] | DataSource] | None = None,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        file_depends_on: str | list[str] | None = None,
        ttl: int | None = None,
    ) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator to cache a function's return value.

        Can be used with or without arguments::

            @c.cache
            def f(x): ...

            @c.cache(ttl=3600)
            def g(x): ...

            @c.cache(file_depends_on="data.csv")
            def load_data():
                return pd.read_csv("data.csv")

        Args:
            func: The function to cache (set automatically when used without parens).
            depends_on: Static dependencies (functions or DataSources) to include
                in the cache key.
            dynamic_depends_on: Callable(s) that receive the same args as the
                decorated function and return DataSource(s) for cache key.
            file_depends_on: File path(s) to track as dependencies. The cache
                is automatically invalidated when any tracked file changes.
                Shorthand for ``depends_on=[FileDataSource("path")]``.
            ttl: Time-to-live in seconds. ``None`` means never expires.

        Returns:
            The decorated function with caching behavior.

        See Also:
            ``docs/caching-class-methods.md`` for the recipe for caching
            methods on stateful objects (databases, file handles,
            connections) via :meth:`register_hasher`.
        """
        if func is None:
            return lambda f: self.cache(
                f, depends_on=depends_on, dynamic_depends_on=dynamic_depends_on,
                file_depends_on=file_depends_on, ttl=ttl,
            )

        func_name = self._register_func(func, depends_on, file_depends_on)

        # Async generators are not cached; warn once and return unwrapped.
        if inspect.isasyncgenfunction(func):
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: async generators are not "
                f"cached in this release. The function is returned unwrapped.",
                stacklevel=3,
            )
            return func

        if inspect.iscoroutinefunction(func):
            wrapper = self._make_async_wrapper(func, func_name, dynamic_depends_on, ttl)
        else:
            wrapper = self._make_wrapper(func, func_name, dynamic_depends_on, ttl)
        return self._wrap_with_stats(func, func_name, wrapper)

    def _register_func(
        self,
        func: Callable,
        depends_on: list[Callable[..., Any] | DataSource] | None,
        file_depends_on: str | list[str] | None,
    ) -> str:
        """Register a function in the cache graph and return its key."""
        func_name = self._get_func_key(func)
        self.functions[func_name] = func
        new_hash = CodeAnalyzer.get_source_hash(func)
        old_hash = self.source_hashes.get(func_name)
        if old_hash and old_hash != new_hash:
            self._analyzed.discard(func_name)
        self.source_hashes[func_name] = new_hash
        self.graph.add_node(func_name)
        self._register_static_dependencies(func_name, depends_on)
        if file_depends_on:
            from .data_source import FileDataSource
            file_paths = [file_depends_on] if isinstance(file_depends_on, str) else file_depends_on
            file_deps = [FileDataSource(p) for p in file_paths]
            self._register_static_dependencies(func_name, file_deps)
        return func_name

    def _resolve_cache_key(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
        call_start: float,
    ) -> Any:
        """Compute and return (cache_key, current_state_hash, args_hash) or call func directly on failure.

        Returns a 3-tuple on success, or a callable-result sentinel tuple
        ``(_CACHE_MISS, None, None)`` when args cannot be hashed, or raises
        nothing (logs and returns func result wrapped in a tuple) on key error.
        Actually returns either:
          - (cache_key_str, state_hash_str, args_hash_str)  — normal
          - (_CACHE_MISS, result, 'unhashable')             — unhashable args
          - (_CACHE_MISS, result, 'error')                  — key generation error
        """
        try:
            current_state_hash = self._get_dependency_state_hash(func_name, set())
            dynamic_state_hash = self._resolve_dynamic_dependencies(func_name, dynamic_depends_on, args, kwargs)
            args_hash = self._serialize_args(func_name, args, kwargs)
            if args_hash is None:
                arg_type_name = self._first_unhashable_arg_type(args, kwargs)
                if arg_type_name == "<unknown>":
                    suggestion = (
                        "Cash could not identify which argument is unhashable "
                        "(likely a nested value inside a container). Try passing "
                        "a simpler argument, or register a hasher for the offending "
                        "type via cash.register_hasher(SomeType, ...)."
                    )
                else:
                    suggestion = (
                        f"Register a hasher via cash.register_hasher({arg_type_name}, ...) "
                        f"or pass the argument by a hashable value."
                    )
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    arg_type_name,
                    f"@cash.cache on {func_name}: failed to build cache key from "
                    f"argument of type {arg_type_name}. Call will not cache. {suggestion}",
                )
                result = func(*args, **kwargs)
                self._log_decorator_call(func_name, cache_hit=False, execution_time=time.perf_counter() - call_start, args_hash='unhashable', cache_key='')
                return (_CACHE_MISS, result, 'unhashable')
            cache_key = self._compute_cache_key(func_name, current_state_hash, dynamic_state_hash, args_hash)
            return (cache_key, current_state_hash, args_hash)
        except (TypeError, ValueError, pickle.PicklingError, AttributeError) as e:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            if arg_type_name == "<unknown>":
                hint = (
                    " Cash could not identify the offending argument type; "
                    "check your function's arguments."
                )
            else:
                hint = (
                    f" Consider cash.register_hasher({arg_type_name}, ...) "
                    f"if {arg_type_name} is the unhashable argument."
                )
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                arg_type_name,
                f"@cash.cache on {func_name}: cache-key generation raised "
                f"{type(e).__name__} ({e}). Call will not cache.{hint}",
            )
            result = func(*args, **kwargs)
            self._log_decorator_call(func_name, cache_hit=False, execution_time=time.perf_counter() - call_start, args_hash='error', cache_key='')
            return (_CACHE_MISS, result, 'error')

    @staticmethod
    def _first_unhashable_arg_type(args: tuple, kwargs: dict) -> str:
        """Return the qualname of the first non-built-in arg type, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. Skips strings,
        ints, floats, bools, None, lists, dicts, tuples, sets — they're
        always picklable, so they're never the culprit. The first non-builtin
        wins; this is heuristic but matches the most common single-bad-arg
        case.
        """
        BUILTIN_OK = (str, int, float, bool, type(None), bytes, list, dict, tuple, set, frozenset)
        for a in args:
            if not isinstance(a, BUILTIN_OK):
                return type(a).__qualname__
        for v in kwargs.values():
            if not isinstance(v, BUILTIN_OK):
                return type(v).__qualname__
        return "<unknown>"

    def _try_get_cached(
        self,
        cache_key: str,
        metadata: Any,
        cached_data: Any,
        call_start: float,
        args_hash: str,
        func_name: str,
        ttl: int | None,
    ) -> Any:
        """Return cached_data if valid, else _CACHE_MISS sentinel.

        Key-presence is determined by ``metadata is not None`` — the
        backend contract is that absent keys return ``(None, None)``,
        so a non-None metadata dict with a ``None`` data value still
        counts as a hit (a function that legitimately returned ``None``).

        Auto-tracked file dependencies stored in
        ``metadata['auto_file_deps']`` are re-stat'd here; any file with
        a different mtime/size than recorded forces a miss so the
        function re-reads the changed file.
        """
        if metadata is not None:
            try:
                self._validate_ttl(metadata, ttl)
                if not self._auto_file_deps_fresh(metadata):
                    return _CACHE_MISS
                self._log_decorator_call(func_name, cache_hit=True, execution_time=time.perf_counter() - call_start, args_hash=args_hash, cache_key=cache_key)
                return cached_data
            except CacheExpiredError:
                pass
            except (TypeError, KeyError):
                logger.debug("Cache hit but validation failed for %s", func_name)
        return _CACHE_MISS

    @staticmethod
    def _auto_file_deps_fresh(metadata: dict[str, Any]) -> bool:
        """Return True if every file recorded in ``metadata['auto_file_deps']``
        still has the same (mtime, size) on disk.

        Auto-tracked deps are captured during the first compute via
        :class:`cash.notebook.file_tracker.FileAccessTracker` and stored
        as ``{path: {'mtime': float, 'size': int}}``. If a recorded path
        is gone or its (mtime, size) changed, we invalidate the cache
        so the next compute re-reads the file. A path that disappears
        is also a change.
        """
        snap = metadata.get('auto_file_deps') or {}
        if not snap:
            return True  # nothing to check
        import os
        for path, recorded in snap.items():
            try:
                st = os.stat(path)
            except OSError:
                return False  # file vanished — invalidate
            if st.st_mtime != recorded.get('mtime') or st.st_size != recorded.get('size'):
                return False
        return True

    def _make_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl: int | None,
    ) -> Callable:
        """Build and return the core caching wrapper for *func*."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_start = time.perf_counter()

            if func_name not in self._analyzed:
                self._analyze_dependencies(func)
                self._analyzed.add(func_name)

            key_result = self._resolve_cache_key(func, func_name, dynamic_depends_on, args, kwargs, call_start)
            if key_result[0] is _CACHE_MISS:
                return key_result[1]
            cache_key, current_state_hash, args_hash = key_result

            metadata, cached_data = self.backend.get(cache_key)
            hit = self._try_get_cached(cache_key, metadata, cached_data, call_start, args_hash, func_name, ttl)
            if hit is not _CACHE_MISS:
                return hit

            def _compute_and_store() -> Any:
                # Wrap the function call in FileAccessTracker so any
                # auto-tracked file reads (pandas/numpy/joblib/open/...)
                # are recorded as implicit cache dependencies — a later
                # mtime/size change forces a recompute.
                from cash.notebook.file_tracker import FileAccessTracker
                from cash.notebook.cache_freshness import snapshot_file_deps
                tracker = FileAccessTracker(getattr(func, '__globals__', None))
                with tracker:
                    res = func(*args, **kwargs)
                accessed = tracker.get_accessed_files()
                auto_file_deps = snapshot_file_deps(accessed) if accessed else None

                self._attach_lineage(res, cache_key)
                execution_time = time.perf_counter() - call_start
                self._store_in_cache(
                    cache_key, func_name, res, metadata, ttl,
                    current_state_hash, args_hash, execution_time,
                    auto_file_deps=auto_file_deps,
                )
                self._log_decorator_call(func_name, cache_hit=False, execution_time=execution_time, args_hash=args_hash, cache_key=cache_key)
                return res

            if self.use_locking:
                return self._compute_with_lock(cache_key, func_name, ttl, args_hash, call_start, _compute_and_store)
            return _compute_and_store()

        return wrapper

    def _make_async_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl: int | None,
    ) -> Callable:
        """Build and return the async caching wrapper for *func*.

        Mirrors :meth:`_make_wrapper` but the wrapper is ``async def``
        and the underlying ``func()`` invocation is awaited inside a
        ``FileAccessTracker`` block. Shared helpers (``_resolve_cache_key``,
        ``_try_get_cached``, ``_store_in_cache``, ``_log_decorator_call``)
        are sync and reused as-is.
        """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_start = time.perf_counter()

            if func_name not in self._analyzed:
                self._analyze_dependencies(func)
                self._analyzed.add(func_name)

            key_result = self._resolve_cache_key(
                func, func_name, dynamic_depends_on, args, kwargs, call_start
            )
            if key_result[0] is _CACHE_MISS:
                # _resolve_cache_key called `func(*args, **kwargs)` on the
                # unhashable/error path. For an async function that returns
                # a coroutine — we must await it before handing back.
                result_or_coro = key_result[1]
                if inspect.iscoroutine(result_or_coro):
                    return await result_or_coro
                return result_or_coro
            cache_key, current_state_hash, args_hash = key_result

            metadata, cached_data = self.backend.get(cache_key)
            hit = self._try_get_cached(
                cache_key, metadata, cached_data, call_start,
                args_hash, func_name, ttl,
            )
            if hit is not _CACHE_MISS:
                return hit

            if self.use_locking:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    "",
                    f"@cash.cache on {func_name}: use_locking=True is not "
                    f"yet supported for async functions. Proceeding without lock — "
                    f"two concurrent awaits of the same key will compute redundantly.",
                    stacklevel=5,
                )

            from cash.notebook.file_tracker import FileAccessTracker
            from cash.notebook.cache_freshness import snapshot_file_deps
            tracker = FileAccessTracker(getattr(func, '__globals__', None))
            with tracker:
                res = await func(*args, **kwargs)
            accessed = tracker.get_accessed_files()
            auto_file_deps = snapshot_file_deps(accessed) if accessed else None

            self._attach_lineage(res, cache_key)
            execution_time = time.perf_counter() - call_start
            self._store_in_cache(
                cache_key, func_name, res, metadata, ttl,
                current_state_hash, args_hash, execution_time,
                auto_file_deps=auto_file_deps,
            )
            self._log_decorator_call(
                func_name, cache_hit=False,
                execution_time=execution_time,
                args_hash=args_hash, cache_key=cache_key,
            )
            return res

        return wrapper

    def _delete_backend_entries(self, func_name: str) -> None:
        """Delete all backend cache entries whose key starts with *func_name*."""
        try:
            if hasattr(self.backend, 'keys'):
                for key in list(self.backend.keys()):
                    if key.startswith(f"{func_name}:"):
                        self.backend.delete(key)
        except (OSError, RuntimeError, KeyError):
            logger.debug("Failed to clear cache entries for %s", func_name)

    def _wrap_with_stats(self, func: Callable, func_name: str, wrapper: Callable) -> Callable:
        """Wrap *wrapper* with hit/miss stat tracking and attach introspection API.

        Dispatches on whether *func* is a coroutine function so the stats
        drain (reading ``_decorator_call_log`` for the just-finished call)
        happens AFTER the await for async, and synchronously otherwise.
        """
        _stats = {'hits': 0, 'misses': 0, 'total_time_saved': 0.0}

        def _drain_stats() -> None:
            with self._decorator_call_log_lock:
                for call in reversed(self._decorator_call_log):
                    if call['func_name'] == func_name:
                        if call['cache_hit']:
                            _stats['hits'] += 1
                            _stats['total_time_saved'] += call['execution_time']
                        else:
                            _stats['misses'] += 1
                        break

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await wrapper(*args, **kwargs)
                _drain_stats()
                return result
        else:
            @functools.wraps(func)
            def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = wrapper(*args, **kwargs)
                _drain_stats()
                return result

        def cache_info() -> dict[str, Any]:
            """Return cache statistics for this function.

            Returns:
                Dict with keys: hits, misses, hit_rate, total_time_saved.
            """
            total = _stats['hits'] + _stats['misses']
            hit_rate = _stats['hits'] / total if total > 0 else 0.0
            return {
                'hits': _stats['hits'],
                'misses': _stats['misses'],
                'hit_rate': hit_rate,
                'total_time_saved': _stats['total_time_saved'],
            }

        def cache_clear() -> None:
            """Clear all cached results for this function.

            Removes all cache entries whose key starts with the function name.
            Resets hit/miss statistics.
            """
            _stats['hits'] = 0
            _stats['misses'] = 0
            _stats['total_time_saved'] = 0.0
            self._delete_backend_entries(func_name)

        stats_wrapper.cache_info = cache_info
        stats_wrapper.cache_clear = cache_clear
        stats_wrapper.__wrapped__ = func
        return stats_wrapper

    def _register_static_dependencies(
        self, func_name: str, depends_on: list[Callable[..., Any] | DataSource] | None
    ) -> None:
        if not depends_on:
            return
        for dep in depends_on:
            if isinstance(dep, DataSource):
                dep_id = dep.get_id()
                self.data_sources[dep_id] = dep
                self.graph.add_dependency(func_name, dep_id)
            elif callable(dep):
                self.graph.add_dependency(func_name, self._get_func_key(dep))

    def _resolve_dynamic_dependencies(
        self,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
        if not dynamic_depends_on:
            return ""

        dynamic_state_parts = []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]

        for resolver in resolvers:
            try:
                # Resolver receives the same args as the function
                ds_result = resolver(*args, **kwargs)

                # Normalize to list
                dss = ds_result if isinstance(ds_result, list) else [ds_result]

                for ds in dss:
                    if isinstance(ds, DataSource):
                        if hasattr(ds, '_get_mtime'):
                            dynamic_state_parts.append(str(ds._get_mtime()))
                        else:
                            dynamic_state_parts.append(str(ds.has_changed()))
            except (OSError, TypeError, ValueError, AttributeError, RuntimeError) as e:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    "",
                    f"@cash.cache on {func_name}: dynamic_depends_on resolver raised "
                    f"{type(e).__name__} ({e}). Call will not include this dependency "
                    f"in the cache key — results may be stale if the underlying data changes.",
                    stacklevel=6,
                )

        if dynamic_state_parts:
            # Sort to ensure deterministic order if multiple sources
            return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode('utf-8')).hexdigest()
        return ""

    def _serialize_args(self, func_name: str, args: tuple, kwargs: dict) -> str | None:
        try:
            def get_arg_hash(arg):
                if hasattr(arg, '_cash_lineage_hash'):
                    return arg._cash_lineage_hash
                for type_, hasher_fn in self._type_hashers.items():
                    if isinstance(arg, type_):
                        return hasher_fn(arg)
                # 3. Try built-in type hashers for common non-picklable types
                builtin_hash = self._try_builtin_type_hash(arg)
                if builtin_hash is not None:
                    return builtin_hash
                return arg

            hashed_args = tuple(get_arg_hash(a) for a in args)
            hashed_kwargs = {k: get_arg_hash(v) for k, v in kwargs.items()}

            args_bytes = pickle.dumps((hashed_args, hashed_kwargs))
            return hashlib.sha256(args_bytes).hexdigest()
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            # Pickle failure here is surfaced via CashCacheIneffectiveWarning in
            # _resolve_cache_key (which sees the None return). Keep this log at
            # debug level so it's available when explicitly enabled but doesn't
            # double-warn.
            logger.debug("Could not serialize arguments for %s: %s", func_name, e)
            return None

    @staticmethod
    def _try_hash_pandas(value: Any, type_name: str) -> str | None:
        """Hash a pandas DataFrame or Series."""
        try:
            import pandas as pd
            return hashlib.sha256(
                pd.util.hash_pandas_object(value).values.tobytes()
            ).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash pandas %s via hash_pandas_object", type_name)
            return None

    @staticmethod
    def _try_hash_numpy(value: Any) -> str | None:
        """Hash a numpy ndarray."""
        try:
            if value.nbytes < 10 * 1024 * 1024:  # < 10 MB: full hash
                return hashlib.sha256(value.tobytes()).hexdigest()
            # Large array: hash shape + dtype + first/last 1 KB
            header = f"{value.shape}:{value.dtype}".encode()
            flat = value.ravel()
            sample = flat[:128].tobytes() + flat[-128:].tobytes()
            return hashlib.sha256(header + sample).hexdigest()
        except (TypeError, ValueError, AttributeError, MemoryError):
            logger.debug("Failed to hash numpy ndarray")
            return None

    @staticmethod
    def _try_hash_polars(value: Any, type_name: str) -> str | None:
        """Hash a polars DataFrame, Series, or LazyFrame."""
        try:
            import polars as pl
            if isinstance(value, pl.DataFrame):
                return hashlib.sha256(
                    value.hash_rows().to_list().__repr__().encode('utf-8')
                ).hexdigest()
            if isinstance(value, pl.Series):
                return hashlib.sha256(
                    value.hash().to_list().__repr__().encode('utf-8')
                ).hexdigest()
            if isinstance(value, pl.LazyFrame):
                return hashlib.sha256(str(value.explain()).encode('utf-8')).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash polars %s", type_name)
        return None

    @staticmethod
    def _try_hash_pyarrow(value: Any, type_name: str) -> str | None:
        """Hash a PyArrow Table or RecordBatch."""
        try:
            import pyarrow as pa
            if isinstance(value, (pa.Table, pa.RecordBatch)):
                schema_str = str(value.schema)
                header = f"{schema_str}:{value.num_rows}".encode()
                if value.nbytes < 10 * 1024 * 1024:
                    return hashlib.sha256(
                        header + value.to_pandas().values.tobytes()
                    ).hexdigest()
                return hashlib.sha256(header).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError, MemoryError):
            logger.debug("Failed to hash PyArrow %s", type_name)
        return None

    @staticmethod
    def _try_hash_modin(value: Any, type_name: str) -> str | None:
        """Hash a modin DataFrame or Series via pandas conversion."""
        try:
            pandas_val = value._to_pandas() if hasattr(value, '_to_pandas') else value
            import pandas as pd
            return hashlib.sha256(
                pd.util.hash_pandas_object(pandas_val).values.tobytes()
            ).hexdigest()
        except (ImportError, TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash modin %s", type_name)
            return None

    @staticmethod
    def _try_hash_dask(value: Any) -> str | None:
        """Hash a dask object via its task graph key."""
        try:
            graph_key = str(value.__dask_keys__())
            return hashlib.sha256(graph_key.encode('utf-8')).hexdigest()
        except (TypeError, ValueError, AttributeError):
            logger.debug("Failed to hash dask object via __dask_keys__")
            return None

    @staticmethod
    def _try_builtin_type_hash(value: Any) -> str | None:
        """Attempt to hash common types that may not pickle well.

        Returns a hex-digest string for recognised types (pandas, numpy,
        polars, PyArrow, modin, dask) or ``None`` when the value is not
        a supported type and should fall through to the default pickle
        path.

        The ``_try_`` prefix signals that ``None`` is a normal,
        expected return value — not an error.
        """
        type_name = type(value).__name__
        module = type(value).__module__ or ''

        if module.startswith('pandas') and type_name in ('DataFrame', 'Series'):
            return Cash._try_hash_pandas(value, type_name)

        if type_name == 'ndarray' and module.startswith('numpy'):
            return Cash._try_hash_numpy(value)

        if module.startswith('polars'):
            return Cash._try_hash_polars(value, type_name)

        if module.startswith('pyarrow'):
            return Cash._try_hash_pyarrow(value, type_name)

        if module.startswith('modin'):
            return Cash._try_hash_modin(value, type_name)

        if module.startswith('dask'):
            return Cash._try_hash_dask(value)

        # Generators / iterators — cannot hash
        if hasattr(value, '__next__') and hasattr(value, '__iter__'):
            return None

        return None

    def _compute_with_lock(
        self,
        cache_key: str,
        func_name: str,
        ttl: int | None,
        args_hash: str,
        call_start: float,
        compute_and_store: Callable[[], Any],
    ) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error."""
        try:
            with self.backend.lock(cache_key):
                locked_metadata, locked_data = self.backend.get(cache_key)
                # Use metadata presence (not data presence) as the
                # existence test — see _try_get_cached for rationale.
                if locked_metadata is not None:
                    try:
                        self._validate_ttl(locked_metadata, ttl)
                        self._log_decorator_call(func_name, cache_hit=True, execution_time=time.perf_counter() - call_start, args_hash=args_hash, cache_key=cache_key)
                        return locked_data
                    except (TypeError, KeyError, CacheExpiredError):
                        logger.debug("Cache validation failed under lock for %s", func_name)
                return compute_and_store()
        except (OSError, RuntimeError, TimeoutError) as e:
            logger.warning("Locking failed for %s, proceeding without lock: %s", func_name, e)
        return compute_and_store()

    def _compute_cache_key(self, func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
        return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"

    def _validate_ttl(self, metadata: CacheMetadata | None, ttl: int | None) -> None:
        if ttl is not None and metadata:
            timestamp = metadata.get('timestamp', 0)
            if time.time() - timestamp > ttl:
                raise CacheExpiredError("Cache expired")

    def _attach_lineage(self, result: Any, cache_key: str) -> None:
        """Attach lineage hash to result if it supports attribute setting.

        Works with pandas DataFrame/Series, polars DataFrame/Series, PyArrow
        Table, modin DataFrame, and any object that allows setting attributes.
        """
        try:
            type_name = type(result).__name__
            module = type(result).__module__ or ''

            # pandas DataFrame / Series (has attrs dict)
            if module.startswith('pandas') and type_name in ('DataFrame', 'Series'):
                result._cash_lineage_hash = cache_key
                return

            # polars DataFrame / Series
            if module.startswith('polars') and type_name in ('DataFrame', 'Series'):
                try:
                    result._cash_lineage_hash = cache_key
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to polars %s", type_name)
                return

            # modin DataFrame / Series
            if module.startswith('modin') and type_name in ('DataFrame', 'Series'):
                try:
                    result._cash_lineage_hash = cache_key
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to modin %s", type_name)
                return

            # PyArrow Table
            if module.startswith('pyarrow') and type_name in ('Table', 'RecordBatch'):
                try:
                    result._cash_lineage_hash = cache_key
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to PyArrow %s", type_name)
                return

            # Generic: try setting on DataFrame-like objects with attrs
            if type_name == 'DataFrame' and hasattr(result, 'attrs'):
                result._cash_lineage_hash = cache_key
                return

        except (AttributeError, TypeError):
            logger.debug("Failed to attach lineage hash to %s result", type(result).__name__)

    def _log_decorator_call(
        self,
        func_name: str,
        cache_hit: bool,
        execution_time: float,
        args_hash: str,
        cache_key: str,
    ) -> None:
        """Record a decorator call event for notebook integration.

        Thread-safe: uses a lock to protect concurrent appends.
        The notebook ``StatementProcessor`` drains this log after each
        statement execution to include decorator call metrics in the badge.
        """
        entry = {
            'func_name': func_name,
            'cache_hit': cache_hit,
            'execution_time': execution_time,
            'args_hash': args_hash,
            'cache_key': cache_key,
            'timestamp': time.time(),
        }
        with self._decorator_call_log_lock:
            self._decorator_call_log.append(entry)

    def _warn_once(
        self,
        category: type[Warning],
        func_name: str,
        arg_type_name: str,
        message: str,
        *,
        stacklevel: int = 5,
    ) -> None:
        """Emit ``warnings.warn(message, category)`` at most once per
        ``(category, func_name, arg_type_name)`` for this Cash instance.

        ``arg_type_name`` is the empty string for warnings that do not
        attach to a specific arg type (e.g. store-failed). The seen-set
        key still distinguishes by func_name.

        ``stacklevel`` controls which frame is blamed in the warning's
        filename/lineno. The default of ``5`` is correct for warnings
        emitted from ``_resolve_cache_key`` via the standard call chain
        ``user → stats_wrapper → wrapper → _resolve_cache_key → _warn_once``.
        Callers reached through a deeper or shallower chain pass their
        own value:

        * ``_store_in_cache`` via ``_compute_and_store`` → ``stacklevel=6``
        * ``_resolve_dynamic_dependencies`` (called from ``_resolve_cache_key``) → ``stacklevel=6``
        * ``cache(func)`` decoration-time checks → ``stacklevel=3``
        """
        key = (category, func_name, arg_type_name)
        with self._decorator_call_log_lock:
            if key in self._warning_keys_seen:
                return
            self._warning_keys_seen.add(key)
        warnings.warn(message, category=category, stacklevel=stacklevel)

    def drain_decorator_calls(self) -> list[dict[str, Any]]:
        """Return and clear all recorded decorator call events.

        Thread-safe: atomically copies and clears the log.
        Called by the notebook statement processor after executing a statement
        to collect decorator-level cache metrics for badge display.

        Returns:
            List of call event dicts, each with keys:
            ``func_name``, ``cache_hit``, ``execution_time``, ``args_hash``,
            ``cache_key``, ``timestamp``.
        """
        with self._decorator_call_log_lock:
            calls = list(self._decorator_call_log)
            self._decorator_call_log.clear()
        return calls

    def register_hasher(self, type_: type, hasher_fn: Callable[[Any], str]) -> None:
        """Register a custom hasher for a specific type.

        When ``_serialize_args`` encounters an argument of ``type_``, it will
        call ``hasher_fn(value)`` to produce a hash string instead of relying
        on ``pickle.dumps``.

        Args:
            type_: The Python type to register a hasher for.
            hasher_fn: A callable that takes a value of ``type_`` and returns
                a deterministic hash string.

        Example::

            import pandas as pd
            from cash import Cash

            c = Cash()
            c.register_hasher(
                pd.DataFrame,
                lambda df: hashlib.sha256(
                    pd.util.hash_pandas_object(df).values.tobytes()
                ).hexdigest()
            )
        """
        self._type_hashers[type_] = hasher_fn

    def _store_in_cache(
        self,
        cache_key: str,
        func_name: str,
        result: Any,
        metadata: dict[str, Any] | None,
        ttl: int | None,
        state_hash: str,
        args_hash: str,
        execution_time: float = 0.0,
        auto_file_deps: dict[str, dict[str, float]] | None = None,
    ) -> None:
        try:
            serializer = get_serializer(result)

            metadata = {
                'key': cache_key,
                'func_name': func_name,
                'timestamp': time.time(),
                # The decorator's measured wall-clock cost. ``TieredBackend``
                # reads this to decide whether the value is expensive enough
                # to promote past RAM (otherwise the smart-persistence
                # policy gates everything at the 0.1s floor, and script
                # runs that recompute the same cheap value forever).
                'execution_time': execution_time,
                'serializer_cls': type(serializer),
                'ttl': ttl,
                'args_hash': args_hash,
                'state_hash': state_hash,
            }
            if auto_file_deps:
                # Each entry: path → {'mtime': float, 'size': int}
                # Validated on subsequent get() via _auto_file_deps_fresh.
                metadata['auto_file_deps'] = auto_file_deps

            self.backend.set(cache_key, result, metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed to store "
                f"result ({type(e).__name__}: {e}). Compute succeeded; next call will recompute.",
                stacklevel=6,
            )

    def cleanup(self, max_age: int | None = None) -> int:
        """Remove expired items from the cache.

        Args:
            max_age: If provided, remove items older than *max_age* seconds,
                regardless of their stored TTL.

        Returns:
            Number of entries removed.
        """
        now = time.time()

        def is_expired(metadata):
            try:
                timestamp = metadata.get('timestamp', 0)
                age = now - timestamp

                if max_age is not None and age > max_age:
                    return True

                stored_ttl = metadata.get('ttl')
                return bool(stored_ttl is not None and age > stored_ttl)
            except (TypeError, KeyError, ValueError):
                return True

        return self.backend.cleanup_expired(is_expired)

    def explorer(self) -> CacheExplorer:
        """Return a :class:`CacheExplorer` instance for interactive cache browsing."""
        from .ui.explorer import CacheExplorer
        return CacheExplorer(self)

    def show_stats(self) -> None:
        """Display the interactive analytics dashboard.

        Requires IPython/Jupyter and ipywidgets. In script environments,
        prints a text summary instead.
        """
        try:
            from .ui.dashboard import show_analytics_dashboard
            show_analytics_dashboard()
        except (ImportError, RuntimeError):
            # Fallback for script environments
            n_funcs = len(self.functions)
            n_sources = len(self.data_sources)
            backend_name = type(self.backend).__name__
            print("Cash Statistics:")
            print(f"  Backend: {backend_name}")
            print(f"  Cached functions: {n_funcs}")
            print(f"  Data sources: {n_sources}")

    def register_magic(self) -> None:
        """Register IPython magic commands (``%cash_on``, ``%%cash``, etc.)."""
        try:
            from IPython import get_ipython

            from .notebook.magics import CashMagics

            ip = get_ipython()
            if ip:
                magics = CashMagics(ip, self)
                ip.register_magics(magics)
            else:
                logger.debug("No active IPython session found. Magic commands not registered.")
        except ImportError:
            logger.debug("IPython not available. Magic commands not registered.")

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Static analysis to find calls to other cached functions."""
        func_name = self._get_func_key(func)
        called_names = CodeAnalyzer.find_called_functions(func, self.functions)
        for called in called_names:
            if called != func_name:
                self.graph.add_dependency(func_name, called)

    def register_file_handler(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]) -> None:
        """
        Register a custom file dependency handler.

        Args:
            module_name: Name of the module (e.g., 'my_lib').
            func_name: Name of the function to track (e.g., 'read_data'). Supports wildcards like 'read_*'.
            handler_factory: A function that takes (original_function, track_callback)
                             and returns a wrapper function.

                             Example handler_factory:
                             def my_handler(original_func, track_callback):
                                 def wrapper(path, *args, **kwargs):
                                     track_callback(path)
                                     return original_func(path, *args, **kwargs)
                                 return wrapper
        """
        from .notebook.file_tracker import FileDependencyRegistry
        registry = FileDependencyRegistry()
        registry.register(module_name, func_name, handler_factory)

    def _get_dependency_state_hash(self, node: str, visited: set[str]) -> str:
        """Compute a dependency state hash recursively for the given graph node."""
        if node in visited:
            return ""
        visited.add(node)

        hashes = []

        # 1. Node's own state
        if node in self.functions:
            hashes.append(self.source_hashes.get(node, ""))
        elif node in self.data_sources:
            ds = self.data_sources[node]
            if hasattr(ds, '_get_mtime'):
                hashes.append(str(ds._get_mtime()))
            else:
                hashes.append(str(ds.has_changed()))

        # 2. Dependencies' state
        dependencies = self.graph.get_dependencies(node)
        for dep in sorted(dependencies):
            dep_hash = self._get_dependency_state_hash(dep, visited)
            hashes.append(dep_hash)

        return hashlib.sha256(":".join(hashes).encode('utf-8')).hexdigest()

    def shutdown(self) -> None:
        """Cleanup resources (e.g. wait for async writes)."""
        if hasattr(self, 'backend') and self.backend:
            self.backend.shutdown()
