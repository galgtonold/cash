"""Main Cash class - decorator-based caching with automatic dependency tracking.

Provides the `Cash` entry point for ``@cash.cache`` function-level
caching and the `Cash.notebook` bridge for Jupyter integration.
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from .backends import CacheBackend, CacheMetadata, CascadingBackend
from .backends.factory import build_backend_from_config

if TYPE_CHECKING:
    from .ui.explorer import CacheExplorer
from .backends.serialization import get_serializer
from .config import CashConfig, get_config
from .data_source import DataSource
from .dependency_state import DependencyStateHasher, SysModulesHelperResolver
from .exceptions import (
    CacheExpiredError,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from .graph import DependencyGraph
from .notebook.analysis import CodeAnalyzer
from .purity_analyzer import PurityReport, get_analyzer

# Configure Logging
logger = logging.getLogger(__name__)

try:
    from IPython import get_ipython
except ImportError:  # IPython not installed
    def get_ipython():  # type: ignore[misc]
        return None

# Sentinel object used by wrapper helpers to signal a cache miss without
# conflicting with any legitimate cached value (including None).
_CACHE_MISS = object()

P = ParamSpec("P")
T = TypeVar("T")

__all__ = ["Cash", "CacheExplanation"]


# Reason codes returned by `Cash._explain_call` / ``f.explain(...)``.
# Kept as module-level constants so external code can match against them
# without string-typo risk: ``if e.reason == EXPLAIN_HIT: ...``.
EXPLAIN_HIT = "hit"
EXPLAIN_KEY_UNCOMPUTABLE = "key_uncomputable"
EXPLAIN_NO_ENTRY = "no_entry"
EXPLAIN_TTL_EXPIRED = "ttl_expired"
EXPLAIN_FILE_CHANGED = "file_changed"


@dataclass(frozen=True)
class CacheExplanation:
    """Why a specific call would hit or miss the cache *right now*.

    Returned by ``f.explain(*args, **kwargs)`` on any ``@cash.cache``-wrapped
    function. Inspecting an explanation does NOT mutate stats, call the
    underlying function, or write to the backend - it only reads what the
    cache already knows.

    Attributes:
        would_hit: True if the next call with these args would return a
            cached value (without recomputing).
        reason: Short stable string identifying the outcome. One of:
            ``"hit"``, ``"key_uncomputable"``, ``"no_entry"``,
            ``"ttl_expired"``, ``"file_changed"``.
        func_name: Module-qualified name of the cached function.
        cache_key: The cache key computed for these args, or ``None``
            when key generation failed (``reason == "key_uncomputable"``).
        details: Reason-specific extras. Common keys:

            * ``hit``: ``cached_at`` (unix ts), ``execution_time_saved`` (s),
              ``cache_age_seconds``.
            * ``key_uncomputable``: ``arg_type`` (qualname or ``"<unknown>"``),
              ``error`` (exception type+message), ``hint``.
            * ``no_entry``: ``hint``.
            * ``ttl_expired``: ``ttl_seconds``, ``age_seconds``, ``cached_at``.
            * ``file_changed``: ``changed_files`` (dict of path -> reason).
    """

    would_hit: bool
    reason: str
    func_name: str
    cache_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        verdict = "HIT" if self.would_hit else "MISS"
        lines = [f"[{verdict}] {self.func_name} - {self.reason}"]
        if self.cache_key:
            lines.append(f"  cache_key: {self.cache_key}")
        for k, v in self.details.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class _ListCachedIterator:
    """Lazy iterator wrapper over a fully-materialized cached list.

    Returned on cache hit when the function's output fit in a single
    chunk: the yielded values are stored as one Python list and this
    iterator streams them on each call. See `_ChunkedCachedIterator`
    for the multi-chunk variant used for large iterators.
    """

    __slots__ = ("_iter",)

    def __init__(self, items):
        self._iter = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        """Stop iteration. Subsequent ``next()`` raises ``StopIteration``."""
        # No generator to throw GeneratorExit into - the iterator is
        # just a replay over a materialized list. Emptying the iterator
        # is the right semantic equivalent.
        self._iter = iter(())

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported. The cached "
            "iterator replays a previously materialized list. If you "
            "need send() semantics, the function cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported. The cached "
            "iterator replays a previously materialized list. If you "
            "need throw() semantics, the function cannot be cached."
        )


class _ChunkedCachedIterator:
    """Lazy iterator that reads cached chunks from the backend on demand.

    Used by `Cash.cache` for iterator-returning functions whose
    output spans multiple backend keys. Each chunk is fetched only
    when the user iterates into it; chunks the user never reaches are
    never read. The retrieval is RAM-bounded by chunk size.

    The class satisfies the iterator protocol (``iter(x) is x``,
    ``__next__``, ``close``); generator-specific methods (``send``,
    ``throw``) raise ``AttributeError`` - the cached iterator is a
    replay of stored values, not a coroutine.

    Args:
        cash: The owning `Cash` instance (used for backend access).
        cache_key: The canonical key under which the manifest is stored.
            Chunk keys are derived as ``f"{cache_key}:chunk_{i}"``.
        n_chunks: Total chunk count, taken from the manifest at construction.

    The iterator is robust to chunk loss: if ``backend.get`` returns
    ``(None, None)`` for any chunk (e.g. RAM-only eviction), iteration
    terminates cleanly via ``StopIteration``. The next call to the
    decorated function will see a cache miss and recompute.
    """

    __slots__ = ("_cash", "_cache_key", "_n_chunks",
                 "_chunk_index", "_current_chunk_iter", "_closed")

    def __init__(self, cash: Any, cache_key: str, n_chunks: int):
        self._cash = cash
        self._cache_key = cache_key
        self._n_chunks = n_chunks
        self._chunk_index = 0
        self._current_chunk_iter = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        while True:
            if self._current_chunk_iter is not None:
                try:
                    return next(self._current_chunk_iter)
                except StopIteration:
                    self._current_chunk_iter = None
                    # Fall through to load the next chunk.
            if self._chunk_index >= self._n_chunks:
                raise StopIteration
            chunk_key = f"{self._cache_key}:chunk_{self._chunk_index}"
            _, chunk = self._cash.backend.get(chunk_key)
            self._chunk_index += 1
            if chunk is None:
                # Chunk lost (eviction, partial cleanup). Safe termination -
                # the next call to the decorated function will see a cache
                # miss on the manifest and recompute from scratch.
                raise StopIteration
            self._current_chunk_iter = iter(chunk)

    def close(self):
        """Stop iteration. Subsequent ``next()`` raises ``StopIteration``."""
        self._closed = True
        self._current_chunk_iter = None

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported on chunked "
            "iterators. The cached iterator replays values from the "
            "backend. If you need send() semantics, the function "
            "cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported on chunked "
            "iterators. If you need throw() semantics, the function "
            "cannot be cached."
        )


def _make_opaque_issue(func_name: str, opaque_list: str) -> Any:
    """Build a synthetic `PurityIssue` for opaque callees
    encountered in ``strict`` mode. Defined at module scope so the
    ``_surface_purity`` import stays local."""
    from .purity_analyzer import ISSUE_IMPURE_CALL, PurityIssue
    return PurityIssue(
        kind=ISSUE_IMPURE_CALL,
        description=f"opaque callees (strict): {opaque_list}",
        where=func_name,
        line=0,
    )


def _format_issues_summary(func_name: str, issues: list[Any]) -> str:
    """Pretty-print a list of `PurityIssue` records, grouped
    by their ``where`` field. Used by both the warning body and the
    strict-mode exception body so users get the same diagnostic.
    """
    by_where: dict[str, list[Any]] = {}
    for i in issues:
        by_where.setdefault(i.where, []).append(i)
    lines = []
    for where in sorted(by_where):
        lines.append(f"  in {where}:")
        for issue in by_where[where]:
            line_part = f"line {issue.line}: " if issue.line else ""
            lines.append(f"    {line_part}[{issue.kind}] {issue.description}")
    return "\n".join(lines)


def _is_one_shot_iterator(value: Any) -> bool:
    """Return True if *value* is its own iterator (a one-shot consumable).

    Matches Python generators, ``map``/``filter``/``zip`` results, and
    custom iterators that return ``self`` from ``__iter__``. Returns
    False for collections (``list``/``dict``/``set``/``tuple``/``str``/
    ``range``) which are iterable but return fresh iterators on
    ``iter()`` - those are safely cacheable as-is.
    """
    try:
        return iter(value) is value
    except TypeError:
        return False

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

    Example:

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
        register_magic: bool | None = None,
        debug: bool | None = None,
        use_locking: bool = False,
        config_path: str | None = None,
        verbose: bool = False,
        **config_overrides: Any,
    ) -> None:
        # Map the explicit convenience kwargs (cache_dir, compress, debug)
        # into the overrides dict so the config layer treats them with the
        # same priority as any other constructor-supplied override (highest).
        for key, val in (("cache_dir", cache_dir), ("compress", compress), ("debug", debug)):
            if val is not None:
                config_overrides.setdefault(key, val)
        self.config = get_config(config_path=config_path, overrides=config_overrides or None)

        debug = self.config.debug

        # Store params for lazy backend construction. If an explicit
        # backend (or list of backends) was provided, that wins - those
        # are concrete objects, not config - and we skip the factory.
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
        self._analyzed = set() # Track which functions we've *surfaced* purity for
        # Track which functions have had their graph edges + purity report
        # populated (separate from _analyzed: a dependency can be populated to
        # complete a parent's state hash long before it is called directly and
        # surfaced). Keeps the cache key stable from the first call (finding #7).
        self._populated: set[str] = set()
        self._func_key_cache: dict[int, str] = {}  # id(func) -> module-qualified key
        self.debug = debug  # Debug mode flag
        self.use_locking = use_locking
        self.verbose = verbose

        # Decorator call log for notebook integration.
        # Each entry is a dict with: func_name, cache_hit (bool), execution_time,
        # args_hash, cache_key, timestamp.  The notebook statement processor
        # drains this list after executing each statement so it can include
        # decorator metrics in the badge.
        self._decorator_call_log: list[dict[str, Any]] = []
        self._decorator_call_log_lock = threading.Lock()
        # Custom type hasher registry: maps type -> (callable(value) -> str, source hash).
        # The source hash is embedded in the args_hash composition so that
        # changing a hasher's body invalidates dependent cache entries.
        self._type_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}

        # Dedup keys for _warn_once: (category, func_name, arg_type_name).
        # Guarded by _decorator_call_log_lock (already exists for thread safety).
        self._warning_keys_seen: set[tuple[type[Warning], str, str]] = set()

        # Per-function rolling log of recent warning emissions, surfaced
        # via ``f.cache_info()['warnings']``. Each entry is
        # ``{'category': str, 'message': str, 'timestamp': float}``.
        # Capped at ``_func_warnings_max`` entries per function so a
        # noisy function can't grow this dict unboundedly. Guarded by
        # ``_decorator_call_log_lock``.
        self._func_warnings: dict[str, list[dict[str, Any]]] = {}
        self._func_warnings_max = 20

        # Registry of wrapped (stats) functions for clear_all() support.
        # Maps func_name -> stats_wrapper (the object returned to the user).
        self._wrapped_funcs: dict[str, Any] = {}

        # Per-function purity mode set at decoration time:
        # ``"warn"`` (default) | ``"silent"`` (assume_safe=True) |
        # ``"strict"`` (raises). Read on first call.
        self._purity_modes: dict[str, str] = {}
        # Per-function purity report cache. Populated on first call.
        # Helper source hashes from this report fold into the cache
        # key state hash so cross-process helper edits invalidate.
        self._purity_reports: dict[str, PurityReport] = {}

        # Deep seam over the registries above: folds source/dependency/
        # helper state into the cache key's ``state_hash`` segment. Borrows
        # the registry dicts by reference so later registrations are seen.
        self._state_hasher = DependencyStateHasher(
            functions=self.functions,
            data_sources=self.data_sources,
            source_hashes=self.source_hashes,
            purity_reports=self._purity_reports,
            graph=self.graph,
            helper_resolver=SysModulesHelperResolver(self._hash_callable_source),
        )

        atexit.register(self.shutdown)

        # register_magic=None (default) auto-detects: only register when an
        # active IPython session exists.  True forces registration; False skips.
        if register_magic is True or (register_magic is None and get_ipython() is not None):
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

    @staticmethod
    def _hash_callable_source(fn: Callable) -> str:
        """Return a stable hex digest representing *fn*'s body.

        Used by `register_hasher` to embed the hasher's source
        identity in the cache key, so that changing a hasher's body
        invalidates dependent cache entries even when the hasher's
        output coincidentally matches the old one.

        Resolution order:

        1. ``inspect.getsource(fn)`` - primary. Works for module-level
           functions and lambdas defined in a discoverable source file.
        2. ``fn.__code__.co_code`` - fallback. Works for functions defined
           in a REPL or via ``exec()``. Bytecode is stable within a Python
           version; a Python upgrade conservatively invalidates the cache.
        3. ``fn.__call__.__code__.co_code`` - fallback for callable
           instances (objects with ``__call__``). Two instances of the
           same callable class share the same source hash.
        4. ``type(fn).__qualname__`` - last resort. Doesn't differentiate
           instances of the same class; the user gets stability within
           a process but coarse cross-process behavior.
        """
        try:
            src = inspect.getsource(fn)
            return hashlib.sha256(src.encode("utf-8")).hexdigest()
        except (OSError, TypeError):
            pass
        code = getattr(fn, "__code__", None)
        if code is None:
            # Callable instance - try its __call__.__code__
            call_method = getattr(fn, "__call__", None)
            code = getattr(call_method, "__code__", None)
        if code is not None:
            return hashlib.sha256(code.co_code).hexdigest()
        return hashlib.sha256(type(fn).__qualname__.encode("utf-8")).hexdigest()


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
        cache_if: Callable[[Any], bool] | None = ...,
        chunk_max_items: int = ...,
        chunk_max_bytes: int = ...,
        strict: bool = ...,
        assume_safe: bool = ...,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def cache(
        self,
        func: Callable[P, T] | None = None,
        *,
        depends_on: list[Callable[..., Any] | DataSource] | None = None,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        file_depends_on: str | list[str] | None = None,
        ttl: int | None = None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
        strict: bool = False,
        assume_safe: bool = False,
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
            cache_if: Optional predicate ``callable(result) -> bool``. When
                provided, called with the function's return value after
                computation. If it returns a falsy value, the result is
                NOT stored in the cache (but is still returned to the
                caller). Useful for skipping the caching of negative
                results, e.g. ``cache_if=lambda r: r is not None``.
                The predicate is only invoked when the function returns
                normally; an exception from the function is re-raised
                without consulting the predicate. If the predicate itself
                raises, a one-shot `CashCacheIneffectiveWarning`
                fires and the result is treated as not-cacheable (the
                user's call still returns the result).
                If the cache key cannot be built at all (unhashable
                argument with no registered hasher, key-generation
                error), the predicate is not consulted - nothing is
                cached on that fallback path either.
            chunk_max_items: When the decorated function returns an
                iterator, close the current chunk after this many
                items. Default ``1_000_000``. A chunk closes when
                either ``chunk_max_items`` or ``chunk_max_bytes`` is
                reached (whichever comes first). For iterators below
                both thresholds, the entire result lands in a single
                chunk and storage is indistinguishable from a list.
            chunk_max_bytes: When the decorated function returns an
                iterator, close the current chunk after this many
                bytes (estimated via ``estimate_object_size``).
                Default ``1_000_000_000`` (1 GB). See
                ``chunk_max_items`` for the joint behavior.
            strict: When ``True``, raise `CashImpureFunctionError`
                on first call if the analyzer finds any purity issues
                (known-impure calls, scope mutations, explicit
                dynamism, or discarded calls to non-known-pure
                callees). Also promotes the analyzer's optimistic
                opaque-leaf treatment: opaque callees become issues.
                Use in CI to fail builds that introduce caching of
                side-effecting code. Mutually exclusive with
                ``assume_safe``.
            assume_safe: When ``True``, suppress the
                `CashImpurityWarning` even when the analyzer
                finds issues. Use when you've audited the function
                and know caching is correct (the side effect is
                idempotent, the dynamism is bounded, etc.). The
                analyzer still runs because it captures helper
                source hashes for cache invalidation. Mutually
                exclusive with ``strict``.

        Returns:
            The decorated function with caching behavior.

        See Also:
            [Caching class methods](../tutorials/feature-guides/caching-class-methods.md)
            for the recipe for caching methods on stateful objects
            (databases, file handles, connections) via
            [`register_hasher`][cash.Cash.register_hasher].
        """
        if strict and assume_safe:
            raise ValueError(
                "@cash.cache: strict=True and assume_safe=True are mutually "
                "exclusive. strict raises on purity issues; assume_safe silences "
                "them. Pick one."
            )

        if func is None:
            return lambda f: self.cache(
                f, depends_on=depends_on, dynamic_depends_on=dynamic_depends_on,
                file_depends_on=file_depends_on, ttl=ttl, cache_if=cache_if,
                chunk_max_items=chunk_max_items, chunk_max_bytes=chunk_max_bytes,
                strict=strict, assume_safe=assume_safe,
            )

        func_name = self._register_func(func, depends_on, file_depends_on)
        self._purity_modes[func_name] = ("strict" if strict else "silent" if assume_safe else "warn")

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
            wrapper = self._make_async_wrapper(
                func, func_name, dynamic_depends_on, ttl, cache_if,
                chunk_max_items, chunk_max_bytes,
            )
        else:
            wrapper = self._make_wrapper(
                func, func_name, dynamic_depends_on, ttl, cache_if,
                chunk_max_items, chunk_max_bytes,
            )
        return self._wrap_with_stats(
            func, func_name, wrapper,
            dynamic_depends_on=dynamic_depends_on, ttl=ttl,
        )

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
            self._populated.discard(func_name)
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
          - (cache_key_str, state_hash_str, args_hash_str)  - normal
          - (_CACHE_MISS, result, 'unhashable')             - unhashable args
          - (_CACHE_MISS, result, 'error')                  - key generation error
        """
        try:
            current_state_hash = self._state_hasher.compute(func_name)
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

    def _explain_call(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl: int | None,
        args: tuple,
        kwargs: dict,
    ) -> CacheExplanation:
        """Return why a call with these args would hit or miss the cache.

        Pure introspection - does NOT call ``func``, does NOT touch
        `Cash` stats, does NOT emit warnings, and does NOT
        mutate the backend. Mirrors the logic of `_resolve_cache_key`
        + `_try_get_cached` so that the answer reflects what would
        actually happen on the next real call.

        See `CacheExplanation` for the return shape.
        """
        # Populate the dependency closure first so the state hash matches what
        # a real call computes (otherwise explain() reports a stale pre-analysis
        # key and a false `no_entry` - finding #7). This only fills internal
        # analysis caches; it does not warn, run the function, or touch the
        # backend.
        self._ensure_closure_analyzed(func)

        # Build cache key (silently - explain() does not warn).
        try:
            current_state_hash = self._state_hasher.compute(func_name)
        except (TypeError, ValueError, RuntimeError) as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'error': f'{type(e).__name__}: {e}',
                    'hint': 'Dependency state hash computation raised.',
                },
            )

        try:
            dynamic_state_hash = self._resolve_dynamic_dependencies_silent(
                dynamic_depends_on, args, kwargs,
            )
        except (TypeError, ValueError, RuntimeError, AttributeError, OSError) as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'error': f'{type(e).__name__}: {e}',
                    'hint': 'dynamic_depends_on resolver raised.',
                },
            )

        try:
            args_hash = self._serialize_args(func_name, args, kwargs)
        except (TypeError, ValueError, pickle.PicklingError, AttributeError) as e:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'arg_type': arg_type_name,
                    'error': f'{type(e).__name__}: {e}',
                    'hint': (
                        f'Register a hasher via cash.register_hasher({arg_type_name}, ...)'
                        if arg_type_name != '<unknown>'
                        else 'Could not identify the offending argument.'
                    ),
                },
            )

        if args_hash is None:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    'arg_type': arg_type_name,
                    'hint': (
                        f'Register a hasher via cash.register_hasher({arg_type_name}, ...)'
                        if arg_type_name != '<unknown>'
                        else 'Could not identify the offending argument; likely a nested unpicklable value.'
                    ),
                },
            )

        cache_key = self._compute_cache_key(
            func_name, current_state_hash, dynamic_state_hash, args_hash,
        )

        raw_metadata, _data = self.backend.get(cache_key)
        if raw_metadata is None:
            details = {
                'hint': (
                    'No matching cache entry. First call with these arguments, '
                    'or the cache was cleared.'
                ),
            }
            # A tracked dynamic dependency that changed produces a NEW cache key,
            # so the miss surfaces as no_entry rather than file_changed. Make the
            # explanation say so and list what's tracked (finding #8).
            dyn_ids = self._describe_dynamic_dependencies(dynamic_depends_on, args, kwargs)
            if dyn_ids:
                details['dynamic_dependencies'] = dyn_ids
                details['hint'] = (
                    'No matching cache entry. Either the first call with these '
                    'arguments, or a tracked dynamic dependency changed - a '
                    'dynamic_depends_on change yields a new cache key, so it '
                    'shows up here as no_entry, not file_changed. Tracked '
                    'dynamic dependencies: ' + ', '.join(dyn_ids) + '.'
                )
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_NO_ENTRY,
                func_name=func_name,
                cache_key=cache_key,
                details=details,
            )

        metadata = CacheMetadata.from_dict(raw_metadata)

        # TTL check - match _validate_ttl semantics: only if ttl was set
        # at decoration time.
        if ttl is not None:
            timestamp = metadata.timestamp or 0
            age = time.time() - timestamp
            if age > ttl:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_TTL_EXPIRED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={
                        'ttl_seconds': ttl,
                        'age_seconds': age,
                        'cached_at': timestamp,
                    },
                )

        # Auto-tracked file deps freshness.
        snap = metadata.auto_file_deps or {}
        if snap:
            import os
            stale: dict[str, str] = {}
            for path, recorded in snap.items():
                try:
                    st = os.stat(path)
                except OSError:
                    stale[path] = 'file missing'
                    continue
                if st.st_mtime != recorded.get('mtime'):
                    stale[path] = 'mtime changed'
                elif st.st_size != recorded.get('size'):
                    stale[path] = 'size changed'
            if stale:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_FILE_CHANGED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={'changed_files': stale},
                )

        timestamp = metadata.timestamp or 0
        return CacheExplanation(
            would_hit=True,
            reason=EXPLAIN_HIT,
            func_name=func_name,
            cache_key=cache_key,
            details={
                'cached_at': timestamp,
                'cache_age_seconds': time.time() - timestamp if timestamp else None,
                'execution_time_saved': metadata.execution_time or 0.0,
            },
        )

    def _resolve_dynamic_dependencies_silent(
        self,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> str:
        """Variant of `_resolve_dynamic_dependencies` that re-raises
        instead of warning - used by `_explain_call` so introspection
        never emits warnings as a side effect."""
        if not dynamic_depends_on:
            return ""
        dynamic_state_parts = []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
        for resolver in resolvers:
            ds_result = resolver(*args, **kwargs)
            dss = ds_result if isinstance(ds_result, list) else [ds_result]
            for ds in dss:
                if isinstance(ds, DataSource):
                    if hasattr(ds, '_get_mtime'):
                        dynamic_state_parts.append(str(ds._get_mtime()))
                    else:
                        dynamic_state_parts.append(str(ds.has_changed()))
        if dynamic_state_parts:
            return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode('utf-8')).hexdigest()
        return ""

    def _describe_dynamic_dependencies(
        self,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> list[str]:
        """Best-effort list of the ``DataSource`` ids a function's
        ``dynamic_depends_on`` resolves to for these args - so ``explain()`` can
        report *what* is being tracked. Returns ``[]`` when there are none or
        resolution fails (introspection must never raise)."""
        if not dynamic_depends_on:
            return []
        resolvers = dynamic_depends_on if isinstance(dynamic_depends_on, list) else [dynamic_depends_on]
        ids: list[str] = []
        for resolver in resolvers:
            try:
                ds_result = resolver(*args, **kwargs)
            except Exception:  # noqa: BLE001 - explain() is best-effort
                continue
            dss = ds_result if isinstance(ds_result, list) else [ds_result]
            for ds in dss:
                if isinstance(ds, DataSource):
                    try:
                        ids.append(ds.get_id())
                    except Exception:  # noqa: BLE001
                        ids.append(repr(ds))
        return ids

    @staticmethod
    def _first_unhashable_arg_type(args: tuple, kwargs: dict) -> str:
        """Return the qualname of the first non-built-in arg type, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. Skips strings,
        ints, floats, bools, None, lists, dicts, tuples, sets - they're
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
        metadata: CacheMetadata | None,
        cached_data: Any,
        call_start: float,
        args_hash: str,
        func_name: str,
        ttl: int | None,
    ) -> Any:
        """Return cached_data if valid, else _CACHE_MISS sentinel.

        Key-presence is determined by ``metadata is not None`` - the
        backend contract is that absent keys return ``(None, None)``,
        so a non-None metadata view with a ``None`` data value still
        counts as a hit (a function that legitimately returned ``None``).

        Auto-tracked file dependencies stored in
        ``metadata.auto_file_deps`` are re-stat'd here; any file with
        a different mtime/size than recorded forces a miss so the
        function re-reads the changed file.
        """
        if metadata is not None:
            try:
                self._validate_ttl(metadata, ttl)
                if not self._auto_file_deps_fresh(metadata):
                    return _CACHE_MISS
                # If this hit happens *inside* another cached function's
                # computation, replay the files this entry depends on into the
                # enclosing tracker, so the outer function records them too.
                # Without this, a dependency that was already cached before the
                # consumer's first run hides its file deps behind a cache hit
                # and the consumer never invalidates when that file changes.
                self._propagate_file_deps_to_active_tracker(metadata)
                self._log_decorator_call(
                    func_name, cache_hit=True,
                    execution_time=time.perf_counter() - call_start,
                    args_hash=args_hash, cache_key=cache_key,
                    time_saved=metadata.execution_time or 0.0,
                )
                return cached_data
            except CacheExpiredError:
                pass
            except (TypeError, KeyError) as e:
                self._warn_metadata_invalid(func_name, e)
        return _CACHE_MISS

    @staticmethod
    def _propagate_file_deps_to_active_tracker(metadata: CacheMetadata) -> None:
        """Register this entry's recorded file deps with the enclosing
        ``FileAccessTracker`` (if any), so a cached function that calls this
        one on a *hit* still inherits its file dependencies. Best-effort: any
        failure (no tracker active, import issue) is silently ignored."""
        snap = getattr(metadata, "auto_file_deps", None)
        if not snap:
            return
        try:
            from cash.notebook.file_tracker import _active_tracker
            tracker = _active_tracker.get()
        except Exception:  # noqa: BLE001 - tracking is best-effort
            return
        if tracker is not None:
            for path in snap:
                tracker._add_tracked(path)

    @staticmethod
    def _auto_file_deps_fresh(metadata: CacheMetadata) -> bool:
        """Return True if every file recorded in ``metadata.auto_file_deps``
        still has the same (mtime, size) on disk.

        Auto-tracked deps are captured during the first compute via
        `cash.notebook.file_tracker.FileAccessTracker` and stored
        as ``{path: {'mtime': float, 'size': int}}``. If a recorded path
        is gone or its (mtime, size) changed, we invalidate the cache
        so the next compute re-reads the file. A path that disappears
        is also a change.
        """
        snap = metadata.auto_file_deps or {}
        if not snap:
            return True  # nothing to check
        import os
        for path, recorded in snap.items():
            try:
                st = os.stat(path)
            except OSError:
                return False  # file vanished - invalidate
            if st.st_mtime != recorded.get('mtime') or st.st_size != recorded.get('size'):
                return False
        return True

    def _wrap_iterator_hit(
        self,
        cache_key: str,
        metadata: CacheMetadata | None,
        hit: Any,
    ) -> Any:
        """Wrap a cache-hit value in the right iterator class.

        Iterators (including the single-chunk case) are stored under
        an ``iterator_storage='chunked'`` manifest plus N chunk
        entries; on hit they're returned as a fresh
        ``_ChunkedCachedIterator``. Non-iterator return types live as
        a single blob and are returned as *hit* directly.

        Used by all three cache-hit paths in this module:
        `_make_wrapper` (sync unlocked), `_compute_with_lock`
        (sync locked re-read), and `_make_async_wrapper`. Keeping
        the dispatch in one place ensures the three paths can't drift.
        """
        if metadata and metadata.iterator_storage == 'chunked':
            n_chunks = metadata.n_chunks or 0
            return _ChunkedCachedIterator(self, cache_key, n_chunks)
        return hit

    def _make_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl: int | None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
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

            raw_metadata, cached_data = self.backend.get(cache_key)
            metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
            hit = self._try_get_cached(cache_key, metadata, cached_data, call_start, args_hash, func_name, ttl)
            if hit is not _CACHE_MISS:
                return self._wrap_iterator_hit(cache_key, metadata, hit)

            def _compute_and_store() -> Any:
                # Wrap the function call in FileAccessTracker so any
                # auto-tracked file reads (pandas/numpy/joblib/open/...)
                # are recorded as implicit cache dependencies - a later
                # mtime/size change forces a recompute.
                from cash.notebook.file_tracker import FileAccessTracker
                from cash.notebook.file_dep_snapshot import snapshot_file_deps
                tracker = FileAccessTracker(getattr(func, '__globals__', None), propagate_to_parent=True)
                with tracker:
                    res = func(*args, **kwargs)
                accessed = tracker.get_accessed_files()
                auto_file_deps = snapshot_file_deps(accessed) if accessed else None

                # Detect one-shot iterators and route them through the
                # chunked-storage path.
                if _is_one_shot_iterator(res):
                    manifest, single_chunk_buffer = self._write_chunks(
                        res, cache_key, chunk_max_items, chunk_max_bytes,
                        func_name, cache_if, ttl=ttl,
                    )
                    execution_time = time.perf_counter() - call_start

                    if single_chunk_buffer is not None:
                        # Single-chunk path. Apply cache_if before writing.
                        should_cache = True
                        if cache_if is not None:
                            try:
                                should_cache = bool(cache_if(single_chunk_buffer))
                            except Exception as e:
                                self._warn_cache_if_raised(func_name, e)
                                should_cache = False

                        if should_cache and single_chunk_buffer:
                            # Write the one chunk now that the predicate approved.
                            self._write_one_chunk(cache_key, 0, single_chunk_buffer, ttl=ttl)
                            self._store_chunked_manifest(
                                cache_key, func_name, manifest, metadata,
                                ttl, current_state_hash, args_hash,
                                execution_time, auto_file_deps,
                            )
                        elif should_cache and not single_chunk_buffer:
                            # Empty iterator - still write a (zero-chunk) manifest
                            # so a hit returns an empty iterator instead of recomputing.
                            self._store_chunked_manifest(
                                cache_key, func_name, manifest, metadata,
                                ttl, current_state_hash, args_hash,
                                execution_time, auto_file_deps,
                            )
                        # else: cache_if rejected - return result un-cached.

                        self._log_decorator_call(
                            func_name, cache_hit=False,
                            execution_time=execution_time,
                            args_hash=args_hash, cache_key=cache_key,
                        )
                        return _ListCachedIterator(single_chunk_buffer)

                    # Multi-chunk path: chunks already written. Write manifest.
                    self._store_chunked_manifest(
                        cache_key, func_name, manifest, metadata,
                        ttl, current_state_hash, args_hash,
                        execution_time, auto_file_deps,
                    )
                    self._log_decorator_call(
                        func_name, cache_hit=False,
                        execution_time=execution_time,
                        args_hash=args_hash, cache_key=cache_key,
                    )
                    return _ChunkedCachedIterator(self, cache_key, manifest["n_chunks"])

                # Non-iterator return: existing single-blob path.
                self._attach_lineage(res, cache_key)
                execution_time = time.perf_counter() - call_start

                should_cache = True
                if cache_if is not None:
                    try:
                        should_cache = bool(cache_if(res))
                    except Exception as e:
                        self._warn_cache_if_raised(func_name, e)
                        should_cache = False

                if should_cache:
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
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
    ) -> Callable:
        """Build and return the async caching wrapper for *func*.

        Mirrors `_make_wrapper` but the wrapper is ``async def``
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
                # a coroutine - we must await it before handing back.
                result_or_coro = key_result[1]
                if inspect.iscoroutine(result_or_coro):
                    return await result_or_coro
                return result_or_coro
            cache_key, current_state_hash, args_hash = key_result

            raw_metadata, cached_data = self.backend.get(cache_key)
            metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
            hit = self._try_get_cached(
                cache_key, metadata, cached_data, call_start,
                args_hash, func_name, ttl,
            )
            if hit is not _CACHE_MISS:
                return self._wrap_iterator_hit(cache_key, metadata, hit)

            if self.use_locking:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    "",
                    f"@cash.cache on {func_name}: use_locking=True is not "
                    f"yet supported for async functions. Proceeding without lock - "
                    f"two concurrent awaits of the same key will compute redundantly.",
                    stacklevel=5,
                )

            from cash.notebook.file_tracker import FileAccessTracker
            from cash.notebook.file_dep_snapshot import snapshot_file_deps
            tracker = FileAccessTracker(getattr(func, '__globals__', None), propagate_to_parent=True)
            with tracker:
                res = await func(*args, **kwargs)
            accessed = tracker.get_accessed_files()
            auto_file_deps = snapshot_file_deps(accessed) if accessed else None

            # Iterator returns: chunked storage path.
            if _is_one_shot_iterator(res):
                # warn_stacklevel=5 for async: one fewer frame than sync
                # because there's no _compute_and_store closure layer
                # (user -> stats_wrapper -> wrapper -> _write_chunks -> _warn_once).
                manifest, single_chunk_buffer = self._write_chunks(
                    res, cache_key, chunk_max_items, chunk_max_bytes,
                    func_name, cache_if, ttl=ttl, warn_stacklevel=5,
                )
                execution_time = time.perf_counter() - call_start

                if single_chunk_buffer is not None:
                    should_cache = True
                    if cache_if is not None:
                        try:
                            should_cache = bool(cache_if(single_chunk_buffer))
                        except Exception as e:
                            self._warn_cache_if_raised(func_name, e, stacklevel=6)
                            should_cache = False

                    if should_cache and single_chunk_buffer:
                        self._write_one_chunk(cache_key, 0, single_chunk_buffer, ttl=ttl)
                        self._store_chunked_manifest(
                            cache_key, func_name, manifest, metadata,
                            ttl, current_state_hash, args_hash,
                            execution_time, auto_file_deps,
                        )
                    elif should_cache and not single_chunk_buffer:
                        self._store_chunked_manifest(
                            cache_key, func_name, manifest, metadata,
                            ttl, current_state_hash, args_hash,
                            execution_time, auto_file_deps,
                        )

                    self._log_decorator_call(
                        func_name, cache_hit=False,
                        execution_time=execution_time,
                        args_hash=args_hash, cache_key=cache_key,
                    )
                    return _ListCachedIterator(single_chunk_buffer)

                # Multi-chunk path.
                self._store_chunked_manifest(
                    cache_key, func_name, manifest, metadata,
                    ttl, current_state_hash, args_hash,
                    execution_time, auto_file_deps,
                )
                self._log_decorator_call(
                    func_name, cache_hit=False,
                    execution_time=execution_time,
                    args_hash=args_hash, cache_key=cache_key,
                )
                return _ChunkedCachedIterator(self, cache_key, manifest["n_chunks"])

            # Non-iterator return: single-blob path (unchanged).
            self._attach_lineage(res, cache_key)
            execution_time = time.perf_counter() - call_start

            should_cache = True
            if cache_if is not None:
                try:
                    should_cache = bool(cache_if(res))
                except Exception as e:
                    self._warn_cache_if_raised(func_name, e, stacklevel=6)
                    should_cache = False

            if should_cache:
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
            prefix = f"{func_name}:"
            for entry in self.backend.list_entries():
                key = CacheMetadata.from_dict(entry).key or ''
                if key.startswith(prefix):
                    self.backend.delete(key)
        except (OSError, RuntimeError, KeyError):
            logger.debug("Failed to clear cache entries for %s", func_name)

    def _wrap_with_stats(
        self,
        func: Callable,
        func_name: str,
        wrapper: Callable,
        *,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        ttl: int | None = None,
    ) -> Callable:
        """Wrap *wrapper* with hit/miss stat tracking and attach introspection API.

        Dispatches on whether *func* is a coroutine function so the stats
        drain (reading ``_decorator_call_log`` for the just-finished call)
        happens AFTER the await for async, and synchronously otherwise.

        Attaches the introspection API:

        * ``cache_info()`` - hit/miss stats plus a rolling list of recent
          warnings emitted for this function.
        * ``cache_clear()`` - drop backend entries, reset stats, drop the
          warning log + dedup marks so re-warnings can fire.
        * ``explain(*args, **kwargs)`` - return a `CacheExplanation`
          for that specific call (sync, even on async wrappers).
        """
        _stats = {'hits': 0, 'misses': 0, 'total_time_saved': 0.0}

        def _drain_stats() -> None:
            with self._decorator_call_log_lock:
                for call in reversed(self._decorator_call_log):
                    if call['func_name'] == func_name:
                        if call['cache_hit']:
                            _stats['hits'] += 1
                            _stats['total_time_saved'] += call.get('time_saved', 0.0)
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
            """Return cache statistics + recent warnings for this function.

            Returns:
                Dict with keys:

                * ``hits`` (int) - cache hits since this wrapper was created.
                * ``misses`` (int) - cache misses (including key-uncomputable
                  and store-failed paths).
                * ``hit_rate`` (float) - ``hits / (hits + misses)``, or 0.0.
                * ``total_time_saved`` (float) - sum of execution times that
                  were avoided by serving from cache.
                * ``warnings`` (list[dict]) - rolling log of recent warning
                  emissions for this function. Each entry has ``category``,
                  ``message``, ``timestamp``. Capped at the last
                  ``_func_warnings_max`` (default 20) so it can't grow
                  unboundedly. Useful for spotting silent misbehavior
                  (cache_if predicate raised, lock failure, etc.) when
                  ``warnings.simplefilter`` swallowed the stderr emission.
            """
            total = _stats['hits'] + _stats['misses']
            hit_rate = _stats['hits'] / total if total > 0 else 0.0
            with self._decorator_call_log_lock:
                warnings_log = list(self._func_warnings.get(func_name, []))
            return {
                'hits': _stats['hits'],
                'misses': _stats['misses'],
                'hit_rate': hit_rate,
                'total_time_saved': _stats['total_time_saved'],
                'warnings': warnings_log,
            }

        def cache_clear() -> None:
            """Clear all cached results for this function.

            Removes all cache entries whose key starts with the function name.
            Resets hit/miss statistics, drops the per-function warnings log,
            and forgets ``_warn_once`` dedup marks for this function so the
            next misbehavior re-warns instead of being silently swallowed.
            """
            _stats['hits'] = 0
            _stats['misses'] = 0
            _stats['total_time_saved'] = 0.0
            self._delete_backend_entries(func_name)
            with self._decorator_call_log_lock:
                self._func_warnings.pop(func_name, None)
                # Drop dedup marks for this function so future misbehavior
                # re-warns the user instead of staying silent.
                self._warning_keys_seen = {
                    k for k in self._warning_keys_seen if k[1] != func_name
                }

        def explain(*args: Any, **kwargs: Any) -> CacheExplanation:
            """Return why the next call with these args would hit or miss.

            See `CacheExplanation` for the return shape. Inspection
            only - does not call the underlying function, mutate stats,
            or write to the backend. Safe to call from sync code even
            on async-wrapped functions.
            """
            return self._explain_call(
                func, func_name, dynamic_depends_on, ttl, args, kwargs,
            )

        stats_wrapper.cache_info = cache_info
        stats_wrapper.cache_clear = cache_clear
        stats_wrapper.explain = explain
        stats_wrapper.__wrapped__ = func
        # Marker so the purity analyzer treats a call to this wrapper as a
        # dependency-graph edge rather than recursing into cash's own wrapper
        # machinery (finding #9). functools.wraps copies __module__, which would
        # otherwise make the wrapper look like same-package user code.
        stats_wrapper._cash_cached = True
        self._wrapped_funcs[func_name] = stats_wrapper
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
                    f"in the cache key - results may be stale if the underlying data changes.",
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
                for type_, (hasher_fn, src_hash) in self._type_hashers.items():
                    if isinstance(arg, type_):
                        # Embed the hasher source hash so that changing the
                        # hasher's body invalidates dependent cache entries
                        # even when the hasher's output coincidentally matches.
                        return f"{src_hash}:{hasher_fn(arg)}"
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
        expected return value - not an error.
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

        # Generators / iterators - cannot hash
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
                raw_locked_metadata, locked_data = self.backend.get(cache_key)
                locked_metadata = (
                    CacheMetadata.from_dict(raw_locked_metadata)
                    if raw_locked_metadata is not None else None
                )
                # Use metadata presence (not data presence) as the
                # existence test - see _try_get_cached for rationale.
                if locked_metadata is not None:
                    try:
                        self._validate_ttl(locked_metadata, ttl)
                        self._log_decorator_call(
                            func_name, cache_hit=True,
                            execution_time=time.perf_counter() - call_start,
                            args_hash=args_hash, cache_key=cache_key,
                            time_saved=locked_metadata.execution_time or 0.0,
                        )
                        return self._wrap_iterator_hit(cache_key, locked_metadata, locked_data)
                    except CacheExpiredError:
                        pass
                    except (TypeError, KeyError) as e:
                        self._warn_metadata_invalid(func_name, e, stacklevel=6)
                return compute_and_store()
        except (OSError, RuntimeError, TimeoutError) as e:
            self._warn_lock_failed(func_name, e, stacklevel=4)
        return compute_and_store()

    def _compute_cache_key(self, func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
        return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"

    def _validate_ttl(self, metadata: CacheMetadata | None, ttl: int | None) -> None:
        if ttl is not None and metadata:
            timestamp = metadata.timestamp or 0
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
        time_saved: float = 0.0,
    ) -> None:
        """Record a decorator call event for notebook integration.

        Thread-safe: uses a lock to protect concurrent appends.
        The notebook ``StatementProcessor`` drains this log after each
        statement execution to include decorator call metrics in the badge.

        ``execution_time`` is the wall-time of *this* operation - a lookup on a
        hit, the compute on a miss. ``time_saved`` is the compute a hit
        *avoided* (the originally-measured execution time stored with the
        cached entry), and 0.0 on a miss. They are distinct: a hit's
        ``execution_time`` is microseconds, but its ``time_saved`` is the full
        compute it stood in for. ``cache_info()['total_time_saved']`` sums the
        latter - summing ``execution_time`` (the old behaviour) under-reported
        savings by orders of magnitude.
        """
        entry = {
            'func_name': func_name,
            'cache_hit': cache_hit,
            'execution_time': execution_time,
            'time_saved': time_saved,
            'args_hash': args_hash,
            'cache_key': cache_key,
            'timestamp': time.time(),
        }
        with self._decorator_call_log_lock:
            self._decorator_call_log.append(entry)
        if self.verbose:
            status = 'hit' if cache_hit else 'miss'
            logger.info('cash %s: %s (%.3fs)', status, func_name, execution_time)

    def _warn_cache_if_raised(
        self, func_name: str, error: BaseException, *, stacklevel: int = 7,
    ) -> None:
        """Surface a raised ``cache_if`` predicate as a user-visible warning.

        Previously this was a ``logger.debug`` - invisible to anyone not
        explicitly configuring logging. Promoted to a one-shot
        `CashCacheIneffectiveWarning` so a buggy predicate is
        diagnosed instead of silently disabling the cache.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "cache_if",
            f"@cash.cache on {func_name}: cache_if predicate raised "
            f"{type(error).__name__} ({error}). The result is returned "
            f"un-cached and will be recomputed on the next call. Fix "
            f"the predicate or remove cache_if= to restore caching.",
            stacklevel=stacklevel,
        )

    def _warn_metadata_invalid(
        self, func_name: str, error: BaseException, *, stacklevel: int = 5,
    ) -> None:
        """Surface a malformed cache-metadata read as a user-visible warning.

        Happens when a backend returns a metadata dict missing the
        expected keys (e.g. a partially-written entry from an older
        cash version, or a corrupted file on disk). The call falls
        through to recompute - but the user should know.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "metadata_invalid",
            f"@cash.cache on {func_name}: a stored cache entry's metadata "
            f"could not be validated ({type(error).__name__}: {error}). "
            f"Treating as a miss and recomputing. This usually means the "
            f"entry was written by an older cash version or partially "
            f"corrupted; clearing the cache for this function may help.",
            stacklevel=stacklevel,
        )

    def _warn_lock_failed(
        self, func_name: str, error: BaseException, *, stacklevel: int = 5,
    ) -> None:
        """Surface a backend-locking failure as a user-visible warning.

        Previously this was ``logger.warning`` - visible to anyone who
        wired up logging.warning, but invisible to anyone running with
        default config. Promoted to a CashCacheIneffectiveWarning so
        the user notices the implicit race risk.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "lock_failed",
            f"@cash.cache on {func_name}: backend lock acquisition failed "
            f"({type(error).__name__}: {error}). Proceeding without lock - "
            f"concurrent calls with the same args may compute redundantly. "
            f"Investigate the backend (disk full, permissions, broken lockfile?).",
            stacklevel=stacklevel,
        )

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
        ``user -> stats_wrapper -> wrapper -> _resolve_cache_key -> _warn_once``.
        Callers reached through a deeper or shallower chain pass their
        own value:

        * ``_store_in_cache`` via ``_compute_and_store`` -> ``stacklevel=6``
        * ``_resolve_dynamic_dependencies`` (called from ``_resolve_cache_key``) -> ``stacklevel=6``
        * ``cache(func)`` decoration-time checks -> ``stacklevel=3``
        """
        key = (category, func_name, arg_type_name)
        with self._decorator_call_log_lock:
            if key in self._warning_keys_seen:
                return
            self._warning_keys_seen.add(key)
            # Also record in per-function rolling log so the warning is
            # discoverable after the fact via ``f.cache_info()['warnings']``
            # - even if the user missed the stderr emission.
            entry = {
                'category': category.__name__,
                'message': message,
                'timestamp': time.time(),
            }
            log = self._func_warnings.setdefault(func_name, [])
            log.append(entry)
            if len(log) > self._func_warnings_max:
                del log[: len(log) - self._func_warnings_max]
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

        Example:

            import pandas as pd
            from cash import Cash

            c = Cash()
            c.register_hasher(
                pd.DataFrame,
                lambda df: hashlib.sha256(
                    pd.util.hash_pandas_object(df).values.tobytes()
                ).hexdigest()
            )

            Note: when ``hasher_fn`` is a callable object (an instance
            with ``__call__``), the source hash is derived from the
            class's ``__call__.__code__`` - so two instances of the
            same callable class share a source hash, even if they hold
            different per-instance state. If your hasher's behavior
            depends on instance state, prefer a function or lambda
            that closes over the state explicitly.
        """
        src_hash = self._hash_callable_source(hasher_fn)
        self._type_hashers[type_] = (hasher_fn, src_hash)

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

            meta = CacheMetadata(
                key=cache_key,
                func_name=func_name,
                timestamp=time.time(),
                # The decorator's measured wall-clock cost. ``TieredBackend``
                # reads this to decide whether the value is expensive enough
                # to promote past RAM (otherwise the smart-persistence
                # policy gates everything at the 0.1s floor, and script
                # runs that recompute the same cheap value forever).
                execution_time=execution_time,
                serializer_cls=type(serializer),
                ttl=ttl,
                args_hash=args_hash,
                state_hash=state_hash,
                # Each entry: path -> {'mtime': float, 'size': int}.
                # Validated on subsequent get() via _auto_file_deps_fresh.
                auto_file_deps=auto_file_deps or None,
            )

            self.backend.set(cache_key, result, meta.to_dict(), serializer=serializer)
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

    def _write_chunks(
        self,
        iterator: Any,
        cache_key: str,
        chunk_max_items: int,
        chunk_max_bytes: int,
        func_name: str,
        cache_if: Callable[[Any], bool] | None,
        *,
        ttl: int | None = None,
        warn_stacklevel: int = 6,
    ) -> tuple[dict[str, Any], list[Any] | None]:
        """Stream *iterator* into chunks, writing each chunk to the backend.

        Returns ``(manifest, single_chunk_buffer)``. The manifest is a
        dict describing the layout (``n_chunks``, ``total_items``). The
        ``single_chunk_buffer`` is non-None only when the iterator
        exhausted before the first threshold was reached AND no chunks
        have been written yet - i.e. the entire result fits in a single
        chunk and the caller may want to apply ``cache_if`` to it before
        committing.

        When the buffer crosses a threshold and we close the second
        chunk while ``cache_if is not None``, emit a one-shot
        `CashCacheIneffectiveWarning` documenting that the
        predicate is bypassed for multi-chunk results.

        ``warn_stacklevel`` controls which frame the bypass warning
        attributes to. Default ``6`` is correct for the sync caller
        (``user -> stats_wrapper -> wrapper -> _compute_and_store ->
        _write_chunks -> _warn_once``). Async callers pass ``5``
        (one fewer frame - no ``_compute_and_store`` closure).

        Chunks are written under keys ``f"{cache_key}:chunk_{i}"``.
        Chunk metadata is minimal - the manifest at *cache_key* is the
        authoritative entry; chunks themselves carry only enough
        metadata for the backend to deserialize them (timestamp, key,
        execution_time=0).
        """
        from cash.notebook.object_hashing import estimate_object_size

        buffer: list[Any] = []
        buffer_bytes = 0
        chunk_index = 0
        total_items = 0

        for item in iterator:
            buffer.append(item)
            buffer_bytes += estimate_object_size(item)
            total_items += 1

            if len(buffer) >= chunk_max_items or buffer_bytes >= chunk_max_bytes:
                # About to close the current chunk. If this is the
                # transition to the second chunk AND cache_if is set,
                # warn the user that the predicate is bypassed.
                if chunk_index == 1 and cache_if is not None:
                    self._warn_once(
                        CashCacheIneffectiveWarning,
                        func_name,
                        "",
                        f"@cash.cache on {func_name}: cache_if was bypassed "
                        f"because the result exceeded a single chunk "
                        f"(chunk_max_items={chunk_max_items}, "
                        f"chunk_max_bytes={chunk_max_bytes}). The result "
                        f"is cached without consulting the predicate. To "
                        f"keep cache_if gating in effect, lower the chunk "
                        f"thresholds or materialize the iterator manually.",
                        stacklevel=warn_stacklevel,
                    )
                self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl)
                buffer = []
                buffer_bytes = 0
                chunk_index += 1

        # Generator exhausted. If no chunks have been written yet AND
        # there's a tail buffer, the full result fits in a single chunk -
        # return the buffer so the caller can apply cache_if before
        # committing.
        if chunk_index == 0:
            # Single-chunk case - defer the write so cache_if can gate it.
            manifest = {
                "n_chunks": 1 if buffer else 0,
                "total_items": total_items,
            }
            return manifest, buffer

        # Multi-chunk path: flush the tail buffer if non-empty.
        if buffer:
            # The threshold-hit branch in the for-loop fires the cache_if-
            # bypass warning when chunk_1 fills via its own threshold. If
            # chunk_1 is only PARTIALLY filled (we exhausted the iterator
            # mid-chunk_1), the warning never fired there - but we are
            # still committing to a multi-chunk result with cache_if
            # bypassed. _warn_once dedups, so firing here is safe even if
            # the for-loop already fired.
            if chunk_index == 1 and cache_if is not None:
                self._warn_once(
                    CashCacheIneffectiveWarning,
                    func_name,
                    "",
                    f"@cash.cache on {func_name}: cache_if was bypassed "
                    f"because the result exceeded a single chunk "
                    f"(chunk_max_items={chunk_max_items}, "
                    f"chunk_max_bytes={chunk_max_bytes}). The result "
                    f"is cached without consulting the predicate. To "
                    f"keep cache_if gating in effect, lower the chunk "
                    f"thresholds or materialize the iterator manually.",
                    stacklevel=warn_stacklevel,
                )
            self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl)
            chunk_index += 1

        manifest = {
            "n_chunks": chunk_index,
            "total_items": total_items,
        }
        return manifest, None  # single_chunk_buffer is None for multi-chunk

    def _write_one_chunk(
        self,
        cache_key: str,
        chunk_index: int,
        chunk_buffer: list[Any],
        ttl: int | None = None,
    ) -> None:
        """Write a single chunk to the backend.

        The chunk's metadata is minimal - the authoritative manifest
        lives at the canonical cache_key. We need *some* metadata for
        the serializer to round-trip correctly; the timestamp and the
        key are enough. We also propagate the manifest's ``ttl`` so
        ``Cash.cleanup()`` (without a ``max_age`` argument) can reclaim
        expired chunks alongside the expired manifest.
        """
        chunk_key = f"{cache_key}:chunk_{chunk_index}"
        serializer = get_serializer(chunk_buffer)
        chunk_metadata = CacheMetadata(
            key=chunk_key,
            timestamp=time.time(),
            serializer_cls=type(serializer),
            execution_time=0.0,
            ttl=ttl,
        ).to_dict()
        try:
            self.backend.set(chunk_key, chunk_buffer, chunk_metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                f"{cache_key}:chunk_{chunk_index}",
                "",
                f"@cash.cache: backend {backend_name} failed to store "
                f"chunk {chunk_index} of {cache_key} ({type(e).__name__}: {e}). "
                f"The cache entry will be incomplete on retrieval.",
                stacklevel=4,
            )

    def _store_chunked_manifest(
        self,
        cache_key: str,
        func_name: str,
        manifest_data: dict[str, Any],
        existing_metadata: dict[str, Any] | None,
        ttl: int | None,
        state_hash: str,
        args_hash: str,
        execution_time: float,
        auto_file_deps: dict[str, dict[str, float]] | None,
    ) -> None:
        """Write the manifest entry for a chunked iterator at *cache_key*.

        The value stored at the key is the manifest dict (``n_chunks``,
        ``total_items``). The metadata flags this entry as chunked so
        the hit path knows to use ``_ChunkedCachedIterator``.
        """
        try:
            serializer = get_serializer(manifest_data)
            metadata = CacheMetadata(
                key=cache_key,
                func_name=func_name,
                timestamp=time.time(),
                execution_time=execution_time,
                serializer_cls=type(serializer),
                ttl=ttl,
                args_hash=args_hash,
                state_hash=state_hash,
                iterator_storage="chunked",
                n_chunks=manifest_data["n_chunks"],
                auto_file_deps=auto_file_deps or None,
            ).to_dict()
            self.backend.set(cache_key, manifest_data, metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed "
                f"to store chunked manifest ({type(e).__name__}: {e}). "
                f"Compute succeeded; next call will recompute.",
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

        def is_expired(raw_metadata):
            try:
                metadata = CacheMetadata.from_dict(raw_metadata)
                timestamp = metadata.timestamp or 0
                age = now - timestamp

                if max_age is not None and age > max_age:
                    return True

                stored_ttl = metadata.ttl
                return bool(stored_ttl is not None and age > stored_ttl)
            except (AttributeError, TypeError, ValueError):
                return True

        return self.backend.cleanup_expired(is_expired)

    def explorer(self) -> CacheExplorer:
        """Return a `CacheExplorer` instance for interactive cache browsing."""
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
        except ImportError:
            logger.debug("IPython not available. Magic commands not registered.")
            return

        ip = get_ipython()
        if ip is None:
            logger.debug("No active IPython session found. Magic commands not registered.")
            return

        # Internal import - must always succeed when IPython is present.
        # Kept outside the ImportError guard above so a broken import path
        # surfaces loudly instead of masquerading as "IPython not available".
        from .notebook.ipython.magics import CashMagics

        magics = CashMagics(ip, self)
        ip.register_magics(magics)

    def clear_all(self) -> None:
        """Clear cached results for every function registered with this instance.

        Equivalent to calling ``f.cache_clear()`` on every ``@cash.cache``-decorated
        function. Resets hit/miss statistics and removes all backend entries.
        """
        for wrapped in self._wrapped_funcs.values():
            wrapped.cache_clear()

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Populate analysis for *func* + its transitive cached-dependency
        closure, then surface *func*'s own purity issues.

        Populating the WHOLE closure (not just *func*) before the first cache
        key is computed is what keeps the key stable from the very first call.
        The state hash folds in each dependency's purity-report
        ``helper_source_hashes``; those used to be filled lazily on each
        dependency's own first call, so the key deepened only after the chain
        warmed - and a fresh process therefore missed the first call to every
        cached function even though a valid entry was on disk (finding #7).

        Surfacing stays per-function: each dependency warns/raises on its OWN
        first direct call, not here, so eager population doesn't change which
        warnings fire or when.
        """
        self._ensure_closure_analyzed(func)
        func_name = self._get_func_key(func)
        report = self._purity_reports.get(func_name) or PurityReport()
        mode = self._purity_modes.get(func_name, "warn")
        self._surface_purity(func_name, report, mode)

    def _ensure_closure_analyzed(self, func: Callable[..., Any]) -> None:
        """Populate graph edges + purity reports for *func* and every cached
        function transitively reachable from it, WITHOUT surfacing warnings.

        Idempotent per source version (guarded by ``self._populated``). Always
        traverses the dependency edges - even when the root is already
        populated - so a dependency invalidated by a source edit gets
        re-populated. The local ``seen`` set bounds cyclic graphs.
        """
        stack = [func]
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            fname = self._get_func_key(f)
            if fname in seen:
                continue
            seen.add(fname)
            if fname not in self._populated:
                self._populate_analysis(f, fname)
            for dep in self.graph.get_dependencies(fname):
                dep_func = self.functions.get(dep)
                if dep_func is not None:
                    stack.append(dep_func)

    def _populate_analysis(self, func: Callable[..., Any], func_name: str) -> None:
        """Record *func*'s cached-call graph edges and purity report (no
        surfacing). The analyzer caches by source hash globally, so this is
        cheap on repeated registrations.
        """
        self._populated.add(func_name)
        called_names = CodeAnalyzer.find_called_functions(func, self.functions)
        for called in called_names:
            if called != func_name:
                self.graph.add_dependency(func_name, called)
        try:
            report = get_analyzer().analyze(func)
        except (OSError, TypeError, SyntaxError, RecursionError) as e:
            # Analyzer must never break caching. On error, treat as clean -
            # the user's compute still runs.
            logger.debug("Purity analyzer failed for %s: %s", func_name, e)
            report = PurityReport()
        self._purity_reports[func_name] = report

    def _surface_purity(
        self, func_name: str, report: PurityReport, mode: str,
    ) -> None:
        """Turn a `PurityReport` into warnings or an exception.

        Called once per function on first call (after first
        ``_analyze_dependencies``).

        * ``warn`` (default): one-shot `CashImpurityWarning`
          summarising issues; also recorded in
          ``cache_info()['warnings']``.
        * ``silent`` (``assume_safe=True``): the user has audited
          this; suppress the warning. The report is still stored
          so helper source hashes invalidate correctly.
        * ``strict``: raise `CashImpureFunctionError`. Opaque
          callees count as issues in this mode (paranoid).
        """
        issues = list(report.issues)
        if mode == "strict" and report.opaque_callees:
            opaque_list = ", ".join(report.opaque_callees[:5])
            if len(report.opaque_callees) > 5:
                opaque_list += f", ... +{len(report.opaque_callees) - 5} more"
            issues.append(_make_opaque_issue(func_name, opaque_list))

        if not issues:
            return
        if mode == "silent":
            return

        summary = _format_issues_summary(func_name, issues)
        if mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: purity issues "
                f"detected. Either fix the function, mark callees with "
                f"@pure / @stateful, or relax to assume_safe=True.\n{summary}"
            )
        # mode == "warn"
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "purity",
            f"@cash.cache on {func_name}: the analyzer found likely "
            f"side effects or scope mutations. Cached results may not "
            f"reflect side-effect intent. Suppress with "
            f"@cash.cache(assume_safe=True) after auditing, or refactor.\n{summary}",
            stacklevel=6,
        )

    def register_file_handler(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]) -> None:
        """Register a custom file-dependency handler.

        Cash already intercepts the popular reader functions
        (``pd.read_csv``, ``np.load``, ``open``, ``json.load``,
        etc.) so any cached function that uses them gets automatic
        file-dep tracking. Use this method when your code reads
        files via a custom or vendored reader that Cash doesn't
        know about yet.

        The handler is a closure-style factory: Cash gives it the
        original function and a ``track_callback(path)``; it returns
        a replacement function that calls ``track_callback`` for
        each file path it touches and then forwards to the original.
        The wrapper is installed on the target module so all callers
        - yours and any library code - get tracking transparently.

        Args:
            module_name: The module that owns the reader function
                (e.g. ``"my_lib"``, ``"my_lib.io"``). Use a dotted
                path for nested modules.
            func_name: The reader function's name in that module
                (e.g. ``"read_data"``). Supports glob wildcards like
                ``"read_*"`` to track several readers at once.
            handler_factory: Factory that produces the wrapper. Must
                accept two arguments -
                ``(original_function, track_callback)`` - and return
                a callable with the same signature as the original.
                See *Example* below for the exact shape.

        Example:

            ```python
            import cash

            c = cash.Cash()

            # my_lib.read_data(path) reads a custom binary format.
            # Make any cached function calling it invalidate when
            # the file on disk changes.
            def custom_reader_handler(original_func, track_callback):
                def wrapper(path, *args, **kwargs):
                    track_callback(path)              # record the dep
                    return original_func(path, *args, **kwargs)
                return wrapper

            c.register_file_handler("my_lib", "read_data", custom_reader_handler)

            @c.cache
            def load_features():
                import my_lib
                return my_lib.read_data("/data/features.bin")
                # ^ when /data/features.bin changes, cache invalidates
            ```

            For multiple reader names in one go:

            ```python
            c.register_file_handler("my_lib", "read_*", custom_reader_handler)
            # Catches read_data, read_metadata, read_index, ...
            ```

        Notes:
            * The wrapper replaces the attribute on the live module
              object - so existing imports
              (``from my_lib import read_data``) still see the
              original unwrapped version. Track callers that go
              through the module namespace
              (``my_lib.read_data(...)``).
            * Inside the wrapper, call ``track_callback(path)`` with
              the **absolute or resolvable** path you want recorded.
              Relative paths are resolved against ``os.getcwd()`` at
              tracking time.
            * Tracking is on the file's ``(mtime, size)``; downstream
              cache-key computation is automatic.
        """
        from .notebook.file_tracker import FileDependencyRegistry
        registry = FileDependencyRegistry()
        registry.register(module_name, func_name, handler_factory)

    def shutdown(self) -> None:
        """Cleanup resources (e.g. wait for async writes).

        Guards on the *private* ``_backend`` rather than the ``backend``
        property: this runs from an ``atexit`` handler, and touching the
        lazy property during interpreter teardown would *build* a backend
        (spawning a ThreadPoolExecutor that calls
        ``threading._register_atexit``), raising "can't register atexit
        after shutdown". If the backend was never materialised there is
        nothing to drain, so we no-op.
        """
        backend = getattr(self, '_backend', None)
        if backend is not None:
            backend.shutdown()
