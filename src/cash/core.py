"""Main Cash class - decorator-based caching with automatic dependency tracking.

Provides the `Cash` entry point for ``@cash.cache`` function-level
caching. The notebook path lives in `cash.notebook`; it reads the decorator's
per-call events through `Cash.drain_decorator_calls`.
"""

from __future__ import annotations

import atexit
import datetime
import inspect
import logging
import os
import sys
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from . import _log
from ._console import encodable
from .analytics import AnalyticsManager
from .backends import CacheBackend
from .backends.factory import build_tiered
from .config.resolve import get_config
from .config.schema import CashConfig
from .content_hashers import builtin_hash_family
from .data_source import DataSource
from .decorator.arg_hashing import (
    CODE_VALUE_TYPES,
    ArgHasher,
    mark_opaque,
)
from .decorator.backend_slot import BackendSlot
from .decorator.cached_function import CHUNK_MAX_BYTES, CHUNK_MAX_ITEMS, CachedFunction, checked_ttl
from .decorator.cash_key import KeyCheck
from .decorator.class_data import ClassDataFold
from .decorator.closure_fold import CaptureAnalysis, ClosureFold, HelperIdentity
from .decorator.code_args import CodeArgs
from .decorator.code_surface import CodeSurface
from .decorator.environment_fold import EnvironmentFold
from .decorator.explain import (
    CacheExplanation,
    Explainer,
    MissHistory,
)
from .decorator.file_deps import FileDeps
from .decorator.frozen import FrozenResults
from .decorator.function_identity import OwnSourcePins, func_key, hash_callable_source
from .decorator.global_reads import GlobalReads
from .decorator.global_values import GlobalValues
from .decorator.globals_fold import GlobalsFold
from .decorator.maintenance import Maintenance
from .decorator.method_deps import MethodClassDeps
from .decorator.module_attrs import ModuleAttrFold
from .decorator.purity_checks import LearnedMutations, PurityChecks
from .decorator.registry import FunctionRegistry, checked_depends_on, warn_inert_dependency
from .decorator.reporting import CallLog, Notices
from .decorator.rng import RngWatch
from .decorator.run_summary import RunSummary
from .decorator.runtime import CallRunner, KeyBuilder
from .decorator.script_pickling import refuse_pickling_by_value
from .decorator.store import ResultStore
from .decorator.stored_keys import StoredKeyRecord
from .decorator.wrappers import Wrappers
from .dependency_state import (
    DependencyStateHasher,
    SysModulesHelperResolver,
)
from .diagnostics import (
    warn_diagnostic,
)
from .effectiveness import EffectivenessLedger
from .exceptions import (
    CashCacheIneffectiveWarning,
)
from .graph import DependencyGraph
from .reconfigure import apply_overrides
from .tracking.file_tracker import install_read_watch
from .tracking.reader_patches import file_registry

if TYPE_CHECKING:
    from .ui.explorer import CacheExplorer

# Configure Logging
logger = logging.getLogger(__name__)


def get_ipython():
    """Return the live IPython shell, or ``None``.

    Resolved on FIRST CALL rather than at import. ``from IPython import
    get_ipython`` pulls the whole package (``IPython.terminal.embed``
    included), which is most of the cost of ``import cash`` when paid up
    front, in front of every kernel start and every subprocess.

    Outside IPython this is the common case and stays cheap: ``sys.modules`` is
    consulted first, so a plain script never imports IPython at all. Inside a
    notebook IPython is already imported, so the lookup is free.
    """
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        # Not already imported. In a notebook it always is, so reaching here
        # means we are not in one -- do not pay the import to find that out.
        return None
    try:
        return ipython_module.get_ipython()
    except Exception:  # noqa: BLE001 - never break a call over shell detection
        return None


P = ParamSpec("P")
T = TypeVar("T")


__all__ = ["Cash", "CacheExplanation"]


def _declared_files(file_depends_on: str | list[str] | None) -> tuple[tuple[str, str], ...]:
    """``file_depends_on=`` as ``(as written, absolute)`` pairs. Each miss records
    the absolute paths as if the body had read them (`FileDeps.track_declared_files`);
    the paths as written are in the key (`FileDeps.fold_declared_files`)."""
    if not file_depends_on:
        return ()
    paths = [file_depends_on] if isinstance(file_depends_on, (str, os.PathLike)) else file_depends_on
    return tuple((str(p), os.path.abspath(p)) for p in paths)


class _ExitWork:
    """What a `Cash` must finish at exit: its stored-key record and its
    backend's pending writes, plus the end-of-run CACHE-NET-LOSS verdicts.

    Held apart from the instance, so that ``atexit`` keeps only this alive
    and a discarded instance can be collected. Not run when the instance is
    collected: a finalizer runs on whatever thread the collection happens
    on -- a writer thread, which would wait on itself.
    """

    __slots__ = ("backend_slot", "effectiveness", "stored_keys")

    def __init__(
        self, backend_slot: BackendSlot, stored_keys: StoredKeyRecord, effectiveness: EffectivenessLedger
    ) -> None:
        self.backend_slot = backend_slot
        self.stored_keys = stored_keys
        self.effectiveness = effectiveness

    def run(self) -> None:
        """Warn the run's verdicts, then drain. Never builds a backend: at exit
        that would start threads the interpreter can no longer register."""
        try:
            found = self.effectiveness.final_verdicts()
        except Exception:  # noqa: BLE001 - a notice must never block shutdown
            found = []
        for what, fix in found:
            try:
                warn_diagnostic(CashCacheIneffectiveWarning, "CACHE-NET-LOSS", what, fix)
            except Exception:  # noqa: BLE001 - -W error at exit, or teardown
                pass
        backend = self.backend_slot.built
        if backend is not None:
            self.stored_keys.close()
            backend.shutdown()


def _constructor_overrides(
    backend: CacheBackend | str | None,
    use_locking: bool,
    config_overrides: dict[str, Any],
    **convenience: Any,
) -> tuple[CacheBackend | None, bool]:
    """Check `Cash`'s arguments and fold the settings among them into
    *config_overrides*; return the backend instance (if one was given) and
    ``use_locking`` as a bool.

    *convenience* is the named settings `Cash` takes as its own keywords
    (``cache_dir``, ``compress``, ``debug``, ``verbose``).
    """
    # ``backend`` is also the name of a setting ("tiered", "file",
    # "sqlite", ...), and a setting may be passed by name: a string is
    # that setting, never a backend instance.
    if isinstance(backend, str):
        config_overrides.setdefault("backend", backend)
        backend = None
    elif backend is not None and not isinstance(backend, CacheBackend):
        raise TypeError(
            f"Cash(backend=...) takes a backend instance or a backend type name such as 'sqlite', "
            f"not {type(backend).__name__}"
        )
    if not isinstance(use_locking, bool) and use_locking not in (0, 1):
        raise ValueError(f"Cash(use_locking={use_locking!r}): expected True or False")
    use_locking = bool(use_locking)
    # Map the explicit convenience kwargs (cache_dir, compress, debug)
    # into the overrides dict so the config layer treats them with the
    # same priority as any other constructor-supplied override (highest).
    for key, val in convenience.items():
        if val is not None:
            config_overrides.setdefault(key, val)
    return backend, use_locking


def _summary_at_exit(ref: weakref.ref[Cash]) -> None:
    """Print the run summary if the ``summary`` setting is on NOW: read at
    exit, so ``cash.configure(summary=...)`` after the instance was built
    switches it either way."""
    cash = ref()
    if cash is not None and cash.config.summary:
        cash._summary.print_at_exit()


def _in_kernel() -> bool:
    """Whether this runs inside a Jupyter kernel, where widgets can be drawn."""
    return getattr(get_ipython(), "kernel", None) is not None


class Cash:
    """A cache with its own configuration and backend.

    ``cash.cache`` uses a default instance; create your own when you need
    different settings. Without ``backend`` or ``backends``, the backend is
    built from the configuration on first use: by default a RAM tier in
    front of a disk tier in ``.cash``.

    Args:
        backend: A backend instance to use as is, or the name of a backend
            type (``"file"``, ``"sqlite"``, ...: the ``backend`` setting) to
            build from the configuration.
        cache_dir: Directory for the disk tier (the ``cache_dir`` setting).
        backends: Backends to stack as tiers, fastest first; two or more are
            combined into a `TieredBackend`.
        compress: gzip entries on disk (the ``compress`` setting).
        register_magic: Register the notebook magics. ``None`` (default)
            registers them only when an IPython session is running.
        debug: Log every cache decision (the ``debug`` setting).
        use_locking: Take a per-key lock so concurrent calls with the same
            arguments compute once.
        config_path: A TOML file read above the project and user config.
        verbose: Log one line per cached call (the ``verbose`` setting).
        **config_overrides: Any other setting by name, for example
            ``Cash(max_cache_size="2GB")``. These win over every config file
            and environment variable.

    Raises:
        ValueError: a keyword that is not a setting, or a value the setting
            cannot take, as ``cash.configure`` raises for the same.

    Examples:
        ```python
        from cash import Cash
        c = Cash()

        @c.cache
        def expensive(x):
            return x ** 2
        ```
    """

    graph: DependencyGraph
    functions: dict[str, Callable[..., Any]]
    data_sources: dict[str, DataSource]
    source_hashes: dict[str, str]
    debug: bool | None
    use_locking: bool
    config: CashConfig

    def __reduce__(self):
        """A Cash cannot be pickled -- it holds locks, threads and a backend.
        See `refuse_pickling_by_value`."""
        refuse_pickling_by_value()

    @staticmethod
    def get_func_key(func: Callable) -> str:
        """Return a module-qualified key for a function (``module.qualname``).

        See `cash.decorator.function_identity.func_key`.
        """
        return func_key(func)

    @staticmethod
    def mark_opaque(*types_: type) -> None:
        """Exclude *types_* from code-surface hashing: what ``cash.opaque`` records."""
        mark_opaque(*types_)

    @staticmethod
    def builtin_hashed_family(type_: type) -> str | None:
        """Which built-in content hasher claims *type_*, or ``None``.

        Tells a user at ``register_hasher`` time that the hasher they just
        handed over would never be consulted -- the moment they can still do
        something about it. See `cash.content_hashers.builtin_hash_family`.
        """
        return builtin_hash_family(type_)

    def __init__(
        self,
        backend: CacheBackend | str | None = None,
        cache_dir: str | None = None,
        backends: list[CacheBackend] | None = None,
        compress: bool | None = None,
        register_magic: bool | None = None,
        debug: bool | None = None,
        use_locking: bool = False,
        config_path: str | None = None,
        verbose: bool | None = None,
        **config_overrides: Any,
    ) -> None:
        backend, use_locking = _constructor_overrides(
            backend, use_locking, config_overrides, cache_dir=cache_dir, compress=compress, debug=debug, verbose=verbose
        )
        self.config = get_config(config_path=config_path, overrides=config_overrides or None)
        self._init_infrastructure(backend, backends)
        self.use_locking = use_locking
        # Asking for debug output has to produce some, also in a script that
        # configured no logging.
        debug, verbose = self.config.debug, self.config.verbose
        if debug or verbose:
            _log.enable(logging.DEBUG if debug else logging.INFO)
        self._init_key_path()
        self._init_call_path()

        atexit.register(self._exit_work.run)

        # register_magic=None (default) auto-detects: only register when an
        # active IPython session exists.  True forces registration; False skips.
        if register_magic is True or (register_magic is None and get_ipython() is not None):
            self.register_magic()

    def _init_infrastructure(self, backend: CacheBackend | None, backends: list[CacheBackend] | None) -> None:
        """The backend slot, the registries, the record of stored keys and
        what reports to the user: what both the key path and the call path
        are built over."""
        # An explicit backend (or list of backends) wins over the config: those
        # are concrete objects, not settings, and the factory is skipped.
        if backend is None and backends:
            backend = build_tiered(backends, self.config) if len(backends) > 1 else backends[0]
        self._backend_slot = BackendSlot(self.config, backend)
        self._analytics: AnalyticsManager | None = None

        self._registry = FunctionRegistry()
        # The registry's tables, under the names Cash publishes them by.
        self.graph = self._registry.graph
        self.functions = self._registry.functions
        self.data_sources = self._registry.data_sources
        self.source_hashes = self._registry.source_hashes
        # Per instance: two Cash instances are two independent caches, and
        # each accounts for itself. Registered whatever ``summary`` says now,
        # since ``configure(summary=True)`` may turn it on later. Through a
        # weakref, so the hook does not keep the instance alive; registered
        # before the exit work, so it runs after it.
        atexit.register(_summary_at_exit, weakref.ref(self))
        # The keys earlier runs stored, recorded beside the cache.
        self._stored_keys = StoredKeyRecord(self._backend_slot.local_dir)
        self._notices = Notices(self._registry.cached, self.functions, self._stored_keys, self._backend_slot)
        self._misses = MissHistory(self._registry.cached, self._stored_keys, self._backend_slot)
        effectiveness = EffectivenessLedger()
        self._calls = CallLog(self.config, self._registry.cached, self._misses, effectiveness)
        self._exit_work = _ExitWork(self._backend_slot, self._stored_keys, effectiveness)
        self._frozen = FrozenResults(self.config, self._notices)

    def _init_key_path(self) -> None:
        """The steps that turn a call into its cache key."""
        self._args = ArgHasher(
            self._registry.cached,
            self._frozen,
            self._notices,
            KeyCheck(self._backend_slot.local_dir, lambda: self.config.check_cash_keys),
        )
        self._code = CodeSurface(self._args)
        self._pins = OwnSourcePins()
        self._captures = CaptureAnalysis()
        self._helpers = HelperIdentity(self._args, self._captures)
        self._mutations = LearnedMutations()
        # Deep seam over the registries above: folds source/dependency/
        # helper state into the cache key's ``state_hash`` segment. Borrows
        # the registry dicts by reference so later registrations are seen.
        self._state_hasher = DependencyStateHasher(
            functions=self.functions,
            data_sources=self.data_sources,
            source_hashes=self.source_hashes,
            purity_reports=self._registry.purity_reports,
            graph=self.graph,
            helper_resolver=SysModulesHelperResolver(self._helpers.identity),
            declared_dep_snapshots=self._registry.declared_dep_snapshots,
            declared_dep_resolver=self._registry.resolve_declared_dep_hash,
            reached_callee=self._registry.reached_callee,
        )
        self._reads = GlobalReads()
        self._values = GlobalValues(self._args, self._helpers, self._notices)
        self._classes = ClassDataFold(
            self._args, self._reads, self._values, self._registry, self._mutations, self._notices
        )
        self._globals = GlobalsFold(
            self._reads,
            self._values,
            self._classes,
            ModuleAttrFold(self._args, self._reads, self._values, self._classes),
            self._code,
            self._registry,
            self._mutations,
            self._notices,
        )
        self._closures = ClosureFold(
            self._args, self._captures, self._helpers, self._values, self._classes, self._mutations, self._notices
        )
        self._code_args = CodeArgs(self._code, self._globals, self._values, self._classes, self._frozen, self._args)
        self._rng = RngWatch(self._registry, self._backend_slot, self._notices)
        self._files = FileDeps(self._registry, self._notices)
        self._purity = PurityChecks(
            self.config,
            self._registry,
            self._args,
            self._frozen,
            self._reads,
            self._values,
            self._classes,
            self._mutations,
            self._notices,
        )
        self._keys = KeyBuilder(
            self._registry,
            self._args,
            MethodClassDeps(self._args),
            self._pins,
            self._files,
            self._closures,
            self._globals,
            EnvironmentFold(self._registry),
            self._rng,
            self._code_args,
            self._state_hasher,
            self._misses,
            self._notices,
        )

    def _init_call_path(self) -> None:
        """The steps a call runs through: lookup, body, store, and the
        wrappers ``cache`` hands back."""
        self._store = ResultStore(
            self._registry,
            self._backend_slot,
            self._frozen,
            self._files,
            self._purity,
            self._misses,
            self._notices,
            self._stored_keys,
        )
        self._runner = CallRunner(
            self._registry,
            self._backend_slot,
            self._keys,
            self._store,
            self._calls,
            self._files,
            self._purity,
            self._rng,
            self._misses,
            self._notices,
        )
        self._maintenance = Maintenance(self._registry, self._backend_slot)
        self._summary = RunSummary(lambda: self.config, self._registry, self._backend_slot)
        self._explainer = Explainer(
            self.config,
            self._registry,
            self._keys,
            self._args,
            self._frozen,
            self._backend_slot,
            self._misses,
            self._runner,
        )

        self._wrappers = Wrappers(
            lambda: self.config,
            lambda: self.use_locking,
            self._registry,
            self._backend_slot,
            self._keys,
            self._runner,
            self._rng,
            self._explainer,
            self._maintenance,
            self._notices,
        )

    @property
    def backend(self) -> CacheBackend:
        """The cache backend, built from the configuration on first use.

        Assign a backend instance to replace it.
        """
        return self._backend_slot.backend

    @backend.setter
    def backend(self, value: CacheBackend) -> None:
        """Allow direct assignment (e.g. ``c.backend = MyBackend()``)."""
        self._backend_slot.backend = value

    @property
    def analytics(self) -> AnalyticsManager:
        """This session's analytics: the one manager its events are recorded
        on and the dashboard reads, so "Current Session" is this one and its
        buffered events are counted. Created on first use."""
        if self._analytics is None:
            self._analytics = AnalyticsManager(enabled=self.config.analytics)
        return self._analytics

    @property
    def backend_if_built(self) -> CacheBackend | None:
        """The backend if one has been built, else ``None``; never builds one."""
        return self._backend_slot.built

    @property
    def debug(self) -> bool:
        """``config.debug``: log every cache decision."""
        return bool(self.config.debug)

    @debug.setter
    def debug(self, value: bool) -> None:
        self.config.debug = bool(value)

    @property
    def verbose(self) -> bool:
        """``config.verbose``: log one line per decorated call."""
        return bool(self.config.verbose)

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.config.verbose = bool(value)

    def reconfigure(self, **overrides: Any) -> None:
        """Change settings at runtime, rebuilding the backend only when the
        tiers it is built from changed. See ``cash.configure``."""
        apply_overrides(self, overrides)

    def __repr__(self) -> str:
        built = self._backend_slot.built
        backend_name = type(built).__name__ if built is not None else "<deferred>"
        n_funcs = len(self.functions)
        return f"Cash(backend={backend_name}, functions={n_funcs}, debug={self.debug})"

    @overload
    def cache(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def cache(
        self,
        func: None = None,
        *,
        depends_on: Callable[..., Any] | DataSource | list[Callable[..., Any] | DataSource] | None = ...,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = ...,
        file_depends_on: str | list[str] | None = ...,
        ttl: float | datetime.timedelta | None = ...,
        cache_if: Callable[[Any], bool] | None = ...,
        chunk_max_items: int = ...,
        chunk_max_bytes: int = ...,
        strict: bool = ...,
        assume_safe: bool = ...,
        allow_random: bool = ...,
        frozen: bool = ...,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def cache(
        self,
        func: Callable[P, T] | None = None,
        *,
        depends_on: Callable[..., Any] | DataSource | list[Callable[..., Any] | DataSource] | None = None,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        file_depends_on: str | list[str] | None = None,
        ttl: float | datetime.timedelta | None = None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = CHUNK_MAX_ITEMS,
        chunk_max_bytes: int = CHUNK_MAX_BYTES,
        strict: bool = False,
        assume_safe: bool = False,
        allow_random: bool = False,
        frozen: bool = False,
    ) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
        """Cache a function's results, keyed on its arguments, code and inputs.

        Use it bare (``@c.cache``) or with options (``@c.cache(ttl=3600)``).
        The decorated function also gets ``explain``, ``cache_info`` and
        ``cache_clear``. The decorator guide explains each option.

        Args:
            func: The function; set for you when used without parentheses.
            depends_on: Functions or `DataSource` objects whose changes
                invalidate the entry: a list, or one on its own.
            dynamic_depends_on: Callable(s) that receive the call's arguments
                and return the `DataSource` (or a list, or ``None``) that
                call depends on.
            file_depends_on: Path(s) treated as read by the function: their
                content is checked on every lookup.
            ttl: Seconds an entry stays valid (a number ``>= 0`` or a
                `datetime.timedelta`). ``None``: no expiry. ``0``: recompute
                every call.
            cache_if: ``predicate(result) -> bool``. A result it rejects is
                returned but not stored.
            chunk_max_items: For an iterator result, items per stored chunk.
            chunk_max_bytes: For an iterator result, bytes per stored chunk.
            strict: Raise `CashImpureFunctionError` on the first call for any
                purity finding.
            assume_safe: Waive every purity finding for this function.
            allow_random: Silence the unseeded-randomness warning. The first
                draw is still what is stored.
            frozen: Promise the result is never modified, so cached functions
                that receive it key it by this call instead of hashing it.

        Returns:
            The wrapped function, or a decorator when called with options.

        Raises:
            ValueError: ``strict`` and ``assume_safe`` are both set, or
                ``ttl`` is negative, NaN or infinite.
            TypeError: ``ttl`` is not a number, timedelta or ``None``, or a
                ``depends_on`` entry is neither a callable nor a `DataSource`.
        """
        ttl = checked_ttl(ttl)
        depends_on = checked_depends_on(depends_on)
        if strict and assume_safe:
            raise ValueError(
                "@cash.cache: strict=True and assume_safe=True are mutually "
                "exclusive. strict raises on purity issues; assume_safe silences "
                "them. Pick one."
            )

        if func is None:
            return lambda f: self.cache(
                f,
                depends_on=depends_on,
                dynamic_depends_on=dynamic_depends_on,
                file_depends_on=file_depends_on,
                ttl=ttl,
                cache_if=cache_if,
                chunk_max_items=chunk_max_items,
                chunk_max_bytes=chunk_max_bytes,
                strict=strict,
                assume_safe=assume_safe,
                allow_random=allow_random,
                frozen=frozen,
            )

        cf = CachedFunction(
            func,
            func_key(func),
            dynamic_depends_on=dynamic_depends_on,
            ttl=ttl,
            cache_if=cache_if,
            chunk_max_items=chunk_max_items,
            chunk_max_bytes=chunk_max_bytes,
            purity="strict" if strict else "silent" if assume_safe else "warn",
            frozen=frozen,
            allow_random=allow_random,
            declared_files=_declared_files(file_depends_on),
        )
        func_name = cf.name
        for dep in self._registry.register(cf, depends_on):
            warn_inert_dependency(self._notices, func_name, dep)
        self._pins.pin_own_source(func, self.source_hashes[func_name])

        # Async generators are not cached; warn once and return unwrapped.
        if inspect.isasyncgenfunction(func):
            self._notices.warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: async generators are not cached "
                f"in this release, so the function was returned unwrapped.",
                code="CACHE-ASYNC-GENERATOR",
                fix="move the work into a plain async def that returns a list "
                "and cache that, or leave the generator undecorated and "
                "cache the expensive step inside it.",
            )
            return func

        # Unseeded-randomness check. Deliberately here and not in the
        # wrapper: it is a pure function of the source, so it runs ONCE per
        # decorated function and adds nothing to the per-call path. Placed after
        # the async-generator early return because that path is not cached at
        # all, and the hazard being warned about is a frozen cached value.
        self._rng.warn_unseeded_randomness(func, func_name, allow_random)

        # Watch reads from now, not from the first miss: a memo the cached
        # function will use is usually filled before it is first called
        # (`main()` logging its settings), and a read nobody saw is an input
        # no entry records. Not when caching is off: that promises
        # nothing is patched or analysed.
        if not self.config.disable:
            try:
                install_read_watch()
            except Exception:  # the first miss installs them anyway
                logger.debug("[CORE] could not install the read watch at decoration", exc_info=True)

        return self._wrappers.wrap(cf)

    def drain_decorator_calls(self) -> list[dict[str, Any]]:
        """Return and clear all recorded decorator call events.

        Thread-safe: atomically copies and clears the log.
        Called by the notebook statement processor after executing a statement
        to collect decorator-level cache metrics for badge display.

        Returns:
            List of call event dicts, each with keys:
            ``func_name``, ``cache_hit``, ``execution_time``, ``time_saved``,
            ``args_hash``, ``cache_key``, ``timestamp``.

            ``execution_time`` is what this call cost; ``time_saved`` is the
            recorded cost of the original computation a hit avoided, so it is
            an estimate carried forward from the write, not a measurement of
            this call.
        """
        return self._calls.drain()

    def register_hasher(
        self,
        type_: type,
        hasher_fn: Callable[[Any], str],
        *,
        override: bool = False,
    ) -> None:
        """Tell cash how to hash arguments of a type it cannot hash itself.

        Whatever ``hasher_fn`` returns becomes that value's identity in the
        cache key, so return something that changes whenever the value does:
        a version, a content id, a connection string. The hasher's own code
        is part of the key too.

        Args:
            type_: The type to hash.
            hasher_fn: Takes a value of ``type_`` and returns a stable string.
            override: Use your hasher instead of cash's own, for the types
                cash hashes itself (numpy arrays, pandas, polars, PyArrow and
                modin frames, dask collections). Two values it hashes alike
                share one entry.

        Raises:
            ValueError: ``type_`` is one cash hashes itself and ``override``
                is not set.

        Examples:
            ```python
            c.register_hasher(DatabaseSession, lambda s: s.database_url)
            ```
        """
        # Rejected BEFORE anything is mutated, so a refused call leaves an
        # earlier good registration for this type exactly as it was.
        if not override:
            family = self.builtin_hashed_family(type_)
            if family is not None:
                raise ValueError(
                    f"cash.register_hasher({type_.__name__}): cash fingerprints "
                    f"{family} values itself and that runs first, so this hasher "
                    f"would never be called -- registering it does nothing at "
                    f"all.\n"
                    f"Either drop the registration (cash already hashes this "
                    f"type by content, correctly), or pass override=True to use "
                    f"yours instead.\n"
                    f"Be deliberate about override: what your hasher returns "
                    f"becomes the entire identity of the value, so any two "
                    f"values it hashes alike share one cache entry and the "
                    f"second call gets the first one's result. That is right "
                    f"when you hold a version or content id the value does not "
                    f"carry, and wrong for something like lambda a: a[0, 0]."
                )

        # Allowed -- a hasher that returns what the function captures is
        # correct -- but the one people write is keyed on the name, and that
        # hands one closure's cached result to the next.
        if isinstance(type_, type) and issubclass(type_, CODE_VALUE_TYPES):
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-CALLABLE-HASHER",
                f"cash.register_hasher({type_.__name__}, ...) decides the "
                f"identity of every {type_.__name__} passed to a cached "
                f"function in this process. Closures one factory makes share a "
                f"name and a body, so a hasher that does not return what they "
                f"capture gives them one cache entry, and the second gets the "
                f"first one's result.",
                "prefer passing the captured values to the cached function as "
                "plain arguments, with a module-level function in the closure's "
                "place; if you keep this hasher, make it return the captured "
                "values too.",
            )
        self._args.register_hasher(type_, hasher_fn, hash_callable_source(hasher_fn), override=override)

    def cleanup(self, max_age: int | None = None) -> int:
        """Remove expired entries from this instance's backend.

        Args:
            max_age: Also remove entries older than this many seconds,
                whatever their TTL.

        Returns:
            The number of entries removed.
        """
        return self._maintenance.cleanup(max_age)

    def explorer(self) -> CacheExplorer:
        """Return a `CacheExplorer` for browsing this instance's entries."""
        # Local: the explorer imports pandas, which `import cash` must not load.
        from .ui.explorer import CacheExplorer

        return CacheExplorer(self)

    def run_summary(self) -> str:
        """A per-function hit/miss table for this process, or ``""``.

        Empty when no cached function was ever called, so a caller can print
        this unconditionally without emitting a header over nothing.
        """
        return self._summary.text()

    def show_stats(self) -> None:
        """Show what caching has done in this process.

        With ipywidgets installed (in Jupyter), shows the notebook dashboard,
        which covers notebook statements only. Otherwise prints the
        per-function table of decorated calls that ``summary`` prints at
        exit.
        """

        # Local: the dashboard imports matplotlib, which `import cash` must not load.
        from .ui.dashboard import HAS_WIDGETS, show_analytics_dashboard

        # Asking the dashboard whether it CAN run, rather than calling it and
        # catching: without ipywidgets it prints which package to install and
        # returns normally, so there is nothing to catch. And whether anything
        # can draw it: outside a kernel, displaying the widgets only prints
        # their repr.
        if HAS_WIDGETS and _in_kernel():
            try:
                show_analytics_dashboard(self.analytics)
                return
            except (ImportError, RuntimeError):
                pass
        text = self.run_summary()
        # Escaped where stdout's encoding lacks a character: a function named
        # outside cp1252 raised UnicodeEncodeError into a Windows pipe.
        print(encodable(text) if text else "cash: no cached function has been called yet.")

    def register_magic(self) -> None:
        """Register IPython magic commands (``%cash_on``, ``%cash_stats``, etc.)."""
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
        # Local: the plugin entry; core never imports the notebook package otherwise.
        from .notebook.ipython.magics import CashMagics

        magics = CashMagics(ip, self)
        ip.register_magics(magics)

    def clear_all(self) -> None:
        """Call ``cache_clear()`` on every function this instance caches.

        Entries of functions not decorated in this process are kept.
        """
        for cf in list(self._registry.cached.values()):
            if cf.wrapper is not None:
                cf.wrapper.cache_clear()

    def register_file_handler(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]) -> None:
        """Track the files a reader function cash does not know opens.

        Cash already tracks common readers (``open``, ``pd.read_csv``,
        ``np.load``, ...). For another one, give a factory that wraps the
        reader and reports each path it opens; cash installs the wrapper on
        the module, so library code that calls it is tracked too. Tracked
        files are checked by content, like any other read.

        Args:
            module_name: The module that holds the reader, such as
                ``"my_lib.io"``.
            func_name: The reader's name. Glob patterns such as ``"read_*"``
                match several.
            handler_factory: ``factory(original, track) -> wrapper``. The
                wrapper calls ``track(path)`` for each file, then the
                original.

        Examples:
            ```python
            def handler(original, track):
                def wrapper(path, *args, **kwargs):
                    track(path)
                    return original(path, *args, **kwargs)
                return wrapper

            c.register_file_handler("my_lib", "read_data", handler)
            ```
        """

        file_registry().register(module_name, func_name, handler_factory)

    def shutdown(self) -> None:
        """Finish pending work: report net-loss verdicts and wait for the
        backend's background writes. Runs at exit on its own."""
        self._exit_work.run()
