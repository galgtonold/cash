"""Main Cash class - decorator-based caching with automatic dependency tracking.

Provides the `Cash` entry point for ``@cash.cache`` function-level
caching. The notebook path lives in `cash.notebook`; it reads the decorator's
per-call events through `Cash.drain_decorator_calls`.
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import inspect
import logging
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from . import _log
from .backends import CacheBackend, CacheMetadata
from .backends._base import ttl_expired
from .backends._writes import in_multiprocessing_child
from .backends.factory import build_backend_from_config, build_tiered
from .config import CashConfig, get_config
from .data_source import DataSource
from .decorator.arg_hashing import (
    CODE_VALUE_TYPES,
    ArgHashingMixin,
)
from .decorator.cached_function import CHUNK_MAX_BYTES, CHUNK_MAX_ITEMS, CachedFunction, new_stats
from .decorator.call_state import (
    CACHE_MISS,
    CALL_ENTRY,
    enter_cached_call,
    exit_cached_call,
)
from .decorator.closure_fold import ClosureFoldMixin
from .decorator.code_args import CodeArgsMixin
from .decorator.code_identity import (
    CodeIdentityMixin,
)
from .decorator.explain import (
    CacheExplanation,
    ExplainMixin,
    MissKind,
    MissReason,
)
from .decorator.file_deps import FileDepsMixin
from .decorator.frozen import FrozenMixin
from .decorator.globals_fold import (
    GlobalsFoldMixin,
)
from .decorator.purity_checks import (
    PurityChecksMixin,
)
from .decorator.registry import RegistryMixin
from .decorator.reporting import ReportingMixin
from .decorator.rng import RngMixin
from .decorator.runtime import RuntimeMixin
from .decorator.script_pickling import expose_script_function
from .decorator.store import StoreMixin
from .decorator.stored_keys import StoredKeyRecord
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
from .purity_analyzer import (
    PurityReport,
)
from .reconfigure import apply_overrides
from .source_norm import (
    callable_identity,
)
from .tracking.file_dep_snapshot import (
    ACTIVE_CONFIG,
)
from .tracking.file_tracker import (
    file_registry,
    install_read_watch,
)

if TYPE_CHECKING:
    from .ui.explorer import CacheExplorer

# Configure Logging
logger = logging.getLogger(__name__)


def get_ipython():
    """Return the live IPython shell, or ``None``.

    Resolved on FIRST CALL rather than at import. ``from IPython import
    get_ipython`` looks cheap but pulls the whole package -- measured at ~4s of
    the ~10s ``import cash``, most of it ``IPython.terminal.embed``. That cost
    sits in front of every kernel start and every subprocess a test spawns, and
    a test running three subprocesses tripped the 30s per-test timeout on
    imports alone.

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


def _backend_cache_dir(backend: CacheBackend | None) -> str | None:
    """The directory *backend* keeps entries in -- its disk tier's, if tiered."""
    directory = backend.local_dir if backend is not None else None
    return os.path.abspath(directory) if directory else None


def _local_dir_of(ref: weakref.ref[Cash]) -> Callable[[], str | None]:
    """The built backend's local directory, read through *ref* so the
    stored-key record does not keep its `Cash` alive."""

    def local_dir() -> str | None:
        cash = ref()
        backend = cash._backend if cash is not None else None
        return backend.local_dir if backend is not None else None

    return local_dir


def _declared_files(file_depends_on: str | list[str] | None) -> tuple[tuple[str, str], ...]:
    """``file_depends_on=`` as ``(as written, absolute)`` pairs. Each miss records
    the absolute paths as if the body had read them (`_track_declared_files`);
    the paths as written are in the key (`_fold_declared_files`)."""
    if not file_depends_on:
        return ()
    paths = [file_depends_on] if isinstance(file_depends_on, str) else file_depends_on
    return tuple((str(p), os.path.abspath(p)) for p in paths)


class _ExitWork:
    """What a `Cash` must finish at exit: its stored-key record and its
    backend's pending writes, plus the end-of-run CACHE-NET-LOSS verdicts.

    Held apart from the instance, so that ``atexit`` keeps only this alive
    and a discarded instance can be collected. Not run when the instance is
    collected: a finalizer runs on whatever thread the collection happens
    on -- a writer thread, which would wait on itself.
    """

    __slots__ = ("backend", "effectiveness", "stored_keys")

    def __init__(self, stored_keys: StoredKeyRecord, effectiveness: EffectivenessLedger) -> None:
        self.backend: CacheBackend | None = None
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
        if self.backend is not None:
            self.stored_keys.close()
            self.backend.shutdown()


def _summary_at_exit(ref: weakref.ref[Cash]) -> None:
    cash = ref()
    if cash is not None:
        cash._print_run_summary()


#: How many call events `Cash._decorator_call_log` holds. The notebook drains it
#: after every statement; nothing drains it in a script or a service, so it
#: keeps only the most recent calls rather than one entry per call forever.
_CALL_LOG_MAX = 10_000


class Cash(
    CodeIdentityMixin,
    CodeArgsMixin,
    ClosureFoldMixin,
    GlobalsFoldMixin,
    ArgHashingMixin,
    FrozenMixin,
    RngMixin,
    FileDepsMixin,
    PurityChecksMixin,
    ExplainMixin,
    RuntimeMixin,
    StoreMixin,
    ReportingMixin,
    RegistryMixin,
):
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
        verbose: Log one line per cached call -- hit or miss, and why -- to
            stderr. ``debug`` does too, and adds cash's DEBUG output.
        **config_overrides: Any `CashConfig` field by name, e.g.
            ``Cash(max_cache_size=2 * 1024**3)``; wins over every config file
            and environment variable.

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

    def __reduce__(self):
        """A Cash cannot be pickled -- it holds locks, threads and a backend.

        Reached when something sends a cached function BY VALUE to another
        process, which is what joblib's workers do with a function defined in
        the script being run. Say what works instead of letting the pickler
        report a lock (see `expose_script_function`).
        """
        raise TypeError(
            "a Cash instance cannot be pickled, and something tried to send a "
            "@cash.cache function to another process by value. A cached "
            "function defined in the script you run can be sent to worker "
            "processes (joblib, multiprocessing) when the script's work is "
            'behind `if __name__ == "__main__":`; or define the function in a '
            "module you import."
        )

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
        verbose: bool | None = None,
        **config_overrides: Any,
    ) -> None:
        # Map the explicit convenience kwargs (cache_dir, compress, debug)
        # into the overrides dict so the config layer treats them with the
        # same priority as any other constructor-supplied override (highest).
        for key, val in (("cache_dir", cache_dir), ("compress", compress), ("debug", debug), ("verbose", verbose)):
            if val is not None:
                config_overrides.setdefault(key, val)
        self.config = get_config(config_path=config_path, overrides=config_overrides or None)

        debug = self.config.debug

        # Store params for lazy backend construction. If an explicit
        # backend (or list of backends) was provided, that wins - those
        # are concrete objects, not config - and we skip the factory.
        self._backend: CacheBackend | None = None
        if backend is not None:
            self._backend = backend
        elif backends:
            if len(backends) > 1:
                self._backend = build_tiered(backends, self.config)
            else:
                self._backend = backends[0]
        # Set with the stored-key record below; `_backend` is mirrored into it.
        self._exit_work: _ExitWork | None = None

        self._backend_lock = threading.Lock()

        self.graph = DependencyGraph()
        self.functions: dict[str, Callable[..., Any]] = {}  # Registry of cached functions
        # func_name -> its decoration's options and per-process state.
        self._cached: dict[str, CachedFunction] = {}
        self.data_sources: dict[str, DataSource] = {}  # Registry of data sources
        self.source_hashes: dict[str, str] = {}  # Current source hashes
        # Session-scoped memo: id(arg) -> (weakref, lineage_hash, content_hash).
        # Lets a repeated ``@cash.cache`` call with the SAME unmutated argument
        # skip re-hashing a possibly-huge input. See ``_hash_arg_payload`` for
        # the read-side validation (weakref identity + lineage). Bounded below.
        self._arg_hash_memo: dict[int, tuple] = {}
        # id(frame) -> (weakref, shallow copy, signature, content hash): the
        # pandas copy-on-write memo, see `_frame_memo_store`.
        self._frame_memo: dict[int, tuple] = {}
        # id(ndarray) -> [weakref, producer, content hash or None]: numpy
        # results of frozen functions, which cannot carry a tag.
        self._frozen_arrays: dict[int, list] = {}
        # id(obj) -> [weakref, uses, audit baseline or None], see `_audit_frozen`.
        self._frozen_uses: dict[int, list] = {}
        # id(obj) -> [obj, producer, lineage, uses, audit baseline]: a frozen
        # function's list / tuple / dict result, which carries no tag and no
        # weakref -- so the object is held here, while someone else holds it
        # too (`_remember_frozen_container`).
        self._frozen_containers: dict[int, list] = {}
        # Running account of what caching cost vs what it saved, per function.
        # The decorator always caches by design -- this only ever informs.
        self._effectiveness = EffectivenessLedger()
        if self.config.summary:
            # Per instance: two Cash instances are two independent caches, and
            # each accounts for itself. Through a weakref, so the hook does
            # not keep the instance alive; registered before the exit work,
            # so it runs after it.
            atexit.register(_summary_at_exit, weakref.ref(self))
        self._analyzed = set()  # Track which functions we've *surfaced* purity for
        # ONE lock for the one-time analysis, whatever function triggers it.
        #
        # The check-and-analyze below is not atomic, and the CACHE KEY depends
        # on what the analysis populates (helper source hashes, graph edges).
        # Concurrent first calls therefore resolved two different keys for one
        # call -- the threads that got there before the analysis finished, and
        # the one that did it -- so `use_locking=True` looked like it admitted
        # exactly two threads into the compute at every thread count. It was
        # not the lock: each key was single-flighted correctly, there were just
        # two of them, and the pre-analysis one is an entry no later run will
        # ever look up. Measured: warming the analysis in the main thread first
        # collapsed 6 threads to one key and one execution.
        #
        # RLock, not Lock: analysis walks the dependency graph and re-enters
        # this same guard for the callees it populates on the way.
        #
        # One lock rather than one per function, deliberately. Analysis of f
        # populates f's whole callee closure, so per-function locks could be
        # taken in two orders by two threads and deadlock. It is a one-time,
        # source-reading step measured in milliseconds; serialising unrelated
        # first calls behind it costs nothing worth a lock-ordering rule.
        self._analysis_lock = threading.RLock()
        # Track which functions have had their graph edges + purity report
        # populated (separate from _analyzed: a dependency can be populated to
        # complete a parent's state hash long before it is called directly and
        # surfaced). Keeps the cache key stable from the first call.
        self._populated: set[str] = set()
        self._effective_ttl_cache: dict[str, int | None] = {}
        self._deref_writes: dict = {}  # code object -> frozenset of reassigned freevars
        # id(func) -> (reference to func, decoration-pinned own-source
        # identity). The reference is checked on every read: a redefined
        # function's id can go to a later definition once the old one dies.
        self._own_pins: dict[int, tuple[Callable[[], Any], str]] = {}
        # Pins taken at decoration whose file has not yet been compared with
        # the loaded code; the first call does it once (see _pin_own_source).
        self._own_pins_unverified: set[int] = set()
        # code object -> global names its decorator expressions read
        self._decorator_names_cache: dict = {}
        # code object -> frozenset of free vars with capture-unsafe uses
        self._capture_use_cache: dict = {}
        # code object -> tuple of global names it reads (global folding)
        self._global_read_cache: dict = {}
        # code object -> names folded only provisionally. See
        # `_read_global_data_names`. A missing entry means "unknown", which
        # `_fold_read_globals` treats as "watch everything".
        self._provisional_global_cache: dict = {}
        # (code object, scope) -> names a call was OBSERVED to mutate. Learned
        # once, then those names stop being folded (see `_learn_mutating_captures`).
        self._mutating_globals: dict = {}
        # code object -> closure free vars folded only provisionally.
        self._provisional_capture_cache: dict = {}
        # (module_global, attribute) read pairs per code object; see
        # _read_module_attr_pairs.
        self._module_attr_cache: dict = {}
        self._local_binding_cache: dict[Any, tuple | None] = {}
        self._carrier_verdicts: dict[int, tuple[Any, bool]] = {}
        # ``(class, is user code)`` per class id, for `_iter_attribute_carriers`:
        # a list of 50k instances must not pay the verdict per element. The
        # class is kept so a recycled id is never trusted. Bounded there.
        self._attribute_walk_verdicts: dict[tuple[str, int], tuple[type, bool]] = {}
        # Code carriers already reported (`_warn_unhashable_code_once`,
        # `_warn_untrackable_in_carrier_once`): once per carrier and function.
        self._warned_unhashable_code: set[tuple] = set()
        self._warned_untrackable_carrier: set[tuple] = set()
        # (func_name, state segment) -> the ledger of the key build that first
        # produced it (`_keep_state_ledger`).
        self._state_ledgers: dict[tuple[str, str], dict] = {}
        self._stored_keys = StoredKeyRecord(_local_dir_of(weakref.ref(self)))
        self._exit_work = _ExitWork(self._stored_keys, self._effectiveness)
        self._exit_work.backend = self._backend
        # (first_param, self_attrs, uses_super) per code object; see
        # _analyze_method_self_deps.
        self._method_self_dep_cache: dict = {}
        # user class -> source hash. A class's source cannot change within a
        # running interpreter, so it is hashed once and reused; see
        # _user_class_source_hash / _instance_class_source_parts.
        self._user_class_src_cache: dict = {}
        # user class or function -> code-surface digest (bytecode-based, class-
        # aware); see _code_surface_hash. Keyed on the object itself, not
        # id(), so a redefinition (a new object) is a distinct memo entry.
        self._code_surface_cache: dict = {}
        # object -> tuple of (code object, globals dict) it carries. Static for
        # as long as that object exists (a redefinition makes a new one), so it
        # is safe to memo; the NAMES those code objects reference are resolved
        # fresh per call, because what a name is bound to can change.
        self._code_refs_cache: dict = {}
        # function object -> digest of its parameter defaults, for defaults that
        # are immutable and therefore cannot drift between calls.
        # Weak so the memo dies with the function instead of pinning it (and so
        # a later function object can never inherit a dead one's entry by
        # id-reuse). Mutable defaults are deliberately absent: they must be
        # re-hashed per call to stay correct.
        self._defaults_pins: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # id(helper) -> (helper, __defaults__, __kwdefaults__, identity); see
        # `_hash_helper_identity`. Holding the helper keeps its id from being
        # recycled while the entry lives.
        self._helper_defaults_memo: dict[int, tuple[Any, Any, Any, str]] = {}
        # In-process async single-flight registry: cache_key ->
        # concurrent.futures.Future. When use_locking is set, concurrent awaits
        # of the same key coalesce - one coroutine computes, the rest wait and
        # then read the stored result.
        #
        # A plain future rather than an asyncio.Event, because an Event belongs
        # to the loop that made it: with one slot per key, a leader in a second
        # loop replaced the first loop's event and its followers -- unable to
        # await another loop's event -- each computed for themselves (4 loops x
        # 4 awaits ran the body 16 times). `asyncio.wrap_future` attaches the
        # wait to whichever loop is asking, so every await in the process
        # coalesces, which is what the docs promise.
        self._async_inflight: dict[str, Any] = {}
        self._async_inflight_lock = threading.Lock()
        self.use_locking = use_locking
        verbose = self.config.verbose
        # Asking for debug output has to produce some, also in a script that
        # configured no logging.
        if debug or verbose:
            _log.enable(logging.DEBUG if debug else logging.INFO)

        # What a miss was, for the people asking "why did that recompute?".
        # Both are in-process memory only, and bounded: they explain, they
        # never decide anything. (The last key per function is on its
        # `CachedFunction`.)
        #: cache_key -> what happened when it was last computed here.
        self._store_outcomes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        #: cache_key -> (kind, detail) for a lookup that just missed, taken
        #: by the `_log_decorator_call` that reports it.
        self._pending_miss: dict[str, MissReason] = {}

        # Decorator call log for notebook integration.
        # Each entry is a dict with: func_name, cache_hit (bool), execution_time,
        # args_hash, cache_key, timestamp.  The notebook statement processor
        # drains this after executing each statement so it can include
        # decorator metrics in the badge. Bounded: outside a notebook nothing
        # drains it, and a long-running process must not keep every call.
        self._decorator_call_log: deque[dict[str, Any]] = deque(maxlen=_CALL_LOG_MAX)
        self._decorator_call_log_lock = threading.Lock()
        # Custom type hasher registry: maps type -> (callable(value) -> str, source hash).
        # The source hash is embedded in the args_hash composition so that
        # changing a hasher's body invalidates dependent cache entries.
        self._type_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}
        # Same shape, but consulted BEFORE cash's own content hashers rather
        # than after them. Separate registry rather than a flag in the tuple
        # above so the hot path can skip the whole question with one empty
        # check -- overriding is rare, and every cached call pays for this.
        self._override_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}

        # Dedup keys for _warn_once: (category, func_name, arg_type_name, code).
        # Guarded by _decorator_call_log_lock (already exists for thread safety).
        self._warning_keys_seen: set[tuple[type[Warning], str, str, str]] = set()

        # Functions the STATIC pass already reported on. The runtime effect
        # observer stays quiet for these: it would be a second warning about
        # the same function, and the user has already been told.
        self._purity_static_flagged: set[str] = set()
        # Per-function purity report cache. Populated on first call.
        # Helper source hashes from this report fold into the cache
        # key state hash so cross-process helper edits invalidate.
        self._purity_reports: dict[str, PurityReport] = {}

        # Declared plain-callable dependencies (``depends_on=[proxy_fn]`` where
        # proxy_fn is NOT a decorated cached function). Snapshot source hash at
        # registration + a ``(module, attr_chain)`` path for live re-resolution,
        # so editing the dep on disk + reload invalidates the parent key.
        self._declared_dep_snapshots: dict[str, str] = {}
        self._declared_dep_paths: dict[str, tuple[str, tuple[str, ...]]] = {}

        # Deep seam over the registries above: folds source/dependency/
        # helper state into the cache key's ``state_hash`` segment. Borrows
        # the registry dicts by reference so later registrations are seen.
        self._state_hasher = DependencyStateHasher(
            functions=self.functions,
            data_sources=self.data_sources,
            source_hashes=self.source_hashes,
            purity_reports=self._purity_reports,
            graph=self.graph,
            helper_resolver=SysModulesHelperResolver(self._hash_helper_identity),
            declared_dep_snapshots=self._declared_dep_snapshots,
            declared_dep_resolver=self._resolve_declared_dep_hash,
        )

        # The same live re-resolution, for functions found inside data globals
        # (`_data_callable_identity`).
        self._data_helper_resolver = SysModulesHelperResolver(self._hash_helper_identity)

        atexit.register(self._exit_work.run)

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
            self._exit_work.backend = self._backend
            return self._backend

    @backend.setter
    def backend(self, value: CacheBackend) -> None:
        """Allow direct assignment (e.g. ``c.backend = MyBackend()``)."""
        self._backend = value
        self._exit_work.backend = value

    @property
    def backend_if_built(self) -> CacheBackend | None:
        """The backend if one has been built, else ``None``; never builds one."""
        return self._backend

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
        if overrides.get("debug") or overrides.get("verbose"):
            _log.enable(logging.DEBUG if self.debug else logging.INFO)

    def __repr__(self) -> str:
        backend_name = type(self._backend).__name__ if self._backend is not None else "<deferred>"
        n_funcs = len(self.functions)
        return f"Cash(backend={backend_name}, functions={n_funcs}, debug={self.debug})"

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
        allow_random: bool = ...,
        frozen: bool = ...,
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
        chunk_max_items: int = CHUNK_MAX_ITEMS,
        chunk_max_bytes: int = CHUNK_MAX_BYTES,
        strict: bool = False,
        assume_safe: bool = False,
        allow_random: bool = False,
        frozen: bool = False,
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
            file_depends_on: File path(s) to track as dependencies, as if the
                function had read them: their content is recorded with the
                entry and checked on every lookup, the way an automatically
                tracked read is, so an edit recomputes and a ``touch`` does not.
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
            allow_random: When ``True``, suppress the one-shot
                `CashRandomnessWarning` raised at decoration time
                if the function's source draws from an unseeded RNG
                (``np.random.randn()``, ``random.random()``,
                ``np.random.default_rng()`` with no seed, ...).
                The decorator-path counterpart of the notebook's
                ``# @cash:allow-random``; that comment is also
                honoured when it appears in the decorated
                function's own source. Suppresses only the
                *warning* - it does not change whether the result
                is cached, and the first call's value is still
                frozen and replayed. Seeding the RNG silences the
                warning on its own, because a seeded draw is
                reproducible.
            frozen: When ``True``, declare that the result is not
                modified after it is returned. A cached function
                receiving it as an argument then keys it by this
                call's identity instead of hashing its contents: no
                hash per call, the same key in every process, and it
                works for objects that cannot be pickled. A numpy
                result is returned read-only. Other objects are
                audited -- re-hashed at an occasional use, every use
                under ``CASH_DEBUG`` -- and a change warns
                KEY-FROZEN-MUTATED and falls back to content hashing
                for that object.

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
            self.get_func_key(func),
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
        func_name = self._register_func(cf, depends_on)
        self._pin_own_source(func, self.source_hashes[func_name])
        # A downstream that depends on this function inherits its TTL
        # (effective TTL = min over the dependency closure).
        self._effective_ttl_cache.clear()

        # Async generators are not cached; warn once and return unwrapped.
        if inspect.isasyncgenfunction(func):
            self._warn_once(
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
        self._warn_unseeded_randomness(func, func_name, allow_random)

        # Watch reads from now, not from the first miss: a memo the cached
        # function will use is usually filled before it is first called
        # (`main()` logging its settings), and a read nobody saw is an input
        # no entry records. Not when caching is off: that promises
        # nothing is patched or analysed.
        if not self.config.disable:
            try:
                install_read_watch()
            except Exception:  # noqa: BLE001 - the first miss installs them anyway
                logger.debug("[CORE] could not install the read watch at decoration", exc_info=True)

        return self._wrap_with_stats(cf, self._make_wrapper(cf))

    def _register_func(self, cf: CachedFunction, depends_on: list[Callable[..., Any] | DataSource] | None) -> str:
        """Register *cf* in the cache graph and return its key."""
        func, func_name = cf.func, cf.name
        previous = self._cached.get(func_name)
        if previous is not None:
            cf.carry_over(previous)
        self._cached[func_name] = cf
        self.functions[func_name] = func
        new_hash = callable_identity(func)
        old_hash = self.source_hashes.get(func_name)
        if old_hash and old_hash != new_hash:
            self._analyzed.discard(func_name)
            self._populated.discard(func_name)
        self.source_hashes[func_name] = new_hash
        self.graph.add_node(func_name)
        self._register_static_dependencies(func_name, depends_on)
        return func_name

    # -- why a call missed ---------------------------------------------------
    #
    # Why a call recomputed. The reasons below are decided where the lookup fails, from what that
    # lookup saw plus what this process remembers about the key -- never by
    # re-deriving the key, which would cost every call to explain a few.

    def _make_wrapper(self, spec: CachedFunction) -> Callable:
        """Build and return the caching wrapper for *func*, sync or async.

        One wrapper for both: everything before and after the body is the
        same sync code (`_lookup`, `_body_scope`, `_finish_miss`), and the
        two variants differ only in whether they await the body. The sync and
        async wrappers used to be two ~150-line copies, and they had drifted.
        """
        func = spec.func

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call = self._lookup(spec, args, kwargs, async_body=True)
                if call.outcome is not CACHE_MISS:
                    # A result the key path produced by calling `func` itself
                    # (no key) is a coroutine here: await it before handing back.
                    if inspect.iscoroutine(call.outcome):
                        return await call.outcome
                    return call.outcome

                async def compute() -> Any:
                    with self._body_scope(spec, call) as run:
                        run.res = await func(*args, **kwargs)
                    return self._finish_miss(spec, call, run)

                if not self.use_locking:
                    return await compute()
                return await self._single_flight(spec, call, compute)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = self._lookup(spec, args, kwargs, async_body=False)
            if call.outcome is not CACHE_MISS:
                return call.outcome

            def compute() -> Any:
                with self._body_scope(spec, call) as run:
                    run.res = func(*args, **kwargs)
                return self._finish_miss(spec, call, run)

            if self.use_locking:
                return self._compute_with_lock(spec, call, compute)
            return compute()

        return wrapper

    def _delete_backend_entries(self, func_name: str) -> None:
        """Delete all backend cache entries whose key starts with *func_name*."""
        try:
            prefix = f"{func_name}:"
            for entry in self.backend.list_entries():
                key = CacheMetadata.from_dict(entry).key or ""
                if key.startswith(prefix):
                    self.backend.delete(key)
        except (OSError, RuntimeError, KeyError):
            logger.debug("Failed to clear cache entries for %s", func_name)

    def _wrap_with_stats(self, cf: CachedFunction, wrapper: Callable) -> Callable:
        """Wrap *wrapper* with hit/miss stat tracking and attach introspection API.

        Dispatches on whether *func* is a coroutine function so the stats
        update (from the entry `_log_decorator_call` left in this call's
        `CALL_ENTRY` slot) happens AFTER the await for async, and
        synchronously otherwise.

        Attaches the introspection API:

        * ``cache_info()`` - hit/miss stats plus a rolling list of recent
          warnings emitted for this function.
        * ``cache_clear()`` - drop backend entries, reset stats, drop the
          warning log + dedup marks so re-warnings can fire.
        * ``explain(*args, **kwargs)`` - return a `CacheExplanation`
          for that specific call (sync, even on async wrappers).
        """
        func, func_name, allow_random = cf.func, cf.name, cf.allow_random
        # Read through `cf` by the end-of-run summary too. Not captured by
        # `stats_wrapper` itself: see `_bypass`.
        _stats = cf.stats

        def _count(slot: list) -> None:
            # The entry THIS call logged, if it logged one: a call that raised
            # before its lookup, or went through uncounted, leaves it empty.
            call = slot[0]
            if call is None:
                return
            if call["cache_hit"]:
                _stats["hits"] += 1
                _stats["total_time_saved"] += call.get("time_saved", 0.0)
                _stats["lookup_seconds"] += call.get("execution_time", 0.0)
            else:
                _stats["misses"] += 1
                _stats["miss_overhead_seconds"] += call.get("cash_seconds") or 0.0
                missed = call["miss_reason"]
                _stats["miss_reasons"][missed.kind] += 1
                if missed.changed:
                    _stats["changed"][missed.changed] += 1
                if call.get("not_stored"):
                    _stats["not_stored"][call["not_stored"]] += 1
                elif call.get("not_persisted"):
                    _stats["not_persisted"][call["not_persisted"]] += 1

        def _bypass(args: tuple, kwargs: dict) -> Any:
            # A helper, not inline: a caller that captures this wrapper in a
            # closure has the wrapper's own captures folded into ITS key, and
            # `_stats` read directly there is content-hashed -- so every call
            # moved the caller's key (tests/test_core/test_cold_process_key_stability).
            _stats["bypassed"] += 1
            return func(*args, **kwargs)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.config.disable:
                    return await _bypass(args, kwargs)
                token = ACTIVE_CONFIG.set(self.config)
                slot: list = [None]
                slot_token = CALL_ENTRY.set(slot)
                enter_cached_call()
                try:
                    result = await wrapper(*args, **kwargs)
                finally:
                    exit_cached_call()
                    CALL_ENTRY.reset(slot_token)
                    ACTIVE_CONFIG.reset(token)
                    _count(slot)
                self._warn_unseeded_estimator_result(func_name, result, allow_random)
                return result
        else:

            @functools.wraps(func)
            def stats_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Before anything else: disabled means the function, and
                # nothing of cash's -- no key, no analysis, no lookup, no store.
                if self.config.disable:
                    return _bypass(args, kwargs)
                # This instance's settings for the file checks the call makes
                # (`file_hash_full_max_bytes`); see ACTIVE_CONFIG.
                token = ACTIVE_CONFIG.set(self.config)
                slot: list = [None]
                slot_token = CALL_ENTRY.set(slot)
                enter_cached_call()
                try:
                    result = wrapper(*args, **kwargs)
                finally:
                    exit_cached_call()
                    CALL_ENTRY.reset(slot_token)
                    ACTIVE_CONFIG.reset(token)
                    _count(slot)
                self._warn_unseeded_estimator_result(func_name, result, allow_random)
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
                * ``miss_reasons`` (dict[str, int]) - the misses by why:
                  ``"no entry yet"``, ``"new arguments"``, ``"code or state
                  changed"``, ``"file changed"``, ``"ttl expired"``,
                  ``"not stored last time"`` and so on.
                * ``warnings`` (list[dict]) - rolling log of recent warning
                  emissions for this function. Each entry has ``category``,
                  ``message``, ``timestamp``. Capped at the last
                  ``WARNINGS_MAX`` (20) so it can't grow
                  unboundedly. Useful for spotting silent misbehavior
                  (cache_if predicate raised, lock failure, etc.) when
                  ``warnings.simplefilter`` swallowed the stderr emission.
            """
            total = _stats["hits"] + _stats["misses"]
            hit_rate = _stats["hits"] / total if total > 0 else 0.0
            with self._decorator_call_log_lock:
                warnings_log = list(cf.warnings)
            return {
                "hits": _stats["hits"],
                "misses": _stats["misses"],
                "hit_rate": hit_rate,
                "total_time_saved": _stats["total_time_saved"],
                "miss_reasons": {str(kind): n for kind, n in _stats["miss_reasons"].items()},
                "warnings": warnings_log,
            }

        def cache_clear() -> None:
            """Clear all cached results for this function.

            Removes all cache entries whose key starts with the function name.
            Resets hit/miss statistics, drops the per-function warnings log,
            and forgets ``_warn_once`` dedup marks for this function so the
            next misbehavior re-warns instead of being silently swallowed.
            """
            _stats.update(new_stats())
            self._delete_backend_entries(func_name)
            with self._decorator_call_log_lock:
                cf.warnings.clear()
                # Drop dedup marks for this function so future misbehavior
                # re-warns the user instead of staying silent.
                self._warning_keys_seen = {k for k in self._warning_keys_seen if k[1] != func_name}

        def explain(*args: Any, **kwargs: Any) -> CacheExplanation:
            """Return why the next call with these args would hit or miss.

            See `CacheExplanation` for the return shape. Inspection
            only - does not call the underlying function, mutate stats,
            or write to the backend. Safe to call from sync code even
            on async-wrapped functions.
            """
            token = ACTIVE_CONFIG.set(self.config)
            try:
                explanation = self._explain_call(cf, args, kwargs)
            finally:
                ACTIVE_CONFIG.reset(token)
            return dataclasses.replace(explanation, cache_dir=_backend_cache_dir(self.backend))

        stats_wrapper.cache_info = cache_info
        stats_wrapper.cache_clear = cache_clear
        stats_wrapper.explain = explain
        stats_wrapper.__wrapped__ = func
        # Marker so the purity analyzer treats a call to this wrapper as a
        # dependency-graph edge rather than recursing into cash's own wrapper
        # machinery. functools.wraps copies __module__, which would
        # otherwise make the wrapper look like same-package user code.
        stats_wrapper._cash_cached = True
        # Declared TTL, exposed so the notebook statement cache can see it. A
        # ``ttl=0`` function must recompute every call; without this the
        # statement ``x = f()`` gets cached with no TTL under %cash_on and
        # freezes the value the decorator promised to refresh.
        stats_wrapper._cash_declared_ttl = cf.ttl
        expose_script_function(func, stats_wrapper)
        cf.wrapper = stats_wrapper
        return stats_wrapper

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
        with self._decorator_call_log_lock:
            calls = list(self._decorator_call_log)
            self._decorator_call_log.clear()
        return calls

    def register_hasher(
        self,
        type_: type,
        hasher_fn: Callable[[Any], str],
        *,
        override: bool = False,
    ) -> None:
        """Register a custom hasher for a specific type.

        When ``_serialize_args`` encounters an argument of ``type_``, it will
        call ``hasher_fn(value)`` to produce a hash string instead of relying
        on ``pickle.dumps``.

        Args:
            type_: The Python type to register a hasher for.
            hasher_fn: A callable that takes a value of ``type_`` and returns
                a deterministic hash string.
            override: Take precedence over cash's own content hashers.
                Needed only for the types cash fingerprints itself -- numpy
                arrays, pandas / polars / PyArrow / modin frames, dask
                collections -- where those run first. A plain registration
                for one of those types is REJECTED with ``ValueError``: it
                could not have done anything, and saying so at setup beats
                leaving the user to discover that nothing got faster.

                Off by default because the built-ins read every byte, and a
                hasher that does not can return a *wrong* cached result
                rather than a slow one. Passing it says you accept that: what
                you return is the entire identity of the value, and two
                values sharing it share an entry. That is the right trade
                when you hold a version, a content id, or an immutable
                fingerprint the array itself does not carry -- and the wrong
                one for ``lambda a: a[0, 0]``.

                Overriding hashers are consulted before everything else,
                including a notebook value's lineage hash.

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
        src_hash = self._hash_callable_source(hasher_fn)
        # One type, one registration: re-registering must not leave the
        # previous entry behind in the other registry, still winning.
        self._type_hashers.pop(type_, None)
        self._override_hashers.pop(type_, None)
        if override:
            self._override_hashers[type_] = (hasher_fn, src_hash)
        else:
            self._type_hashers[type_] = (hasher_fn, src_hash)
        # A memoized hash was produced by whichever hasher was in effect
        # before this call; drop them so the new registration is not shadowed
        # for objects already seen.
        self._arg_hash_memo.clear()
        self._frame_memo.clear()

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

                return ttl_expired(timestamp, metadata.ttl, now)
            except (AttributeError, TypeError, ValueError):
                return True

        return self.backend.cleanup_expired(is_expired)

    def explorer(self) -> CacheExplorer:
        """Return a `CacheExplorer` instance for interactive cache browsing."""
        # Local: the explorer imports pandas, which `import cash` must not load.
        from .ui.explorer import CacheExplorer

        return CacheExplorer(self)

    def run_summary(self) -> str:
        """A per-function hit/miss table for this process, or ``""``.

        Empty when no cached function was ever called, so a caller can print
        this unconditionally without emitting a header over nothing.
        """
        stats = [(name, cf.stats) for name, cf in self._cached.items()]
        rows = [(name, s) for name, s in stats if s["hits"] or s["misses"]]
        bypassed = sum(s.get("bypassed", 0) for _, s in stats)
        disabled_line = (
            f"cash: caching disabled (disable=True / CASH_DISABLE) -- {bypassed} "
            f"call{'' if bypassed == 1 else 's'} ran uncached"
            if bypassed
            else ""
        )
        if not rows:
            return disabled_line
        rows.sort(key=lambda r: r[1]["total_time_saved"], reverse=True)

        hits = sum(s["hits"] for _, s in rows)
        calls = hits + sum(s["misses"] for _, s in rows)
        saved = sum(s["total_time_saved"] for _, s in rows)
        # What cash cost: the hits' lookups AND the misses' keys, checks and
        # stores. Counting the lookups alone reported a 24 s loss as 14 s, and
        # a function that never hit as costing nothing.
        spent = sum(s.get("lookup_seconds", 0.0) + s.get("miss_overhead_seconds", 0.0) for _, s in rows)
        width = min(44, max(len(name) for name, _ in rows))

        def _fit(name: str) -> str:
            # Keep the TAIL. A name is ``module.qualname``, so for anything
            # nested -- a closure, a method, a test helper -- the part that
            # identifies it is at the end, and truncating from the right threw
            # away the function name and kept the package path.
            return name if len(name) <= width else "..." + name[-(width - 3) :]

        head = f"cash: {hits} of {calls} calls restored, {saved:.1f}s saved"
        if spent >= 0.1 and spent >= 0.1 * saved:
            # Saved is the compute the hits stood in for; cash itself cost
            # this much. Left out, a run that got 9x SLOWER read as a win.
            head += f" -- and {spent:.1f}s spent by cash on keys, lookups and stores"
            if spent > saved:
                head += f", a net loss of {spent - saved:.1f}s"
        lines = [head]
        where = self._summary_cache_dir()
        if where:
            # Which directory this ran against. A script user has no badge and
            # no other place to see it, and every question that starts "why is
            # nothing cached" is answered or excluded by this one line: a
            # scheduled job's cwd-relative cache, a path typed with one
            # backslash too few, a container volume that is not the one they
            # meant.
            lines.append(f"  cache: {where}")
        for name, stat in rows:
            # Pad the whole "N hits," token, not the word: padding the word
            # puts the space before the comma ("1 hit ,").
            hit_col = f"{stat['hits']} {'hit' if stat['hits'] == 1 else 'hits'},"
            miss_col = f"{stat['misses']} {'miss' if stat['misses'] == 1 else 'misses'}"
            saved_col = f"{stat['total_time_saved']:.1f}s saved" if stat["total_time_saved"] else "-"
            cost = stat.get("lookup_seconds", 0.0) + stat.get("miss_overhead_seconds", 0.0)
            if cost >= 0.1 and cost >= 0.1 * stat["total_time_saved"]:
                saved_col += f", {cost:.1f}s spent by cash"
            lines.append(f"  {_fit(name):<{width}}  {hit_col:<10}{miss_col:<12}{saved_col}")
            lines.extend(self._summary_reasons(stat))
        if disabled_line:
            lines.append("  " + disabled_line)
        return "\n".join(lines)

    @staticmethod
    def _summary_reasons(stat: dict[str, Any]) -> list[str]:
        """The indented lines under a summary row: why it missed, what stayed.

        "1 miss" was the whole story before, and it hid the common surprise:
        a result computed in 0.05 s is never written to disk, so every new
        process misses it. The run that CAUSES that is the one that can say so.
        """
        out = []
        reasons = stat.get("miss_reasons") or {}
        if reasons:
            out.append(
                "      missed: " + ", ".join(f"{n} {kind}" for kind, n in sorted(reasons.items(), key=lambda r: -r[1]))
            )
        for what, n in (stat.get("changed") or {}).items():
            out.append(f"      {MissKind.CODE} ({n}x): {what}")
        for why, n in (stat.get("not_stored") or {}).items():
            out.append(f"      not stored ({n}x): {why}")
        for why, n in (stat.get("not_persisted") or {}).items():
            out.append(f"      kept in RAM only ({n}x): {why}; a new process recomputes it")
        return out

    def _summary_cache_dir(self) -> str | None:
        """The cache directory this instance is using, for the summary header.

        Reads the ALREADY-BUILT backend when there is one and falls back to the
        configured path otherwise: the summary must never be the thing that
        creates a cache directory, and it runs from an ``atexit`` handler where
        building one is worse than saying nothing.
        """
        path = self._backend.local_dir if self._backend is not None else None
        if path:
            return path
        configured = getattr(self.config, "cache_dir", None)
        return configured if isinstance(configured, str) and configured else None

    def _print_run_summary(self) -> None:
        """``atexit`` hook for ``summary=True``. Must never raise.

        Interpreter shutdown tears modules down underneath handlers, so a
        diagnostic that explodes here would turn a finished run into a
        traceback the user cannot act on.
        """
        try:
            text = self.run_summary()
            if text:
                if in_multiprocessing_child():
                    # One table per worker process: say whose it is.
                    text = text.replace("cash:", f"cash (pid {os.getpid()}):", 1)
                # stderr: stdout is the program's output -- a report, a pipe, a
                # JSON response -- and a summary landing in it broke all three.
                # ONE write: pool workers exiting together
                # interleaved print()'s separate writes mid-line.
                # Into the application's log when it will print it: a service
                # whose output goes through dictConfig never saw the summary.
                # Otherwise to stderr -- once: both, with cash's own handler
                # passing it on as well, printed it three times.
                # Through the application's handlers whenever it has one that
                # takes INFO -- past the level filters, as CASH_DEBUG's lines
                # are: CASH_SUMMARY asked for it. Gated on the levels, the
                # block came through the app's formatter at INFO and raw at
                # WARNING, two shapes for a log shipper to parse.
                cash_logger = logging.getLogger("cash")
                if any(h.level <= logging.INFO for h in _log.application_handlers(cash_logger)):
                    summary_logger = logging.getLogger("cash.summary")
                    summary_logger.handle(
                        summary_logger.makeRecord(
                            summary_logger.name, logging.INFO, "(cash summary)", 0, "%s", (text,), None
                        )
                    )
                else:
                    sys.stderr.write(text + "\n")
                    sys.stderr.flush()
        except Exception:  # noqa: BLE001 - a summary must not fail a finished run
            pass

    def show_stats(self) -> None:
        """Display the interactive analytics dashboard.

        Requires IPython/Jupyter and ipywidgets. In script environments,
        prints the same per-function table ``summary=True`` prints at exit.
        """

        # Local: the dashboard imports matplotlib, which `import cash` must not load.
        from .ui.dashboard import HAS_WIDGETS, show_analytics_dashboard

        # Asking the dashboard whether it CAN run, rather than calling it and
        # catching. It prints "ipywidgets is required" and returns normally, so
        # the except-ImportError fallback this replaces was unreachable: the
        # documented script behaviour never once happened.
        if HAS_WIDGETS:
            try:
                show_analytics_dashboard()
                return
            except (ImportError, RuntimeError):
                pass
        text = self.run_summary()
        print(text if text else "cash: no cached function has been called yet.")

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
        # Local: import cycle core -> notebook.ipython -> notebook.ipython.magics -> core.
        from .notebook.ipython.magics import CashMagics

        magics = CashMagics(ip, self)
        ip.register_magics(magics)

    def clear_all(self) -> None:
        """Clear cached results for every function registered with this instance.

        Equivalent to calling ``f.cache_clear()`` on every ``@cash.cache``-decorated
        function. Resets hit/miss statistics and removes all backend entries.
        """
        for cf in list(self._cached.values()):
            if cf.wrapper is not None:
                cf.wrapper.cache_clear()

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

        file_registry().register(module_name, func_name, handler_factory)

    def shutdown(self) -> None:
        """Finish up: warn the run's CACHE-NET-LOSS verdicts, flush the
        stored-key record and wait for the backend's writes.

        Runs at exit on its own. Never builds a backend: at interpreter
        teardown building one would start threads that can no longer
        register, and an unbuilt backend has nothing to drain.
        """
        self._exit_work.run()
