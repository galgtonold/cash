"""Main Cash class - decorator-based caching with automatic dependency tracking.

Provides the `Cash` entry point for ``@cash.cache`` function-level
caching. The notebook path lives in `cash.notebook`; it reads the decorator's
per-call events through `Cash.drain_decorator_calls`.
"""

from __future__ import annotations

import ast
import atexit
import concurrent.futures
import contextlib
import dataclasses
import dis
import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import sys
import textwrap
import threading
import time
import types
import weakref
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterator, Sized
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from . import _log, _plain_data
from ._clock import perf_counter as _perf_counter
from ._paths import MAIN_MODULE_NAMES, normalize_path, resolve_main_module
from .analysis.annotations import parse_annotation_line
from .analysis.cacheability_decision import identity_coupled_reason
from .analysis.code_analyzer import CodeAnalyzer
from .backends import CacheBackend, CacheMetadata
from .backends._base import ttl_expired
from .backends._writes import in_multiprocessing_child
from .backends.factory import build_backend_from_config, build_tiered
from .backends.file_backend import recreate_cache_dir
from .backends.serialization import get_serializer
from .config import CashConfig, get_config
from .data_source import DataSource, state_token_of
from .decorator.arg_hashing import (
    ARG_COST,
    CODE_ARG_FIX,
    CODE_VALUE_TYPES,
    LINEAGE_SRC_DECORATOR,
    LINEAGE_SRC_FROZEN,
    LINEAGE_SRC_STATEMENT,
    NO_SUSPECT,
    PLAIN_CENSUS,
    is_cow_pandas,
    plain_key_part,
    unhashable_arg_fix,
)
from .decorator.call_state import (
    CACHE_MISS,
    CALL_ENTRY,
    CAPTURE_WATCH,
    NESTED_CASH_SECONDS,
    NO_WATCH,
    PROCESS_STARTED,
    THREADS_IN_CALLS,
    BodyRun,
    BuiltKey,
    Call,
    CallSpec,
    KeyBuildFailed,
    UnhashableArgs,
    UnhashableDefault,
    enter_cached_call,
    exit_cached_call,
    run_to_completion,
)
from .decorator.closure_fold import ClosureFoldMixin
from .decorator.code_args import CodeArgsMixin
from .decorator.code_identity import (
    CODE_KEYED_STATS,
    CodeIdentityMixin,
)
from .decorator.explain import (
    EXPLAIN_DISABLED,
    EXPLAIN_FILE_CHANGED,
    EXPLAIN_HIT,
    EXPLAIN_KEY_UNCOMPUTABLE,
    EXPLAIN_NO_ENTRY,
    EXPLAIN_TTL_EXPIRED,
    MISS_ARGS,
    MISS_CODE,
    MISS_DYNAMIC,
    MISS_FILE,
    MISS_FIRST,
    MISS_GONE,
    MISS_INCOMPLETE,
    MISS_KEY_FAILED,
    MISS_MOCKED,
    MISS_NOT_STORED,
    MISS_RAISED,
    MISS_TTL,
    MISS_UNHASHABLE,
    STALE_REASON_TEXT,
    STORE_OUTCOMES_MAX,
    WHAT_CHANGED,
    CacheExplanation,
    describe_file_deps,
    describe_state_change,
    entry_id_of,
    is_sampled_dep,
    same_file_key,
)
from .decorator.frozen import FROZEN_AUDIT_EVERY, FROZEN_AUDIT_FIRST
from .decorator.globals_fold import (
    GlobalsFoldMixin,
)
from .decorator.iterators import ChunkedCachedIterator, StreamingCachedIterator, is_one_shot_iterator
from .decorator.purity_checks import (
    format_issues_summary,
    is_mutable,
    make_opaque_issue,
    shares_memory,
    static_effect_kinds,
)
from .decorator.reporting import calls_logger
from .decorator.rng import read_seed, seed_parameters
from .decorator.script_pickling import expose_script_function
from .decorator.store import STORE_FAILED_FIX, UNTAGGABLE_TYPES
from .dependency_state import (
    EXPLAINING as _EXPLAINING,
)
from .dependency_state import (
    STATE_LEDGER,
    DependencyStateHasher,
    SysModulesHelperResolver,
    ledger_note,
)
from .diagnostics import (
    format_diagnostic,
    warn_diagnostic,
    warn_diagnostic_message,
)
from .effect_observer import EffectObserver
from .effectiveness import EffectivenessLedger
from .exceptions import (
    SOURCE_RETRIEVAL_ERRORS,
    CacheBackendError,
    CacheExpiredError,
    CashCacheIneffectiveWarning,
    CashCacheStoreFailedWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from .graph import DependencyGraph
from .lineage_tag import own_tag
from .object_hashing import builtin_hash, builtin_hash_family, estimate_object_size, stable_key_repr
from .purity_analyzer import (
    ISSUE_AMBIENT_READ,
    ISSUE_MUTABLE_GLOBAL,
    ISSUE_NETWORK_READ,
    ISSUE_UNTRACKABLE_DEP,
    PurityReport,
    bindings_changed,
    get_analyzer,
    resolve_binding,
)
from .reconfigure import apply_overrides
from .remote_source import measured_validation, validation_is_expensive, warn_validation_cost_once
from .source_norm import (
    callable_identity,
    own_source_digest,
)
from .tracking.file_dep_snapshot import (
    ACTIVE_CONFIG,
    attach_code_relative,
    dep_is_fresh,
    dep_path_for_this_process,
    snapshot_dependencies,
    snapshot_is_fresh,
)
from .tracking.file_tracker import (
    FileAccessTracker,
    active_tracker,
    credited_reads,
    file_registry,
    install_read_watch,
    untracked,
)
from .tracking.randomness import (
    CashRandomnessWarning,
    RandomnessDetector,
    capture_rng_state,
    describe_random_call,
    restore_rng_state,
    rng_modules_changed,
    seed_epoch_component,
    seed_epochs,
)
from .value_types import (
    BUILTIN_CONTAINERS,
    CODELESS_PRIMS,
    IMMUTABLE_PRIMS,
    IMMUTABLE_VALUE_TYPES,
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


#: How many call events `Cash._decorator_call_log` holds. The notebook drains it
#: after every statement; nothing drains it in a script or a service, so it
#: keeps only the most recent calls rather than one entry per call forever.
_CALL_LOG_MAX = 10_000


class Cash(CodeIdentityMixin, CodeArgsMixin, ClosureFoldMixin, GlobalsFoldMixin):
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

        self._backend_lock = threading.Lock()

        self.graph = DependencyGraph()
        self.functions: dict[str, Callable[..., Any]] = {}  # Registry of cached functions
        self.data_sources: dict[str, DataSource] = {}  # Registry of data sources
        # func_name -> ``(as written, absolute)`` for each path its
        # ``file_depends_on=`` names. Each miss records the absolute paths on
        # the call's file tracker, as if the body had read them
        # (`_track_declared_files`); the paths as written are in the key
        # (`_fold_declared_files`).
        self._declared_files: dict[str, tuple[tuple[str, str], ...]] = {}
        self.source_hashes: dict[str, str] = {}  # Current source hashes
        # Session-scoped memo: id(arg) -> (weakref, lineage_hash, content_hash).
        # Lets a repeated ``@cash.cache`` call with the SAME unmutated argument
        # skip re-hashing a possibly-huge input. See ``_hash_arg_payload`` for
        # the read-side validation (weakref identity + lineage). Bounded below.
        self._arg_hash_memo: dict[int, tuple] = {}
        # id(frame) -> (weakref, shallow copy, signature, content hash): the
        # pandas copy-on-write memo, see `_frame_memo_store`.
        self._frame_memo: dict[int, tuple] = {}
        # Functions decorated frozen=True, by key.
        self._frozen_funcs: set[str] = set()
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
        # func_name -> (parameter, type, seconds, producer, pandas without
        # copy-on-write): the costliest argument to hash, for CACHE-NET-LOSS.
        self._arg_costs: dict[str, tuple] = {}
        # Running account of what caching cost vs what it saved, per function.
        # The decorator always caches by design -- this only ever informs.
        self._effectiveness = EffectivenessLedger()
        if self.config.summary:
            # Registered per instance rather than once per process: two Cash
            # instances are two independent caches, and each should account for
            # itself. ``run_summary`` returns "" when nothing was called, so an
            # unused instance prints nothing. (``atexit`` is imported at module
            # level; a local import here would shadow it for the whole method,
            # including the ``atexit.register(self.shutdown)`` further down.)
            atexit.register(self._print_run_summary)
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
        # surfaced). Keeps the cache key stable from the first call (finding #7).
        self._populated: set[str] = set()
        self._func_ttls: dict[str, int | None] = {}  # func_name -> declared ttl
        self._effective_ttl_cache: dict[str, int | None] = {}
        self._deref_writes: dict = {}  # code object -> frozenset of reassigned freevars
        # id(func) -> decoration-pinned own-source identity. The
        # wrapper closure keeps *func* alive, so the id stays valid for the
        # wrapper's lifetime.
        self._own_pins: dict[int, str] = {}
        # Pins taken at decoration whose file has not yet been compared with
        # the loaded code; the first call does it once (see _pin_own_source).
        self._own_pins_unverified: set[int] = set()
        # code object -> global names its decorator expressions read
        self._decorator_names_cache: dict = {}
        # code object -> frozenset of free vars with capture-unsafe uses
        self._capture_use_cache: dict = {}
        # code object -> tuple of global names it reads (global folding)
        self._global_read_cache: dict = {}
        # code object -> names folded only provisionally (CAS-270). See
        # `_read_global_data_names`. A missing entry means "unknown", which
        # `_fold_read_globals` treats as "watch everything".
        self._provisional_global_cache: dict = {}
        # (code object, scope) -> names a call was OBSERVED to mutate. Learned
        # once, then those names stop being folded (see `_learn_mutating_captures`).
        self._mutating_globals: dict = {}
        # code object -> closure free vars folded only provisionally (CAS-270).
        self._provisional_capture_cache: dict = {}
        # func_name -> RNG modules that function was OBSERVED drawing from.
        # Learned on a miss; only these functions get a seed-epoch in their key.
        self._rng_drawing_funcs: dict[str, set[str]] = {}
        # (module_global, attribute) read pairs per code object; see
        # _read_module_attr_pairs.
        self._module_attr_cache: dict = {}
        self._local_binding_cache: dict[Any, tuple | None] = {}
        self._carrier_verdicts: dict[int, tuple[Any, bool]] = {}
        self._stored_doc_memo: dict[str, tuple[tuple[int, int], dict]] = {}
        # (func_name, state segment) -> the ledger of the key build that first
        # produced it (`_keep_state_ledger`).
        self._state_ledgers: dict[tuple[str, str], dict] = {}
        self._ram_only_pending: dict[str, dict[str, list]] = {}
        # func_name -> {digest: when} of warnings shown, not yet in the record
        # (`_first_showing`); written with the record's next write.
        self._warned_pending: dict[str, dict[str, float]] = {}
        self._ram_only_lock = threading.Lock()
        # Serialises this process's reads and rewrites of the stored-key
        # record: on Windows a read that overlaps the rewrite's rename fails,
        # and a pool's misses read an empty record as "new arguments".
        self._stored_doc_lock = threading.RLock()
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
        # func_name -> (function the signature was read from, inspect.Signature
        # or None if introspection failed). Used to bind call arguments to a
        # canonical form so that logically-identical calls written differently
        # (positional vs keyword, omitted vs explicit default, kwargs in
        # different orders) share one cache key.
        #
        # The function is stored alongside so the memo can be invalidated when
        # the name is REBOUND to a new function object (a notebook cell re-run).
        # Keying by name alone pinned the first signature forever, so
        # `apply_defaults()` kept folding a default the callee no longer has
        # .
        self._signatures: dict[str, tuple[Callable | None, inspect.Signature | None]] = {}
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
        # func_name -> {parameter: seeding call} for seeding calls fed by a
        # parameter; checked per call by `_warn_if_seed_is_none`.
        self._seed_params: dict[str, dict[str, tuple]] = {}
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
        # All three are in-process memory only, and bounded: they explain,
        # they never decide anything.
        #: func_name -> the last cache key it looked up.
        self._last_key: dict[str, str] = {}
        #: cache_key -> what happened when it was last computed here.
        self._store_outcomes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        #: cache_key -> (kind, detail) for a lookup that just missed, taken
        #: by the `_log_decorator_call` that reports it.
        self._pending_miss: dict[str, tuple[str, str]] = {}

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
        # func_name -> the live stats dict of its wrapper. Populated by
        # ``_wrap_with_stats``; read by the end-of-run summary.
        self._function_stats: dict[str, dict[str, Any]] = {}
        self._type_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}
        # Same shape, but consulted BEFORE cash's own content hashers rather
        # than after them. Separate registry rather than a flag in the tuple
        # above so the hot path can skip the whole question with one empty
        # check -- overriding is rare, and every cached call pays for this.
        self._override_hashers: dict[type, tuple[Callable[[Any], str], str]] = {}

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
        # Functions the STATIC pass already reported on. The runtime effect
        # observer stays quiet for these: it would be a second warning about
        # the same function, and the user has already been told.
        self._purity_static_flagged: set[str] = set()
        # Functions whose arguments cost too much to re-hash for the
        # mutation check. Checked once, then left alone.
        self._mutation_check_too_costly: set[str] = set()
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
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
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

        func_name = self._register_func(func, depends_on, file_depends_on)
        self._pin_own_source(func, self.source_hashes[func_name])
        if frozen:
            self._frozen_funcs.add(func_name)
        else:
            self._frozen_funcs.discard(func_name)
        self._purity_modes[func_name] = "strict" if strict else "silent" if assume_safe else "warn"
        # Record the declared TTL so a downstream that depends on this function
        # can inherit it (effective TTL = min over the dependency closure).
        self._func_ttls[func_name] = ttl
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
        # no entry records (round 20). Not when caching is off: that promises
        # nothing is patched or analysed.
        if not self.config.disable:
            try:
                install_read_watch()
            except Exception:  # noqa: BLE001 - the first miss installs them anyway
                logger.debug("[CORE] could not install the read watch at decoration", exc_info=True)

        wrapper = self._make_wrapper(
            func,
            func_name,
            dynamic_depends_on,
            ttl,
            cache_if,
            chunk_max_items,
            chunk_max_bytes,
        )
        return self._wrap_with_stats(
            func,
            func_name,
            wrapper,
            dynamic_depends_on=dynamic_depends_on,
            ttl=ttl,
            allow_random=allow_random,
        )

    def _register_func(
        self,
        func: Callable,
        depends_on: list[Callable[..., Any] | DataSource] | None,
        file_depends_on: str | list[str] | None,
    ) -> str:
        """Register a function in the cache graph and return its key."""
        func_name = self.get_func_key(func)
        self.functions[func_name] = func
        new_hash = callable_identity(func)
        old_hash = self.source_hashes.get(func_name)
        if old_hash and old_hash != new_hash:
            self._analyzed.discard(func_name)
            self._populated.discard(func_name)
        self.source_hashes[func_name] = new_hash
        self.graph.add_node(func_name)
        self._register_static_dependencies(func_name, depends_on)
        if file_depends_on:
            file_paths = [file_depends_on] if isinstance(file_depends_on, str) else file_depends_on
            self._declared_files[func_name] = tuple((str(p), os.path.abspath(p)) for p in file_paths)
        else:
            self._declared_files.pop(func_name, None)
        return func_name

    def _fold_declared_files(self, func_name: str, state_hash: str) -> str:
        """Fold which files ``file_depends_on=`` names, as written, into the key.

        Their content is checked against the entry on lookup
        (`_track_declared_files`); this is what makes adding, removing or
        re-pointing one a different key, since an entry that recorded file A
        would otherwise keep hitting after the declaration moved to file B.
        As written rather than absolute, so a relative path keys the same on
        every machine.
        """
        declared = self._declared_files.get(func_name)
        if not declared:
            return state_hash
        names = json.dumps(sorted(raw for raw, _ in declared))
        return hashlib.sha256(f"{state_hash}:files:{names}".encode()).hexdigest()

    def _track_declared_files(self, tracker: Any, func_name: str) -> None:
        """Record *func_name*'s ``file_depends_on=`` paths on *tracker*.

        As reads, so the entry snapshots their content and every lookup checks
        it with ``file_dep_is_fresh``, exactly like a file the body opened. They
        used to become a ``FileDataSource`` in the key, keyed on the mtime
        alone: a ``touch`` recomputed, and an edit that kept the mtime hit.
        Called inside the timed body, so the content hash is taken off the body
        time with the tracker's other read hashes.
        """

        for _, path in self._declared_files.get(func_name, ()):
            if os.path.exists(path):
                tracker.add_tracked(normalize_path(os.path.realpath(path)))
            else:
                tracker.add_tracked_absent(normalize_path(path))

    def _fold_rng_epoch(self, func_name: str, state_hash: str) -> str:
        """Fold the current seed epoch into the key, for RNG-drawing functions.

        A function that draws from the global stream has an input the key never
        saw. Change ``np.random.seed(12345)`` to ``seed(999)``, re-run, and the
        model trained under the old seed came straight back, silently, with a
        green badge -- on the exact idiom ``cash.help()`` rule 4 recommends.

        Deliberately narrow on three axes:

        * Only functions OBSERVED to draw (``_rng_drawing_funcs``), so seeding
          the stream does not invalidate functions that never read it.
        * Only the *epoch*, never the raw RNG state -- the state advances on
          every draw, so keying on it would miss forever.
        * Empty when the module is unseeded, so an unseeded sample keeps being
          replayed. That is the freeze contract, and it is what makes caching an
          expensive unseeded draw still worth it.

        The verdict is learned on a miss, so the call that first reveals the
        draw has already been stored under an epoch-free key; the next call
        recomputes once and is stable from then on.
        """
        modules = self._rng_drawing_funcs.get(func_name)
        if modules is None:
            modules = self._load_rng_draw_marker(func_name)
        if not modules:
            return state_hash
        component = seed_epoch_component(modules)
        if not component:
            return state_hash
        return hashlib.sha256(f"{state_hash}{component}".encode("utf-8")).hexdigest()

    @staticmethod
    def _rng_marker_key(func_name: str) -> str:
        """Backend key for the "this function draws" verdict."""
        return f"cash:rngdraw:{func_name}"

    def _load_rng_draw_marker(self, func_name: str) -> set[str]:
        """Read the persisted draw verdict, caching the answer for this process.

        The verdict is learned by OBSERVING a call, so it lives in memory -- and
        a kernel restart or a fresh `python run.py` throws it away. That is fatal
        for the case this exists to fix: restart-and-run-all gets exactly one
        call per function, so an in-memory-only verdict is never applied and the
        stale value comes straight back.

        A tiny per-function marker survives the process and can be read BEFORE
        the real key is built, which the entry's own metadata cannot (that would
        need the key it is supposed to inform). One backend read per function per
        process; misses are remembered as empty so it is not retried.
        """
        cached = self._rng_drawing_funcs.get(func_name)
        if cached is not None:
            return cached
        modules: set[str] = set()
        try:
            stored = self.backend.get(self._rng_marker_key(func_name))
            # Backends answer with ``(metadata, value)``; unwrap before reading.
            # Treating the pair itself as the payload silently yielded an empty
            # set, so every restart re-learned nothing and the stale value came
            # back -- the whole point of persisting the marker.
            if isinstance(stored, tuple) and len(stored) == 2 and isinstance(stored[0], dict):
                stored = stored[1]
            if isinstance(stored, (set, frozenset, list, tuple)):
                modules = {m for m in stored if isinstance(m, str)}
        except Exception:  # noqa: BLE001 - a marker miss must never break a call
            modules = set()
        self._rng_drawing_funcs[func_name] = modules
        return modules

    def _store_rng_draw_marker(self, func_name: str, modules: set[str]) -> None:
        """Persist the verdict so the next process applies it on its first call."""
        try:
            self.backend.set(self._rng_marker_key(func_name), set(modules))
        except Exception:  # noqa: BLE001 - best effort; correctness degrades to today's
            logger.debug("could not persist RNG draw marker for %s", func_name)

    def _note_rng_draw(self, func_name: str, pre_state: dict | None) -> bool:
        """Record which global RNG modules *func_name* just advanced."""
        if pre_state is None:
            return False
        try:
            changed = rng_modules_changed(pre_state, capture_rng_state())
        except (TypeError, AttributeError):  # pragma: no cover
            return False
        # A module merely imported by the call is newly present rather than
        # advanced; only count streams that already existed.
        drew = {m for m in changed if m in pre_state}
        if not drew:
            return False
        known = self._rng_drawing_funcs.setdefault(func_name, set())
        newly = bool(drew - known)
        if newly:
            known.update(drew)
            self._store_rng_draw_marker(func_name, known)
        if not newly:
            return False
        # Only report "newly seen" -- which suppresses this call's write -- when a
        # drawn module is actually SEEDED. An unseeded draw has no epoch that can
        # change, so its frozen value is correct from the first call; skipping the
        # write there would redraw and break the freeze-from-first-call contract.
        return bool(drew & set(seed_epochs()))

    @staticmethod
    def _capture_rng_pre_state() -> dict | None:
        """Snapshot the global RNG streams, or None if unavailable."""
        try:
            return capture_rng_state()
        except (TypeError, AttributeError):  # pragma: no cover
            return None

    def _resolve_cache_key(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
        call_start: float,
    ) -> Any:
        """The key for a real call, or the call's result when it has none.

        `_build_key`, with a ledger of what the state segment is made of
        (`STATE_LEDGER`) and this key's `CAPTURE_WATCH`. Returns
        ``(resolved, capture_watch)``, where *resolved* is one of:

          - ``(cache_key, state_hash, args_hash)`` - the key was built
          - ``(CACHE_MISS, result, 'unkeyable')`` - a mocked helper, no code to key
          - ``(CACHE_MISS, result, 'unhashable')`` - an argument or default could not be hashed
          - ``(CACHE_MISS, result, 'error')`` - building the key raised

        In the last three, *result* is what ``func(*args, **kwargs)`` returned:
        the call already ran, uncached, was warned about once and logged. The
        body's own exceptions propagate.
        """
        # Outside the key build below, which turns any exception into "no
        # key": an exception from the user's own body must not be caught there
        # and the body run a second time.
        unkeyable = self._refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            return self._run_uncached(func, func_name, args, kwargs, call_start, "unkeyable", unkeyable), {}
        ledger: dict = {}
        ledger_token = STATE_LEDGER.set(ledger)
        watch: dict = {}
        watch_token = CAPTURE_WATCH.set(watch)
        failure: tuple[str, str] | None = None
        try:
            built = self._build_key(func, func_name, dynamic_depends_on, args, kwargs)
        except UnhashableDefault:
            # `_fold_defaults` has warned: an unhashable default means cash
            # cannot tell whether it changed, so caching at all risks a stale
            # result.
            failure = ("unhashable", "")
        except UnhashableArgs:
            self._warn_unhashable_args(func_name, args, kwargs)
            failure = ("unhashable", "")
        except KeyBuildFailed as e:
            self._warn_once(CashCacheIneffectiveWarning, func_name, e.code, e.message, code=e.code, fix=e.fix)
            failure = ("error", "")
        except Exception as e:  # noqa: BLE001 - any failure building the key means no key
            self._warn_key_build_failed(func_name, args, kwargs, e)
            failure = ("error", "")
        finally:
            CAPTURE_WATCH.reset(watch_token)
            STATE_LEDGER.reset(ledger_token)
        if failure is not None:
            return self._run_uncached(func, func_name, args, kwargs, call_start, *failure), watch
        if ledger:
            slot = (func_name, built.state_hash)
            if slot not in self._state_ledgers:
                self._keep_state_ledger(slot, ledger)
        return (built.cache_key, built.state_hash, built.args_hash), watch

    def _run_uncached(
        self,
        func: Callable,
        func_name: str,
        args: tuple,
        kwargs: dict,
        call_start: float,
        why: str,
        detail: str,
    ) -> tuple:
        """Run a call that has no key, log it as a miss, and hand back its result."""
        result = func(*args, **kwargs)
        self._log_decorator_call(
            func_name,
            cache_hit=False,
            execution_time=_perf_counter() - call_start,
            args_hash=why,
            cache_key="",
            miss_detail=detail,
        )
        return (CACHE_MISS, result, why)

    def _build_key(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> BuiltKey:
        """The cache key for calling *func* with these arguments.

        The ONE key build: a real call (`_resolve_cache_key`) and ``explain()``
        (`_explain_call`) both use it, so the key explain() predicts is the key
        the call looks up. It used to be written out twice, and the copy in
        explain() lacked the random-seed epoch and the class members of a
        method's arguments, so it reported ``no_entry`` for calls that hit.

        Raises when there is no key: `UnhashableDefault`, `UnhashableArgs`,
        `KeyBuildFailed`, or whatever else a step raised. Never keys the call
        without a part that failed. Warnings from the steps are silent while
        `_EXPLAINING` is set.
        """
        # One plain-data census per argument, shared across the key
        # (`plain_census`).
        previous = getattr(PLAIN_CENSUS, "memo", None)
        PLAIN_CENSUS.memo = {}
        try:
            # The state after each fold, in `_STATE_STAGES` order: when no
            # named part moved, the first stage whose output did is the one
            # that changed (`describe_state_change`).
            chain: list[str] = []
            ledger_note("@chain", chain)
            state_hash = self._state_hasher.compute(
                func_name,
                own_source_override=self._pin_own_source(func),
                note=True,
            )
            state_hash = self._fold_declared_files(func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._fold_closure(func, func_name, state_hash)
            chain.append(state_hash)
            folded_defaults = self._fold_defaults(func, func_name, state_hash)
            if folded_defaults is None:
                raise UnhashableDefault
            state_hash = folded_defaults
            chain.append(state_hash)
            state_hash = self._fold_bound_self(func, func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._fold_read_globals(func, func_name, state_hash)
            state_hash = self._fold_helper_read_globals(func, func_name, state_hash)
            state_hash = self._fold_dependency_read_globals(func, func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._fold_rng_epoch(func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._fold_environment(func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._fold_method_class_deps(func, args, state_hash)
            chain.append(state_hash)
            # ONE canonicalisation, fed to both the code channel and the value
            # channel. `_fold_code_args` on the RAW arguments saw a class
            # passed explicitly but not the identical class arriving as a
            # parameter DEFAULT, so `build()` and `build(Schema)` -- the same
            # logical call -- produced two cache keys and two executions.
            normalized_args = self._normalize_call_args(func_name, args, kwargs)
            if func_name in self._seed_params:
                self._warn_if_seed_is_none(func, func_name, args, kwargs)
            state_hash = self._fold_code_args(*normalized_args, state_hash, func_name=func_name)
            chain.append(state_hash)
            dynamic_state_hash = self._resolve_dynamic_dependencies(func_name, dynamic_depends_on, args, kwargs)
            args_hash = self._serialize_args(func_name, args, kwargs, normalized=normalized_args)
            self._note_arg_cost(func_name)
        finally:
            PLAIN_CENSUS.memo = previous
        if args_hash is None:
            raise UnhashableArgs
        cache_key = self._compute_cache_key(func_name, state_hash, dynamic_state_hash, args_hash)
        return BuiltKey(cache_key, state_hash, args_hash, normalized_args)

    def _warn_unhashable_args(self, func_name: str, args: tuple, kwargs: dict) -> None:
        """KEY-UNHASHABLE-ARG, naming the argument when one can be singled out."""
        arg_type_name = self._first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            which = (
                "an argument could not be hashed, and cash cannot say which -- the value is nested inside a container"
            )
            suggestion = (
                "find the nested value, then register a hasher for its "
                "type with cash.register_hasher(SomeType, ...) or pass "
                "something hashable in its place."
            )
        else:
            which = f"an argument of type {arg_type_name} could not be hashed"
            suggestion = unhashable_arg_fix(self._first_unhashable_arg(args, kwargs), arg_type_name)
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: {which}, so this call and every call like it does not cache.",
            code="KEY-UNHASHABLE-ARG",
            fix=suggestion,
        )

    def _warn_key_build_failed(self, func_name: str, args: tuple, kwargs: dict, e: Exception) -> None:
        """KEY-BUILD-FAILED: a step of the key build raised where it did not expect to."""
        arg_type_name = self._first_unhashable_arg_type(args, kwargs)
        if arg_type_name == "<unknown>":
            hint = (
                "check the function's arguments -- cash could not identify "
                "the offending type; if the exception does not belong to "
                "your code, report it as a bug with the traceback."
            )
        elif isinstance(self._first_unhashable_arg(args, kwargs), CODE_VALUE_TYPES):
            hint = CODE_ARG_FIX
        else:
            hint = (
                f"register a hasher with "
                f"cash.register_hasher({arg_type_name}, ...) if "
                f"{arg_type_name} is the unhashable argument."
            )
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            arg_type_name,
            f"@cash.cache on {func_name}: cache-key generation raised "
            f"{type(e).__name__} ({e}) somewhere it did not anticipate, so "
            f"this call does not cache.",
            code="KEY-BUILD-FAILED",
            fix=hint,
        )

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
        mutate the backend. The key comes from `_build_key`, the same
        build a real call uses, and the entry is judged by the rules
        `_try_get_cached` applies, so the answer reflects what would
        actually happen on the next real call.

        See `CacheExplanation` for the return shape.
        """
        if self.config.disable:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_DISABLED,
                func_name=func_name,
                details={"hint": "Caching is disabled (disable=True / CASH_DISABLE): every call runs the function."},
            )
        # Populate the dependency closure first so the state hash matches what
        # a real call computes (otherwise explain() reports a stale pre-analysis
        # key and a false `no_entry` - finding #7). This only fills internal
        # analysis caches; it does not warn, run the function, or touch the
        # backend.
        self._ensure_closure_analyzed(func)
        # Same binding check a real call makes first: a patched helper
        # changes the key, and a mock means the call would run uncached.
        unkeyable = self._refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": MISS_MOCKED,
                    "hint": f"{unkeyable}, which has no code to key, so the call would run uncached.",
                },
            )

        # The key a real call builds, built the same way, with every warning
        # a step would give held back: explain() must stay silent.
        token = _EXPLAINING.set(True)
        try:
            built = self._build_key(func, func_name, dynamic_depends_on, args, kwargs)
        except UnhashableDefault:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": "unhashable parameter default",
                    "hint": (
                        "A parameter default could not be hashed, so cash "
                        "cannot detect a change to it and will not cache "
                        "this call."
                    ),
                },
            )
        except UnhashableArgs:
            arg_type_name = self._first_unhashable_arg_type(args, kwargs)
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "arg_type": arg_type_name,
                    "hint": (
                        unhashable_arg_fix(self._first_unhashable_arg(args, kwargs), arg_type_name)
                        if arg_type_name != "<unknown>"
                        else "Could not identify the offending argument; likely a nested unpicklable value."
                    ),
                },
            )
        except KeyBuildFailed as e:
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={"error": e.code, "hint": f"{e.message} {e.fix}"},
            )
        except Exception as e:  # noqa: BLE001 - explain() reports, never raises
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_KEY_UNCOMPUTABLE,
                func_name=func_name,
                details={
                    "error": f"{type(e).__name__}: {e}",
                    "hint": "Building the cache key raised, so the call would run uncached.",
                },
            )
        finally:
            _EXPLAINING.reset(token)
        cache_key = built.cache_key
        frozen_args = self._frozen_arg_names(built.normalized_args)

        # Looking, not reading: `get` would count this as a use (USES / LAST
        # USED in `cash inspect`) and make the file backend rewrite the entry.
        raw_metadata = self.backend.peek_metadata(cache_key)
        if raw_metadata is not None and raw_metadata.get("metadata_only"):
            raw_metadata = None  # nothing to restore: a real call misses
        if raw_metadata is None:
            details = {
                "hint": ("No matching cache entry. First call with these arguments, or the cache was cleared."),
            }
            # A tracked dynamic dependency that changed produces a NEW cache key,
            # so the miss surfaces as no_entry rather than file_changed. Make the
            # explanation say so and list what's tracked (finding #8).
            dyn_ids = self._describe_dynamic_dependencies(dynamic_depends_on, args, kwargs)
            if dyn_ids:
                details["dynamic_dependencies"] = dyn_ids
                details["hint"] = (
                    "No matching cache entry. Either the first call with these "
                    "arguments, or a tracked dynamic dependency changed - a "
                    "dynamic_depends_on change yields a new cache key, so it "
                    "shows up here as no_entry, not file_changed. Tracked "
                    "dynamic dependencies: " + ", ".join(dyn_ids) + "."
                )
            # What this process knows about the key says more than "first call
            # or cleared": that it was never stored, why, or that it expired
            # under the ttl it was WRITTEN with -- which a backend drops on
            # read, so the entry looks absent (round 17).
            kind, why = self._absent_entry_reason(func_name, cache_key)
            if kind == MISS_TTL:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_TTL_EXPIRED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={"why": why},
                )
            details["why"] = f"{kind}: {why}"
            if kind != MISS_FIRST and "dynamic_dependencies" not in details:
                del details["hint"]  # the generic guess, now that we know
            if frozen_args:
                details["frozen_args"] = frozen_args
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_NO_ENTRY,
                func_name=func_name,
                cache_key=cache_key,
                details=details,
            )

        metadata = CacheMetadata.from_dict(raw_metadata)

        # TTL check - the same rule `_try_get_cached` applies.
        ttl = self._entry_ttl(ttl, metadata)
        if ttl_expired(metadata.timestamp, ttl):
            timestamp = metadata.timestamp or 0
            age = time.time() - timestamp
            return CacheExplanation(
                would_hit=False,
                reason=EXPLAIN_TTL_EXPIRED,
                func_name=func_name,
                cache_key=cache_key,
                details={
                    "ttl_seconds": ttl,
                    "age_seconds": age,
                    "cached_at": timestamp,
                },
            )

        # Auto-tracked file deps freshness. Routed through the SAME
        # content-authoritative helper a real lookup uses - comparing
        # raw mtime/size here made explain() report file_changed / 'mtime
        # changed' after a touch while the actual call hit. A diagnostic that
        # contradicts the behavior it describes is worse than none.
        if metadata.auto_file_deps:
            stale = self._stale_file_deps(metadata)
            if stale:
                return CacheExplanation(
                    would_hit=False,
                    reason=EXPLAIN_FILE_CHANGED,
                    func_name=func_name,
                    cache_key=cache_key,
                    details={"changed_files": stale, "file_deps": describe_file_deps(metadata.auto_file_deps)},
                )

        timestamp = metadata.timestamp or 0
        details = {
            "cached_at": timestamp,
            "cache_age_seconds": time.time() - timestamp if timestamp else None,
            "execution_time_saved": metadata.execution_time or 0.0,
        }
        if metadata.auto_file_deps:
            details["file_deps"] = describe_file_deps(metadata.auto_file_deps)
        if frozen_args:
            details["frozen_args"] = frozen_args
        return CacheExplanation(
            would_hit=True,
            reason=EXPLAIN_HIT,
            func_name=func_name,
            cache_key=cache_key,
            details=details,
        )

    def _frozen_arg_names(self, normalized_args: tuple[tuple, dict]) -> list[str]:
        """`explain()`'s list of arguments keyed by a frozen=True producer."""
        args, kwargs = normalized_args
        names = []
        for name, value in [*((f"#{i}", v) for i, v in enumerate(args)), *kwargs.items()]:
            if getattr(value, "_cash_lineage_src", None) == LINEAGE_SRC_FROZEN or (
                self._frozen_arrays and id(value) in self._frozen_arrays
            ):
                producer = (
                    getattr(value, "_cash_lineage_producer", None)
                    or (self._frozen_arrays.get(id(value), [None, None])[1])
                )
                names.append(f"{name} (the result of {producer}, declared frozen)")
        return names

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

    def _first_unhashable_arg_type(self, args: tuple, kwargs: dict) -> str:
        """Return the qualname of the argument that could not be hashed, or '<unknown>'.

        Used to attribute CashCacheIneffectiveWarning to a concrete type name
        so the user knows which register_hasher() call to add. See
        `_first_unhashable_arg` for how the argument is found.
        """
        suspect = self._first_unhashable_arg(args, kwargs)
        return "<unknown>" if suspect is NO_SUSPECT else type(suspect).__qualname__

    def _first_unhashable_arg(self, args: tuple, kwargs: dict) -> Any:
        """The argument that could not be hashed, or ``NO_SUSPECT``.

        Each candidate is hashed ALONE and the first that fails is named. It
        used to be simply the first argument of a non-built-in type, so
        ``score(df, lambda d: d * 2)`` blamed the DataFrame and advised a
        DataFrame hasher -- which cash rejects, and which with override=True
        would re-key every DataFrame function -- while the lambda was the
        culprit (round 18). This runs only on the failure path. Strings,
        numbers, None and built-in containers are skipped: a scalar always
        hashes, and a container holding the culprit is reported as "nested",
        which says more than naming the list. When no single candidate fails
        on its own, the first non-built-in is the best remaining guess.
        """
        candidates = [a for a in (*args, *kwargs.values()) if not isinstance(a, IMMUTABLE_PRIMS + BUILTIN_CONTAINERS)]
        for candidate in candidates:
            try:
                self._hash_arg_payload((candidate,), {})
            except Exception:  # noqa: BLE001 - exactly what we are looking for
                return candidate
        return candidates[0] if candidates else NO_SUSPECT

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
        """Return cached_data if valid, else CACHE_MISS sentinel.

        Key-presence is determined by ``metadata is not None`` - the
        backend contract is that absent keys return ``(None, None)``,
        so a non-None metadata view with a ``None`` data value still
        counts as a hit (a function that legitimately returned ``None``).

        Auto-tracked file dependencies stored in
        ``metadata.auto_file_deps`` are re-checked here; any file whose
        content differs from what was recorded forces a miss so the
        function re-reads the changed file.
        """
        if metadata is None:
            self._note_miss(func_name, cache_key, self._absent_entry_reason(func_name, cache_key))
            return CACHE_MISS
        ttl = self._entry_ttl(ttl, metadata)
        try:
            self._validate_ttl(metadata, ttl)
            if not self._auto_file_deps_fresh(metadata):
                self._note_miss(func_name, cache_key, (MISS_FILE, self._describe_stale_files(metadata)))
                return CACHE_MISS
            if not self._chunks_are_intact(cache_key, metadata):
                self._note_miss(func_name, cache_key, (MISS_INCOMPLETE, "a chunk of the stored result is missing"))
                return CACHE_MISS
            # If this hit happens *inside* another cached function's
            # computation, replay the files this entry depends on into the
            # enclosing tracker, so the outer function records them too.
            # Without this, a dependency that was already cached before the
            # consumer's first run hides its file deps behind a cache hit
            # and the consumer never invalidates when that file changes.
            self._propagate_file_deps_to_active_tracker(metadata)
            # Re-attach the lineage hash to the restored value. It's a plain
            # attribute that doesn't survive pickling, so a value restored
            # from disk would otherwise lose it - and a downstream cached
            # function would fall back to content-hashing under a DIFFERENT
            # key than when the upstream was freshly computed, recomputing
            # needlessly. The hash is deterministic from (cache_key,
            # auto_file_deps), both available here.
            self._attach_lineage(cached_data, cache_key, metadata.auto_file_deps, ttl=ttl, func_name=func_name)
            self._replay_rng_state(metadata)
            self._last_key[func_name] = cache_key
            self._log_decorator_call(
                func_name,
                cache_hit=True,
                execution_time=_perf_counter() - call_start,
                args_hash=args_hash,
                cache_key=cache_key,
                time_saved=(metadata.saves_seconds if metadata.saves_seconds is not None else metadata.execution_time)
                or 0.0,
                file_deps=metadata.auto_file_deps,
            )
            return cached_data
        except CacheExpiredError:
            age = time.time() - (metadata.timestamp or 0)
            self._note_miss(func_name, cache_key, (MISS_TTL, f"the entry is {age:.1f}s old and ttl={ttl}s"))
        except (TypeError, KeyError) as e:
            self._warn_metadata_invalid(func_name, e)
            self._note_miss(func_name, cache_key, (MISS_INCOMPLETE, "the stored entry's metadata did not validate"))
        return CACHE_MISS

    # -- why a call missed ---------------------------------------------------
    #
    # Round 17: four of five testers could not find out why a call recomputed.
    # The reasons below are decided where the lookup fails, from what that
    # lookup saw plus what this process remembers about the key -- never by
    # re-deriving the key, which would cost every call to explain a few.

    def _note_miss(self, func_name: str, cache_key: str, reason: tuple[str, str]) -> None:
        """Hold *reason* for the `_log_decorator_call` that reports this miss."""
        if len(self._pending_miss) > STORE_OUTCOMES_MAX:
            # Only a call that raised leaves one behind; never let those pile up.
            self._pending_miss.clear()
        self._pending_miss[cache_key] = reason
        self._last_key[func_name] = cache_key

    def _tier_default_ttl(self) -> int | None:
        """The ``default_ttl`` of the first tier that has one, as configured now."""
        return self._backend.default_ttl if self._backend is not None else None

    def _entry_ttl(self, ttl: int | None, metadata: Any) -> int | None:
        """The ttl a stored entry is judged by.

        The decorator's ``ttl=`` when it has one -- a per-function setting,
        applied as it stands now, in both directions. Otherwise the SHORTER of
        the ttl the entry was written with and the tier's ``default_ttl`` as
        configured now: lowering a tier's default from a day to 5 seconds left
        every entry written under the day being served (round 19, 3 of 3),
        while lowering a decorator's ttl took effect at once.
        """
        if ttl is not None:
            return ttl
        found = [t for t in (getattr(metadata, "ttl", None), self._tier_default_ttl()) if t is not None]
        return min(found) if found else None

    def _absent_entry_reason(self, func_name: str, cache_key: str) -> tuple[str, str]:
        """Why there is no entry for *cache_key*. Reads state; changes none.

        This process's own history first. With none -- the first call of a
        function in a fresh process, which is where a script's misses are --
        the keys earlier runs stored for this function, recorded beside the
        cache (`_record_stored_key`). Without them every such miss read "no
        earlier run left one on disk", including after a code edit and a TTL
        expiry, whose entries were in fact on disk (round 18, all five
        testers).
        """
        outcome = self._store_outcomes.get(cache_key)
        if outcome is not None:
            if outcome.get("not_stored"):
                return MISS_NOT_STORED, outcome["not_stored"]
            written_ttl = outcome.get("ttl")
            age = time.time() - outcome.get("stored_at", 0)
            if ttl_expired(outcome.get("stored_at", 0), written_ttl):
                return MISS_TTL, f"written {age:.1f}s ago with ttl={written_ttl}s"
            return MISS_GONE, ("stored earlier in this process and since evicted or cleared")
        previous = self._last_key.get(func_name)
        since = "since the last call"
        doc = self._stored_doc(func_name)
        record = doc["keys"]
        if cache_key in record:
            stored_at, written_ttl = record[cache_key][:2]
            age = time.time() - stored_at
            if ttl_expired(stored_at, written_ttl):
                return MISS_TTL, (f"stored {age:.0f}s ago by an earlier run, with ttl={written_ttl}s")
            return MISS_GONE, ("an earlier run stored it; it has since been evicted or cleared")
        if cache_key in doc["ram_only"]:
            why = doc["ram_only"][cache_key][1]
            return MISS_NOT_STORED, (
                f"an earlier run computed it but kept it in RAM only ({why}), so this process recomputed it"
            )
        # The same arguments stored under another state: the code or a value
        # it reads changed. Asked of the record BEFORE the call-to-call
        # comparison, which after a code edit blamed "new arguments" on every
        # call of a loop but the first (round 19).
        new_parts = cache_key.rsplit(":", 3)
        if len(new_parts) == 4:
            for key in reversed([*record, *doc["ram_only"]]):
                old_parts = key.rsplit(":", 3)
                if len(old_parts) == 4 and old_parts[2:] == new_parts[2:] and old_parts[1] != new_parts[1]:
                    return MISS_CODE, self._code_changed_detail(
                        func_name, old_parts[1], new_parts[1], doc, "since an earlier run stored it"
                    )
            # Earlier runs stored entries, and none under the state this
            # process computes: every one of them is out of date, whatever the
            # arguments. A changed DEFAULT moves the arguments too (they are
            # keyed with defaults applied), so the match above cannot see it,
            # and the call-to-call comparison below called 3 of 4 such misses
            # "new arguments" (round 20).
            earlier = {
                key: value
                for kind in ("keys", "ram_only")
                for key, value in doc[kind].items()
                if value and isinstance(value[0], (int, float)) and value[0] < PROCESS_STARTED
            }
            states = {key.rsplit(":", 3)[1] for key in earlier if key.count(":") >= 3}
            if states and new_parts[1] not in states:
                newest = max(earlier, key=lambda key: earlier[key][0])
                return MISS_CODE, self._code_changed_detail(
                    func_name,
                    newest.rsplit(":", 3)[1],
                    new_parts[1],
                    doc,
                    "since an earlier run stored its entries, so none of them applies",
                )
        if previous is None or previous == cache_key:
            others = [key for key in record if key != cache_key]
            if previous is None and others:
                previous = others[-1]
                since = "since an earlier run stored it"
            if previous is None or previous == cache_key:
                return MISS_FIRST, (
                    "the first call with these arguments in this process, and no earlier run stored one"
                )
        # Keys are `func:state:dynamic:args`; the parts that moved say why.
        old = previous.rsplit(":", 3)
        new = cache_key.rsplit(":", 3)
        if len(old) != 4 or len(new) != 4:
            return MISS_FIRST, "no entry for this key"
        moved = []
        what = None
        if old[1] != new[1]:
            moved.append((MISS_CODE, f"the function's code, a helper it calls, or a value it reads changed {since}"))
            what = self._what_changed(func_name, old[1], new[1], doc)
        if old[2] != new[2]:
            moved.append((MISS_DYNAMIC, "a dynamic_depends_on source changed"))
        if old[3] != new[3]:
            moved.append(
                (
                    MISS_ARGS,
                    "called with arguments not seen "
                    + ("on the last call" if since == "since the last call" else "in the last run"),
                )
            )
        if not moved:
            return MISS_FIRST, "no entry for this key"
        detail = "; and ".join(detail for _, detail in moved)
        return moved[0][0], detail + (f"{WHAT_CHANGED}{what}" if what else "")

    def _code_changed_detail(self, func_name: str, old_state: str, new_state: str, doc: dict, since: str) -> str:
        """A "code or state changed" detail, naming what changed when known."""
        detail = f"the function's code, a helper it calls, or a value it reads changed {since}"
        what = self._what_changed(func_name, old_state, new_state, doc)
        return detail + (f"{WHAT_CHANGED}{what}" if what else "")

    def _keep_state_ledger(self, slot: tuple[str, str], ledger: dict) -> None:
        """Keep the ledger of the first key build that produced this
        ``(func_name, state)``."""
        ledgers = self._state_ledgers
        ledgers[slot] = ledger
        if len(ledgers) > 512:
            try:
                ledgers.pop(next(iter(ledgers)))
            except (RuntimeError, StopIteration, KeyError):
                pass  # another thread trimmed it first

    def _flat_ledger(self, func_name: str, state: str, doc: dict | None = None) -> dict[str, str] | None:
        """``{part: short digest}`` for *state*: this process's ledger, else the record's."""
        ledger = self._state_ledgers.get((func_name, state))
        if ledger is None:
            recorded = (doc or {}).get("states", {}).get(state)
            return recorded if isinstance(recorded, dict) else None

        def short(value: Any) -> str:
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]

        flat: dict[str, str] = {}
        grouped: dict[str, list] = {}
        for label, value in list(ledger.items()):
            if label == "@chain":
                for i, link in enumerate(value):
                    flat[f"@{i}"] = short(link)
            elif label == "source":
                flat["source"] = short(value)
            elif isinstance(label, tuple) and label[0] == "globals":
                via = f" (read by {label[1]})" if label[1] else ""
                for name, digest in value:
                    # `X#carried`, `X#cls:C`: more of what global X is.
                    grouped.setdefault(f"global {name.split('#', 1)[0]}{via}", []).append((name, digest))
            elif isinstance(label, tuple) and label[0] == "env":
                # Already worded: "environment variable TENANT".
                flat[label[1]] = short(value)
            elif isinstance(label, tuple):
                kind = "cached function" if label[0] == "calls" else label[0]
                flat[f"{kind} {label[1]}"] = short(value)
        for name, parts in grouped.items():
            flat[name] = short(sorted(parts))
        return flat

    def _what_changed(self, func_name: str, old_state: str, new_state: str, doc: dict | None = None) -> str | None:
        """Name what moved between two states of *func_name*, or None if unknown.

        "code or state changed" alone sent every round-20 tester to diff their
        own edits: a moved helper, an edited constant, a changed default, a
        path whose case differed by launch mode all read the same.
        """
        old = self._flat_ledger(func_name, old_state, doc)
        new = self._flat_ledger(func_name, new_state, doc)
        if not old or not new:
            return None
        return describe_state_change(old, new)

    @staticmethod
    def _stale_file_deps(metadata: CacheMetadata) -> dict[str, str]:
        """``{path: what changed}`` for each recorded dependency that moved.

        The same freshness check a lookup makes, so the answer cannot
        contradict the behaviour it explains.
        """

        stale: dict[str, str] = {}
        seen: set[str] = set()
        for path, recorded in (metadata.auto_file_deps or {}).items():
            here = dep_path_for_this_process(path, recorded)
            same = same_file_key(here)
            if same in seen:
                continue
            resolved, is_fresh, why = dep_is_fresh(path, recorded)
            if not is_fresh:
                seen.add(same)
                stale[resolved or here] = STALE_REASON_TEXT.get(why or "", "changed")
        return stale

    def _describe_stale_files(self, metadata: CacheMetadata) -> str:
        stale = self._stale_file_deps(metadata)
        if not stale:
            return "a file it read"
        path, why = next(iter(stale.items()))
        more = f" and {len(stale) - 1} more" if len(stale) > 1 else ""
        return f"{path} ({why}){more}"

    #: Keys remembered per function beside the cache, most recent last.
    _STORED_KEYS_MAX = 64

    def _stored_keys_path(self, func_name: str) -> str | None:
        """Where this function's recently stored keys are recorded, or None.

        Beside the entries, in the directory of the backend actually built --
        never a configured path, so reading a miss reason cannot create a
        cache directory -- and only for a backend that has a local directory.
        """
        path = self._backend.local_dir if self._backend is not None else None
        if not path:
            return None
        name = hashlib.sha256(func_name.encode("utf-8")).hexdigest()[:32]
        return os.path.join(path, ".keys", f"{name}.json")

    def _stored_doc(self, func_name: str) -> dict[str, dict[str, list]]:
        """What earlier runs recorded for *func_name*, oldest first per kind.

        ``keys``: ``{cache_key: [stored_at, ttl]}`` for results that reached
        disk. ``ram_only``: ``{cache_key: [computed_at, why]}`` for results a
        run computed and kept in RAM only (see `_remember_ram_only`). Read
        through a memo on the file's (mtime, size), because every miss asks.
        """
        empty: dict[str, dict[str, list]] = {"keys": {}, "ram_only": {}, "states": {}, "warned": {}}
        path = self._stored_keys_path(func_name)
        if path is None:
            return empty

        with self._stored_doc_lock:
            try:
                st = os.stat(path)
            except OSError:
                return empty
            memo = self._stored_doc_memo.get(path)
            if memo is not None and memo[0] == (st.st_mtime_ns, st.st_size):
                return {kind: dict(value) for kind, value in memo[1].items()}
            try:
                # Cash's own bookkeeping: a nested call reads this while the
                # OUTER call's file tracker is live, and it must not become
                # that entry's dependency.
                with untracked(), open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                # Another process mid-rewrite: the last record read beats none,
                # which reads every miss as "new arguments".
                if memo is not None:
                    return {kind: dict(value) for kind, value in memo[1].items()}
                return empty
            return self._memo_stored_doc(path, st, data)

    def _memo_stored_doc(self, path: str, st: os.stat_result, data: Any) -> dict[str, dict[str, list]]:
        """Keep *data* as the record at *path* as of *st*; return a copy."""
        doc = {}
        for kind in ("keys", "ram_only", "states", "warned"):
            value = data.get(kind) if isinstance(data, dict) else None
            doc[kind] = value if isinstance(value, dict) else {}
        if len(self._stored_doc_memo) >= 256:
            self._stored_doc_memo.clear()
        self._stored_doc_memo[path] = ((st.st_mtime_ns, st.st_size), doc)
        return {kind: dict(value) for kind, value in doc.items()}

    def _write_stored_doc(self, func_name: str, doc: dict[str, dict[str, list]]) -> None:
        """Replace the record for *func_name*. Raises; callers swallow."""
        path = self._stored_keys_path(func_name)
        if path is None:
            return
        with self._ram_only_lock:
            shown = self._warned_pending.pop(func_name, None)
        if shown:
            doc.setdefault("warned", {}).update(shown)
        for kind, most in (
            ("keys", self._STORED_KEYS_MAX),
            ("ram_only", self._STORED_KEYS_MAX),
            ("states", self._STORED_STATES_MAX),
            ("warned", self._STORED_STATES_MAX * 4),
        ):
            entries = doc.setdefault(kind, {})
            while len(entries) > most:
                entries.pop(next(iter(entries)))

        keys_dir = os.path.dirname(path)
        recreate_cache_dir(os.path.dirname(keys_dir))
        os.makedirs(keys_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with self._stored_doc_lock, untracked():
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"func": func_name, **doc}, fh)
            os.replace(tmp, path)
            # What this process just wrote is what its next miss reads.
            self._memo_stored_doc(path, os.stat(path), doc)

    def _remember_ram_only(self, func_name: str, cache_key: str, why: str) -> None:
        """Note a result this process kept in RAM only, for the NEXT run's reason.

        Without it the next run's miss read "new arguments: not seen in the
        last run" although the last run was called with exactly these (round
        19, 4 of 5 testers). Buffered and written once, at shutdown: a result
        kept in RAM is by definition a quick one, and rewriting the record per
        miss would cost more than the body.
        """
        with self._ram_only_lock:
            pending = self._ram_only_pending.setdefault(func_name, {})
            pending.pop(cache_key, None)
            pending[cache_key] = [time.time(), why]
            while len(pending) > self._STORED_KEYS_MAX:
                pending.pop(next(iter(pending)))

    def _flush_ram_only_keys(self) -> None:
        """Write what `_remember_ram_only` buffered. Never raises."""
        with self._ram_only_lock:
            pending, self._ram_only_pending = self._ram_only_pending, {}
            for func_name in self._warned_pending:
                pending.setdefault(func_name, {})  # its record takes the shown warnings
        for func_name, entries in pending.items():
            try:
                with self._stored_doc_lock:
                    doc = self._stored_doc(func_name)
                    for key, value in entries.items():
                        doc["keys"].pop(key, None)  # its disk copy is gone: this run recomputed it
                        doc["ram_only"].pop(key, None)
                        doc["ram_only"][key] = value
                        self._record_state(func_name, key, doc)
                    self._write_stored_doc(func_name, doc)
            except Exception:  # noqa: BLE001 - a diagnostic aid
                logger.debug("could not record RAM-only keys for %s", func_name, exc_info=True)

    def _record_stored_key(self, func_name: str, cache_key: str, ttl: int | None) -> None:
        """Remember that *cache_key* reached disk, for the next process's reasons.

        One small file per function, rewritten on each persisted store (the
        compute that just ran dwarfs it). A pool's threads take turns (see
        ``_stored_doc_lock``); concurrent PROCESSES race to the last rename,
        and the loser's key is missing from the record, which costs a vaguer
        reason, never a wrong answer. Never raises.
        """
        if self._stored_keys_path(func_name) is None:
            return
        with self._ram_only_lock:
            self._ram_only_pending.get(func_name, {}).pop(cache_key, None)
        try:
            with self._stored_doc_lock:
                doc = self._stored_doc(func_name)
                doc["ram_only"].pop(cache_key, None)
                doc["keys"].pop(cache_key, None)
                doc["keys"][cache_key] = [time.time(), ttl]
                self._record_state(func_name, cache_key, doc)
                self._write_stored_doc(func_name, doc)
        except Exception:  # noqa: BLE001 - a diagnostic aid; the store succeeded
            logger.debug("could not record the stored key for %s", func_name, exc_info=True)

    #: States whose ledger the record keeps per function, most recent last.
    _STORED_STATES_MAX = 8

    def _record_state(self, func_name: str, cache_key: str, doc: dict) -> None:
        """Put the ledger of *cache_key*'s state in *doc*, for the next run's
        "what changed". Once per state; kept most recent last."""
        parts = cache_key.rsplit(":", 3)
        if len(parts) != 4:
            return
        states = doc.setdefault("states", {})
        if parts[1] in states:
            states[parts[1]] = states.pop(parts[1])
            return
        flat = self._flat_ledger(func_name, parts[1])
        if flat:
            states[parts[1]] = flat

    def _remember_outcome(self, cache_key: str, outcome: dict[str, Any]) -> None:
        outcome.setdefault("at", time.time())
        self._store_outcomes[cache_key] = outcome
        self._store_outcomes.move_to_end(cache_key)
        while len(self._store_outcomes) > STORE_OUTCOMES_MAX:
            self._store_outcomes.popitem(last=False)

    #: How deep into a returned container an argument is looked for.
    _SHARED_RESULT_DEPTH = 2

    def _warn_shared_result(self, func, func_name: str, result, args, kwargs) -> None:
        """Say so when the result shares state with something the caller holds.

        A hit hands back a value rebuilt from the stored bytes, so what the
        computing run shares, a later run does not: ``return base[lo:hi]``
        stops sharing memory with ``base``, ``return Wrapper(rows)`` stops
        holding the caller's list, and ``return CONFIG`` stops carrying the
        caller's writes back to the module (found attacking the decorator
        before round 26 -- correct on the computing run, silently different on
        every later one).

        Cash cannot tell whether the caller relies on that sharing, so this
        names what will differ rather than refusing to cache; ``assume_safe``
        waives it. Detection is cheap and evidence-only: memory shared with an
        ndarray argument (a bounds check), the result BEING or holding an
        argument (identity), or the result being one of the function's module
        globals (identity).
        """
        if self._purity_modes.get(func_name, "warn") == "silent":
            return
        try:
            shared = self._shared_with(result, args, kwargs, func)
        except Exception:  # noqa: BLE001 - a diagnostic must never break a call
            return
        if shared is None:
            return
        what, name = shared
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "shared-result",
            f"@cash.cache on {func_name}: the result {what} '{name}', which the "
            f"caller still holds. A cache HIT hands back a value rebuilt from "
            f"the stored bytes, so from the next run on they are separate "
            f"objects: writes through one will not be seen in the other.",
            code="CACHE-RESULT-SHARED",
            fix=(
                "return something of its own (`.copy()`, `dict(...)`, "
                "`list(...)`) if the caller reads it independently; pass "
                "assume_safe=True once you have checked that nothing relies "
                "on the sharing."
            ),
        )

    def _shared_with(self, result, args, kwargs, func) -> tuple[str, str] | None:
        """``(what, name)`` for the first sharing found in *result*, or None."""

        try:
            names = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            names = []
        supplied = [(names[i] if i < len(names) else f"arg{i}", value) for i, value in enumerate(args)]
        supplied += list(kwargs.items())

        def contains(value, target, depth):
            if value is target:
                return True
            if depth <= 0:
                return False
            kind = type(value)
            if kind is dict:
                return any(contains(v, target, depth - 1) for v in value.values())
            if kind in (list, tuple, set, frozenset):
                return any(contains(v, target, depth - 1) for v in value)
            state = getattr(value, "__dict__", None)
            if isinstance(state, dict):
                return any(contains(v, target, depth - 1) for v in state.values())
            return False

        for name, value in supplied:
            if not is_mutable(value):
                # Nothing can be written through it, so nothing can differ.
                continue
            if value is result:
                return "is the argument", name
            if contains(result, value, self._SHARED_RESULT_DEPTH) and is_mutable(value):
                return "holds the argument", name
            shared = shares_memory(result, value)
            if shared:
                return "shares memory with the argument", name
        globals_ = getattr(func, "__globals__", None)
        if isinstance(globals_, dict):
            for name, value in list(globals_.items()):
                if value is result and is_mutable(value):
                    return "is the module global", name
        return None

    def _store_refusal(
        self,
        func: Callable,
        func_name: str,
        res: Any,
        rng_new: bool,
        cache_if: Callable[[Any], bool] | None,
        tracker: Any,
        capture_watch: Any = NO_WATCH,
        observer: Any = None,
    ) -> str | None:
        """Why *res* must not be stored, or ``None`` to store it.

        One decision for the sync, async and streaming paths, which used to
        carry three copies of it -- and it now says WHY, because "not stored"
        is the answer to the next call's "why did that miss?".
        """
        # Skip the write exactly once when THIS call revealed that the
        # function draws: its key was built before we knew, so an entry stored
        # now carries no seed epoch and would be rebuilt and matched forever --
        # serving a result computed under a seed the user has since changed.
        # The next call keys it correctly.
        refusal = (
            "its first call drew random numbers the key did not yet cover; the next call keys them" if rng_new else None
        )
        if refusal is None and cache_if is not None:
            try:
                refusal = None if cache_if(res) else "cache_if returned False"
            except Exception as e:  # noqa: BLE001 - user predicate
                self._warn_cache_if_raised(func_name, e)
                refusal = "cache_if raised"
        # After the body ran, before deciding to store: a provisional global
        # this call moved must stop being folded (CAS-270).
        if capture_watch is not NO_WATCH:
            self._learn_mutating_captures(func, func_name, capture_watch)
        if refusal is None and self._refuses_identity_coupled(func_name, res):
            refusal = "the result is tied to the identity of an object in memory"
        if refusal is None and self._inputs_moved_during_call(func_name, tracker):
            refusal = "a file it read changed while it ran"
        stale_memo = getattr(tracker, "stale_memo_reads", None)
        if refusal is None and stale_memo:
            refusal = (
                f"a memoised helper handed it data read from an earlier version of "
                f"{sorted(stale_memo)[0]}; a fresh process reads the file as it is now"
            )
        if refusal is None and self._code_moved_since_keyed(func, func_name):
            refusal = "its code changed on disk after this process keyed it"
        if refusal is None and getattr(observer, "mock_called", False):
            # Wherever the mock sat -- below the library call the body makes,
            # or swapped in after the key's bindings were read -- the result
            # may be a test's fake, and the next real run would be served it
            # (round 20). Not waivable: no audit makes a fake the answer.
            refusal = "a unittest.mock object was called while it ran, so the result may be a test's fake"
        mutated = getattr(observer, "mutated_args", None)
        if refusal is None and mutated and self._purity_modes.get(func_name, "warn") != "silent":
            # A hit returns the stored value and leaves the caller's object as
            # it was, where this call changed it: downstream of the call, the
            # program then differs between a hit and a miss (round 19:
            # `a -= a.mean()`, `rng.shuffle(a)`, `np.clip(..., out=a)`). Not
            # storing makes every call run, which is what the code means.
            # `assume_safe=True` is the audited opt-out.
            names = ", ".join(repr(n) for n in mutated)
            refusal = (
                f"it changed its argument{'s' if len(mutated) > 1 else ''} {names} in place, which a hit would not do"
            )
        return refusal

    def _note_not_stored(self, cache_key: str, refusal: str) -> None:
        self._remember_outcome(cache_key, {"not_stored": refusal})

    @staticmethod
    def _not_persisted_reason(stored_meta: dict[str, Any], execution_time: float) -> str | None:
        """Why a stored value reached only RAM, or ``None`` if it went further.

        Only a tiered backend says where a value landed; anything else reports
        nothing, and nothing is claimed.
        """
        tiers = stored_meta.get("storage")
        skipped = stored_meta.get("persist_skipped")
        if not isinstance(tiers, list) or skipped is None or any(t != "RAM" for t in tiers):
            return None
        if skipped == "size":
            return "too big for the persistent tier's size cap"
        # There used to be three more answers here: "under the 0.1s
        # persistence floor", "the cost model judged restoring it no cheaper
        # than recomputing it", and the rate ceiling's "more cache per second
        # saved than cash will spend". None can happen to a decorated result:
        # `@cash.cache` persists what it is given, and only a size cap stops it
        # (see `TieredBackend.set`). Reporting a floor that no longer applies
        # would send the reader looking for a setting to change.
        return None

    @staticmethod
    def _snapshot_tracked_deps(tracker: Any, code_module: str | None = None) -> dict[str, dict[str, Any]] | None:
        """Snapshot everything *tracker* saw this call read - local and remote.

        Both land in one dict: they answer the same question ("did what this
        call read change since?"), and every consumer already routes that
        question through ``file_dep_is_fresh``, which branches on the entry.
        Remote entries cost one metadata request each to snapshot; that is the
        price of the read being tracked at all, and it is small against the
        download the entry exists to avoid.
        """

        read_stats = getattr(tracker, "read_stats", {})
        hashed_at = getattr(tracker, "read_hashed_at", {})
        known = {
            path: (read_stats[path], digest, hashed_at.get(path))
            for path, digest in getattr(tracker, "read_digests", {}).items()
            if path in read_stats
        }
        deps = snapshot_dependencies(
            tracker.get_accessed_files(),
            tracker.get_accessed_remote_urls(),
            tracker.get_absent_files(),
            known=known,
        )
        # A file beside the function's own code is part of this INSTALL, not a
        # fixed location: record where it sits relative to the code, so another
        # install or release checks its own copy (CAS-108).
        return attach_code_relative(deps, code_module) or None

    @staticmethod
    def _propagate_file_deps_to_active_tracker(metadata: CacheMetadata) -> None:
        """Register this entry's recorded deps with the enclosing
        ``FileAccessTracker`` (if any), so a cached function that calls this
        one on a *hit* still inherits its dependencies. Best-effort: any
        failure (no tracker active, import issue) is silently ignored."""
        snap = getattr(metadata, "auto_file_deps", None)
        if not snap:
            return
        try:
            tracker = active_tracker.get()
        except Exception:  # noqa: BLE001 - tracking is best-effort
            return
        if tracker is None:
            return
        for path, recorded in snap.items():
            # A remote entry must go back onto the remote channel: routed to
            # ``add_tracked`` it would enter the file set, be stat'ed, and be
            # dropped - so the outer entry would silently lose the dependency.
            if isinstance(recorded, dict) and recorded.get("remote"):
                tracker.add_tracked_remote(path)
            else:
                # The file THIS process would read -- another install's copy
                # would give the enclosing entry the writer's path (CAS-108).
                # The hit just checked this file against the recorded hash, so
                # that hash is the file as it is: no second read to take it.
                digest = recorded.get("hash") if isinstance(recorded, dict) else None
                tracker.add_tracked(dep_path_for_this_process(path, recorded), digest)

    def _auto_file_deps_fresh(self, metadata: CacheMetadata) -> bool:
        """Return True if every file recorded in ``metadata.auto_file_deps``
        still matches on disk.

        Auto-tracked deps are captured during the first compute via
        `cash.tracking.file_tracker.FileAccessTracker` and stored as
        ``{path: {'mtime': float, 'size': int, 'hash': str}}``. If a recorded
        path is gone or its content changed, we invalidate the cache so the next
        compute re-reads the file. A path that disappears is also a change.

        Freshness is decided by the shared
        :func:`cash.tracking.file_dep_snapshot.snapshot_is_fresh` - the same
        content-authoritative check the notebook path uses, so
        the two subsystems can't drift. ``(mtime, size)`` alone was ambiguous in
        both directions: a touch (identical content, bumped mtime)
        recomputed needlessly, and a same-size edit under an indistinguishable
        mtime was missed and served stale. The helper checks the cheap size
        first and only hashes when the size matches.
        """
        snap = metadata.auto_file_deps or {}
        if not snap:
            return True  # nothing to check

        # Remote entries cost a network round trip each to check, so the check
        # itself is worth measuring - see _warn_if_validation_is_expensive.
        #
        # Local ones are measured too, as what is left of the pass once the
        # remote resolutions are taken out. Hashing is not free either, and
        # file deps PROPAGATE: an aggregate that calls ten cached functions
        # inherits their inputs, so a fifty-file pipeline paid for fifty checks
        # on every one of those hits. Measured at 168 ms a hit before the
        # digest memo landed, with nothing anywhere to say so -- the remote
        # channel had a cost warning and the local one, which every user has,
        # did not.
        started = _perf_counter()
        with measured_validation() as validation:
            fresh, stale = snapshot_is_fresh(snap)
        local_seconds = max(0.0, _perf_counter() - started - validation.seconds)
        local_count = sum(
            1 for recorded in snap.values() if not (isinstance(recorded, dict) and recorded.get("remote"))
        )
        if stale is not None:
            logger.debug("[FILE_DEP] stale (%s): %s", stale.reason, stale.path)
        Cash._warn_if_validation_is_expensive(validation, metadata)
        self._warn_if_local_validation_is_expensive(local_seconds, local_count, metadata)
        return fresh

    def _warn_if_local_validation_is_expensive(
        self,
        seconds: float,
        count: int,
        metadata: CacheMetadata,
    ) -> None:
        """Say so when hashing this entry's own files costs a real share of the
        saving.

        The same rule the remote channel uses (``validation_is_expensive``):
        more than half the compute it avoids past a 0.25 s floor, or more than
        2 s outright. Shared deliberately -- "proving it fresh cost more than
        recomputing would" is one judgement, and it should not depend on whether
        the input was a file or a URL.

        After the first check of a file this is microseconds (the digest is
        memoized per process), so reaching the threshold means many
        dependencies, very large ones, or a slow filesystem. Each of those is
        something the user can act on, and none of them shows up anywhere else.
        """
        if not count or not seconds:
            return

        saved = metadata.execution_time
        if not validation_is_expensive(seconds, saved):
            return
        label = metadata.func_name or "a cached call"
        against = f", against {saved:.2f}s of compute it avoids" if saved and saved > 0 else ""
        self._warn_once(
            CashCacheIneffectiveWarning,
            label,
            "local-freshness-cost",
            f"cash spent {seconds:.2f}s checking {count} tracked "
            f"{'file' if count == 1 else 'files'} for freshness on {label}"
            f"{against}, so proving the result fresh costs a serious share of "
            f"what it saves.",
            code="CACHE-FRESHNESS-COST",
            fix="depend on fewer or smaller files -- cache a summary rather than "
            "every input -- or split the function so the expensive inputs are "
            "read by a callee whose deps the aggregates do not inherit. Note "
            "that files above file_hash_full_max_bytes are sampled rather "
            "than hashed in full, which is cheaper per file but not per file "
            "COUNT.",
        )

    @staticmethod
    def _warn_if_validation_is_expensive(validation: Any, metadata: CacheMetadata) -> None:
        """Say so when checking freshness costs a serious share of the saving.

        A freshness check that has to ask the network is the one overhead a user
        cannot see: it happens on the HIT path, where the badge shows a saving
        and nothing shows what the saving cost to establish.
        """
        if not validation.count:
            return

        saved = metadata.execution_time
        if validation_is_expensive(validation.seconds, saved):
            warn_validation_cost_once(
                metadata.func_name or "a cached call",
                validation.count,
                validation.seconds,
                saved,
            )

    def _chunks_are_intact(self, cache_key: str, metadata: CacheMetadata) -> bool:
        """True unless this is a chunked manifest missing some of its chunks.

        A manifest can outlive its chunks -- eviction reaches them separately,
        and until chunks carried the producer's execution_time the persistence
        gate dropped them while keeping the manifest. The reader terminates
        iteration on a missing chunk, so the result of that split was an
        entry that returned FEWER items than it stored, or none at all,
        without a word. A truncated answer is worse than a slow one, so the
        entry is treated as absent and recomputed.

        Metadata-only reads: the point is to check presence, not to load the
        payload and undo the laziness chunking exists for.

        Scope, because the docs depend on it: BOTH read paths apply this --
        ``_try_get_cached`` for the default one, and the double-checked re-read
        inside ``_compute_with_lock`` for ``use_locking=True``. Keep it that
        way. The locking path skipped it until 2026-09-06 and served a short
        iterator with no recompute, no error and no warning: measured 3 of 10
        items when a later chunk was missing, and 0 items when the first one
        was. Both paths are pinned by
        ``tests/test_core/test_iterator_caching.py``.
        """
        if getattr(metadata, "iterator_storage", None) != "chunked":
            return True
        try:
            for index in range(metadata.n_chunks or 0):
                if self.backend.get_metadata(f"{cache_key}:chunk_{index}") is None:
                    logger.debug(
                        "[CORE] chunk %d of %s is missing; treating the entry "
                        "as a miss rather than serving a short result",
                        index,
                        cache_key,
                    )
                    return False
        except Exception:  # noqa: BLE001 - an integrity check must not break a call
            return True
        return True

    def _rng_replay_parts(self, drew: bool, pre_state: dict | None) -> dict:
        """What a later hit needs to leave the RNG where this call left it.

        A hit never runs the body, so the stream it advanced stays where it was
        and the CALLER's next draw returns what the function drew: with
        ``np.random.seed(0)``, the draw after a hit WAS the cached value (found
        attacking the decorator before round 26). The notebook path replays the
        recorded state; this is the same for the decorator.

        Both ends are recorded. Replaying the post-state is only right when the
        stream is where it was when the body ran, so the pre-state is what a hit
        checks first -- a program that drew somewhere else in between is left
        alone rather than rewound.
        """
        if not drew or pre_state is None:
            return {}
        try:
            return {"rng_pre": pre_state, "rng_post": capture_rng_state()}
        except Exception:  # noqa: BLE001 - never break a call over this
            return {}

    @staticmethod
    def _replay_rng_state(metadata: Any) -> None:
        """Put the global RNG where the computed call left it (see
        :meth:`_rng_replay_parts`), when it is where that call started."""
        replay = getattr(metadata, "rng_replay", None) or {}
        post, pre = replay.get("rng_post"), replay.get("rng_pre")
        if not post or not pre:
            return
        try:
            # Only the streams the body advanced, and only while each is where
            # that body found it. Every other module is left alone: a process
            # seeds `random` from the OS at import, so comparing all of them
            # would refuse every replay.
            advanced = rng_modules_changed(pre, post)
            if not advanced:
                return
            live = capture_rng_state()
            if any(m not in live for m in advanced):
                return
            if rng_modules_changed({m: pre[m] for m in advanced}, {m: live[m] for m in advanced}):
                return
            restore_rng_state({m: post[m] for m in advanced})
        except Exception:  # noqa: BLE001 - a replay must never break a hit
            logger.debug("[CORE] could not replay the RNG state of a hit", exc_info=True)

    def _wrap_iterator_hit(self, call: Call, metadata: CacheMetadata | None, hit: Any) -> Any:
        """Wrap a cache-hit value in the right iterator class.

        Iterators (including the single-chunk case) are stored under
        an ``iterator_storage='chunked'`` manifest plus N chunk
        entries; on hit they're returned as a fresh
        ``ChunkedCachedIterator``, which recomputes the rest from the
        function if a chunk is gone. Non-iterator return types live as
        a single blob and are returned as *hit* directly.

        Every hit path goes through here -- the first lookup, the locked
        re-read (`_compute_with_lock`) and the async single-flight follower
        (`_await_leader`) -- and all of them pass the call's `recompute`. The
        async ones used to leave it out, so a missing chunk raised there
        instead of recomputing.
        """
        if metadata and metadata.iterator_storage == "chunked":
            n_chunks = metadata.n_chunks or 0
            return ChunkedCachedIterator(self, call.cache_key, n_chunks, call.recompute)
        return hit

    def _make_wrapper(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        ttl_decl: int | None,
        cache_if: Callable[[Any], bool] | None = None,
        chunk_max_items: int = 1_000_000,
        chunk_max_bytes: int = 1_000_000_000,
    ) -> Callable:
        """Build and return the caching wrapper for *func*, sync or async.

        One wrapper for both: everything before and after the body is the
        same sync code (`_lookup`, `_body_scope`, `_finish_miss`), and the
        two variants differ only in whether they await the body. The sync and
        async wrappers used to be two ~150-line copies, and they had drifted.
        """
        spec = CallSpec(func, func_name, dynamic_depends_on, ttl_decl, cache_if, chunk_max_items, chunk_max_bytes)

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

    def _lookup(self, spec: CallSpec, args: tuple, kwargs: dict, *, async_body: bool) -> Call:
        """Everything a call does before the body: analysis, key, lookup.

        Returns the call's state. ``call.outcome`` is what the wrapper returns
        now -- a hit, or the result of a call that has no key -- or
        ``CACHE_MISS`` when the body has to run.
        """
        func, func_name = spec.func, spec.func_name
        call = Call(args, kwargs)
        call.call_start = _perf_counter()
        if func_name not in self._analyzed:
            # Double-checked under a per-function lock: the key is built
            # from what this populates, so two threads must not race it.
            with self._analysis_lock:
                if func_name not in self._analyzed:
                    self._analyze_dependencies(func)
                    self._analyzed.add(func_name)
        # Inherit the shortest TTL of any TTL'd dependency (computed after
        # analysis populates the graph).
        call.ttl = self._effective_ttl(func_name, spec.ttl_decl)
        if async_body:
            call.recompute = lambda: run_to_completion(lambda: func(*args, **kwargs))
        else:
            call.recompute = lambda: func(*args, **kwargs)

        # Everything from here to the hit/miss verdict is cash's own cost,
        # not the user's work. Two perf_counter pairs measured at 196ns
        # against a 25.5us floor for the cheapest possible cached call --
        # 0.8%, so this is not gated behind a heuristic.
        overhead_t0 = _perf_counter()
        key_result, call.capture_watch = self._resolve_cache_key(
            func, func_name, spec.dynamic_depends_on, args, kwargs, call.call_start
        )
        if key_result[0] is CACHE_MISS:
            call.outcome = key_result[1]
            return call
        call.cache_key, call.state_hash, call.args_hash = key_result

        raw_metadata, cached_data = self.backend.get(call.cache_key)
        call.metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        hit = self._try_get_cached(
            call.cache_key, call.metadata, cached_data, call.call_start, call.args_hash, func_name, call.ttl
        )
        call.cash_overhead = _perf_counter() - overhead_t0
        if hit is not CACHE_MISS:
            self._note_effectiveness(
                func_name,
                call.cash_overhead,
                body_seconds=getattr(call.metadata, "body_seconds", None),
                was_hit=True,
            )
            call.outcome = self._wrap_iterator_hit(call, call.metadata, hit)
        return call

    def _reread(self, spec: CallSpec, call: Call) -> Any:
        """Look the key up again (another caller may have stored it meanwhile):
        the hit, wrapped like any other, or ``CACHE_MISS``.

        The SAME validity test as the first lookup, by calling the same
        function -- not a hand-rolled subset of it. The locked re-read used to
        re-implement the checks, and it kept losing one: first
        ``_chunks_are_intact`` (a short iterator came back), then
        ``_auto_file_deps_fresh`` (under ``use_locking=True`` an edited file
        was served stale). One function decides whether an entry may be served.
        """
        raw_metadata, cached_data = self.backend.get(call.cache_key)
        if raw_metadata is None:
            return CACHE_MISS
        metadata = CacheMetadata.from_dict(raw_metadata)
        hit = self._try_get_cached(
            call.cache_key, metadata, cached_data, call.call_start, call.args_hash, spec.func_name, call.ttl
        )
        if hit is CACHE_MISS:
            return CACHE_MISS
        return self._wrap_iterator_hit(call, metadata, hit)

    @contextlib.contextmanager
    def _body_scope(self, spec: CallSpec, call: Call) -> Iterator[BodyRun]:
        """Run the body inside this: file tracking, effect observation, RNG
        watch and timing, shared by the sync and async wrappers.

        The caller runs the body in the ``with`` block and puts what it
        returned in ``run.res``; on the way out this measures the body and
        reads what it drew, still inside the tracker. A body that raises is
        logged as such and the exception propagates.
        """
        # Wrap the function call in FileAccessTracker so any auto-tracked
        # file reads (pandas/numpy/joblib/open/...) are recorded as implicit
        # cache dependencies - a later content change forces a recompute.
        func, func_name, args, kwargs = spec.func, spec.func_name, call.args, call.kwargs
        run = BodyRun()
        run.tracker = FileAccessTracker(getattr(func, "__globals__", None), propagate_to_parent=True, hash_on_read=True)
        # Watch for side effects the STATIC analyzer cannot see, which is
        # anything happening inside an installed library. Only on this
        # (missing) path: a hit runs no body, so there is nothing to observe
        # and nothing to pay for.
        run.observer = self._make_effect_observer()
        run.observer.arg_snapshot = self._argument_snapshot(func_name, args, kwargs)
        run.observer.arg_identities = self._argument_identities(func_name, args, kwargs)
        # Watch the global RNG across the call: a draw inside the body is an
        # input the key cannot see statically.
        run.rng_pre = self._capture_rng_pre_state()
        with run.tracker, run.observer:
            threads_at_start = THREADS_IN_CALLS[0]
            body_t0 = _perf_counter()
            nested = [0.0]
            nested_token = NESTED_CASH_SECONDS.set(nested)
            try:
                self._track_declared_files(run.tracker, func_name)
                yield run
            except Exception as exc:
                self._log_raised(func_name, exc, call.call_start)
                raise
            finally:
                NESTED_CASH_SECONDS.reset(nested_token)
            # The user's own work, isolated. Everything cash does sits outside
            # this pair, which is the whole point: it is the only number that
            # can answer "did caching pay?".
            run.body_seconds = max(0.0, _perf_counter() - body_t0 - run.tracker.read_hash_seconds - nested[0])
            run.saves_seconds = run.body_seconds / max(threads_at_start, THREADS_IN_CALLS[0], 1)
            run.rng_new = self._note_rng_draw(func_name, run.rng_pre)

    def _finish_miss(self, spec: CallSpec, call: Call, run: BodyRun) -> Any:
        """Everything a missed call does after its body: check, store, log."""
        func, func_name, args, kwargs = spec.func, spec.func_name, call.args, call.kwargs
        res = run.res
        # A generator is handed straight back, wrapped, and cached only once
        # the caller has drained it. Draining it here instead meant a streamed
        # response arrived in one lump after the full latency, so
        # `@cash.cache` changed how the function behaved. `_stream_and_store`
        # carries the tracker into each production step so lazy file reads are
        # still recorded. An async function returning a SYNC generator streams
        # the same way.
        if is_one_shot_iterator(res):
            # Logged HERE, not at exhaustion. `stats_wrapper` counts this
            # call's entry the moment the wrapper returns, so an entry written
            # when the caller finishes iterating would never be counted. The
            # miss is a fact about the LOOKUP, which has already happened. The
            # produce time still reaches the entry, via the manifest, which is
            # what a later hit reports as saved.
            self._log_decorator_call(
                func_name,
                cache_hit=False,
                execution_time=_perf_counter() - call.call_start,
                args_hash=call.args_hash,
                cache_key=call.cache_key,
            )
            return StreamingCachedIterator(
                self._stream_and_store(
                    res,
                    cache_key=call.cache_key,
                    func_name=func_name,
                    tracker=run.tracker,
                    observer=run.observer,
                    rng_new=run.rng_new,
                    args=args,
                    kwargs=kwargs,
                    args_hash=call.args_hash,
                    current_state_hash=call.state_hash,
                    ttl=call.ttl,
                    cache_if=spec.cache_if,
                    chunk_max_items=spec.chunk_max_items,
                    chunk_max_bytes=spec.chunk_max_bytes,
                    code_module=func.__module__,
                )
            )

        self._check_argument_mutation(func_name, args, kwargs, call.args_hash, run.observer)
        self._report_observed_effects(func_name, run.observer)
        self._credit_remembered_reads(func_name, run.tracker, args, kwargs)
        auto_file_deps = self._snapshot_tracked_deps(run.tracker, func.__module__)
        execution_time = _perf_counter() - call.call_start

        self._warn_shared_result(func, func_name, res, args, kwargs)
        refusal = self._store_refusal(
            func, func_name, res, run.rng_new, spec.cache_if, run.tracker, call.capture_watch, observer=run.observer
        )
        if refusal is not None:
            self._note_not_stored(call.cache_key, refusal)
        else:
            # Attach lineage only when the value is actually stored: a lineage
            # hash points downstream at THIS cache entry, so a cache_if-rejected
            # (uncached) value must not carry one - it would reference an entry
            # that was never written.
            self._attach_lineage(res, call.cache_key, auto_file_deps, ttl=call.ttl, func_name=func_name)
            self._store_in_cache(
                call.cache_key,
                func_name,
                res,
                call.metadata,
                call.ttl,
                call.state_hash,
                call.args_hash,
                execution_time,
                auto_file_deps=auto_file_deps,
                body_seconds=run.body_seconds,
                saves_seconds=run.saves_seconds,
                rng_replay=self._rng_replay_parts(bool(self._rng_drawing_funcs.get(func_name)), run.rng_pre),
            )
        # Everything that was not the body: the key and lookup before it, the
        # checks and the store after it.
        miss_overhead = max(call.cash_overhead, _perf_counter() - call.call_start - run.body_seconds)
        self._log_decorator_call(
            func_name,
            cache_hit=False,
            execution_time=execution_time,
            args_hash=call.args_hash,
            cache_key=call.cache_key,
            body_seconds=run.body_seconds,
            cash_seconds=miss_overhead,
        )
        self._note_effectiveness(func_name, miss_overhead, body_seconds=run.body_seconds, was_hit=False)
        return res

    async def _single_flight(self, spec: CallSpec, call: Call, compute: Callable[[], Any]) -> Any:
        """Async single-flight for ``use_locking``: coalesce concurrent awaits
        of the same key in-process, so an expensive idempotent coroutine (a
        paid API call, say) under ``asyncio.gather`` computes once instead of
        N times. The leader computes and stores; followers wait for it and
        then read the stored result, and compute themselves when it stored
        nothing (cache_if rejected it, or it raised)."""
        # Imported HERE, not at module scope: asyncio costs ~76ms of a ~290ms
        # `import cash`, and a synchronous user never needs it. Inside a
        # running loop it is necessarily already imported, so this lookup is
        # free exactly where it is used.
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return await compute()
        with self._async_inflight_lock:
            existing = self._async_inflight.get(call.cache_key)
            if existing is None:
                # Leader: publish the future the followers wait on.
                leader = concurrent.futures.Future()
                self._async_inflight[call.cache_key] = leader
        if existing is not None:
            # Follower, in this loop or another: wait for the leader, then
            # read the stored value. The wait is wrapped per follower, so
            # cancelling one leaves the leader's computation running for the
            # rest.
            try:
                await asyncio.shield(asyncio.wrap_future(existing))
            except Exception:  # noqa: BLE001 - the leader's failure is its own
                pass
            hit = self._reread(spec, call)
            if hit is not CACHE_MISS:
                return hit
            return await compute()
        try:
            return await compute()
        finally:
            # Signal followers (success or failure) and free the slot.
            with self._async_inflight_lock:
                self._async_inflight.pop(call.cache_key, None)
            if not leader.done():
                leader.set_result(None)

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

    def _wrap_with_stats(
        self,
        func: Callable,
        func_name: str,
        wrapper: Callable,
        *,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None = None,
        ttl: int | None = None,
        allow_random: bool = False,
    ) -> Callable:
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
        _stats = {
            "hits": 0,
            "misses": 0,
            "total_time_saved": 0.0,
            "lookup_seconds": 0.0,
            # What the misses cost besides their bodies: key, checks, store.
            "miss_overhead_seconds": 0.0,
            # What the misses were, and which results did not reach
            # disk: the two things "1 miss" alone could not tell anyone.
            "miss_reasons": Counter(),
            "not_persisted": Counter(),
            "not_stored": Counter(),
            # For "code or state changed" misses: WHAT changed.
            "changed": Counter(),
            # Calls that went straight through because caching is off.
            "bypassed": 0,
        }
        # Shared by reference with the end-of-run summary, which otherwise has
        # no way to reach a per-wrapper closure. Last registration wins for a
        # redefined function, which matches what `cache_info()` reports.
        self._function_stats[func_name] = _stats

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
                kind, detail = call.get("miss_reason") or (MISS_FIRST, "")
                _stats["miss_reasons"][kind] += 1
                if kind == MISS_CODE and WHAT_CHANGED in detail:
                    _stats["changed"][detail.split(WHAT_CHANGED, 1)[1]] += 1
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
                  ``_func_warnings_max`` (default 20) so it can't grow
                  unboundedly. Useful for spotting silent misbehavior
                  (cache_if predicate raised, lock failure, etc.) when
                  ``warnings.simplefilter`` swallowed the stderr emission.
            """
            total = _stats["hits"] + _stats["misses"]
            hit_rate = _stats["hits"] / total if total > 0 else 0.0
            with self._decorator_call_log_lock:
                warnings_log = list(self._func_warnings.get(func_name, []))
            return {
                "hits": _stats["hits"],
                "misses": _stats["misses"],
                "hit_rate": hit_rate,
                "total_time_saved": _stats["total_time_saved"],
                "miss_reasons": dict(_stats["miss_reasons"]),
                "warnings": warnings_log,
            }

        def cache_clear() -> None:
            """Clear all cached results for this function.

            Removes all cache entries whose key starts with the function name.
            Resets hit/miss statistics, drops the per-function warnings log,
            and forgets ``_warn_once`` dedup marks for this function so the
            next misbehavior re-warns instead of being silently swallowed.
            """
            _stats["hits"] = 0
            _stats["misses"] = 0
            _stats["total_time_saved"] = 0.0
            _stats["bypassed"] = 0
            for tally in ("miss_reasons", "not_persisted", "not_stored", "changed"):
                _stats[tally].clear()
            self._delete_backend_entries(func_name)
            with self._decorator_call_log_lock:
                self._func_warnings.pop(func_name, None)
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
                explanation = self._explain_call(
                    func,
                    func_name,
                    dynamic_depends_on,
                    ttl,
                    args,
                    kwargs,
                )
            finally:
                ACTIVE_CONFIG.reset(token)
            return dataclasses.replace(explanation, cache_dir=_backend_cache_dir(self.backend))

        stats_wrapper.cache_info = cache_info
        stats_wrapper.cache_clear = cache_clear
        stats_wrapper.explain = explain
        stats_wrapper.__wrapped__ = func
        # Marker so the purity analyzer treats a call to this wrapper as a
        # dependency-graph edge rather than recursing into cash's own wrapper
        # machinery (finding #9). functools.wraps copies __module__, which would
        # otherwise make the wrapper look like same-package user code.
        stats_wrapper._cash_cached = True
        # Declared TTL, exposed so the notebook statement cache can see it. A
        # ``ttl=0`` function must recompute every call; without this the
        # statement ``x = f()`` gets cached with no TTL under %cash_on and
        # freezes the value the decorator promised to refresh.
        stats_wrapper._cash_declared_ttl = ttl
        expose_script_function(func, stats_wrapper)
        self._wrapped_funcs[func_name] = stats_wrapper
        return stats_wrapper

    def _effective_ttl(self, func_name: str, own_ttl: int | None) -> int | None:
        """The TTL actually used for *func_name*: the minimum of its own TTL and
        the TTLs of cached functions it (transitively) depends on.

        A function whose result derives from a TTL'd dependency must refresh at
        least as often as that dependency - otherwise, because ``depends_on``
        tracks source (not runtime freshness), the downstream keeps returning a
        stale value after the dependency's TTL refresh. Functions with no TTL'd
        dependency are unaffected (effective TTL == own TTL)."""
        cached = self._effective_ttl_cache.get(func_name)
        if cached is not None or func_name in self._effective_ttl_cache:
            return cached
        ttls = [t for t in self._collect_dep_ttls(func_name, set()) if t is not None]
        if own_ttl is not None:
            ttls.append(own_ttl)
        eff = min(ttls) if ttls else None
        self._effective_ttl_cache[func_name] = eff
        return eff

    def _collect_dep_ttls(self, func_name: str, visited: set[str]) -> list[int | None]:
        """TTLs of every cached function reachable from *func_name* via the
        dependency graph (cycle-guarded)."""
        if func_name in visited:
            return []
        visited.add(func_name)
        out: list[int | None] = []
        for dep in self.graph.get_dependencies(func_name):
            if dep in self._func_ttls:
                out.append(self._func_ttls[dep])
                out.extend(self._collect_dep_ttls(dep, visited))
        return out

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
                dep_key = self.get_func_key(dep)
                self.graph.add_dependency(func_name, dep_key)
                # A declared callable dep that is NOT a decorated cached function
                # would contribute nothing to the state hash (the hasher only
                # folds functions/data-sources), silently breaking the documented
                # ``depends_on`` promise. Snapshot its source + a live
                # resolution path so edits/reloads invalidate the parent key.
                if dep_key not in self.functions:
                    self._register_declared_callable_dep(dep, dep_key, func_name)

    def _register_declared_callable_dep(self, dep: Callable[..., Any], dep_key: str, func_name: str) -> None:
        """Record a plain-callable ``depends_on`` dependency's source identity.

        Stores a source-hash snapshot and a ``(module, attr_chain)`` path for
        live re-resolution (so an on-disk edit + ``importlib.reload`` is seen).
        If the dep's source cannot be hashed (builtin / C-extension), warn once
        that the declared dependency is inert rather than silently ignore it.
        """
        try:
            snapshot = self._hash_callable_source(dep)
        except (OSError, TypeError, ValueError):
            snapshot = None
        if snapshot is None:
            self._warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: depends_on={getattr(dep, '__qualname__', dep)!r} "
                f"is a callable whose source cannot be read (builtin / C-extension), "
                f"so the declaration is inert and changes to it will NOT "
                f"invalidate the cache.",
                code="KEY-DEPENDS-ON-OPAQUE",
                fix="depend on something cash can read: pass the extension's "
                "version in as an argument, or declare a DataSource whose "
                "token is that version or build id.",
            )
            return
        self._declared_dep_snapshots[dep_key] = snapshot
        module = getattr(dep, "__module__", None)
        qualname = getattr(dep, "__qualname__", None) or getattr(dep, "__name__", None)
        if module and qualname and "<locals>" not in qualname:
            self._declared_dep_paths[dep_key] = (module, tuple(qualname.split(".")))

    def _resolve_declared_dep_hash(self, dep_key: str) -> str | None:
        """Re-resolve a declared plain-callable dep's live source hash.

        Walks the stored ``(module, attr_chain)`` path via ``sys.modules`` and
        hashes the resolved callable's current source. Returns ``None`` on any
        resolution/hash failure so the hasher falls back to the snapshot.
        """
        path = self._declared_dep_paths.get(dep_key)
        if path is None:
            return None
        mod_name, attr_chain = path
        obj: Any = sys.modules.get(mod_name)
        if obj is None:
            return None
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        if not callable(obj):
            return None
        try:
            return self._hash_callable_source(obj)
        except (OSError, TypeError, ValueError):
            return None

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
        fix = (
            "fix the resolver -- it is called with exactly the same arguments "
            "as the function -- so that it returns a DataSource, a list of "
            "them, or None for no dependency."
        )
        for resolver in resolvers:
            # Any failure makes the call unkeyable, never a key without the
            # dependency: that key would keep hitting after the data changed.
            try:
                # Resolver receives the same args as the function
                ds_result = resolver(*args, **kwargs)
                dss = ds_result if isinstance(ds_result, list) else [ds_result]
                for ds in dss:
                    if ds is None:
                        continue
                    if not isinstance(ds, DataSource):
                        raise KeyBuildFailed(
                            "KEY-DYNAMIC-DEP-FAILED",
                            f"@cash.cache on {func_name}: a dynamic_depends_on resolver "
                            f"returned a {type(ds).__name__}, which is not a DataSource, "
                            f"so cash cannot tell when it changes and the call ran uncached.",
                            fix,
                        )
                    dynamic_state_parts.append(state_token_of(ds))
            except KeyBuildFailed:
                raise
            except Exception as e:  # noqa: BLE001 - any failure here is the resolver's
                raise KeyBuildFailed(
                    "KEY-DYNAMIC-DEP-FAILED",
                    f"@cash.cache on {func_name}: dynamic_depends_on resolver raised "
                    f"{type(e).__name__} ({e}), so cash cannot tell whether that "
                    f"dependency changed and the call ran uncached.",
                    fix,
                ) from e

        if dynamic_state_parts:
            # Sort to ensure deterministic order if multiple sources
            return hashlib.sha256(":".join(sorted(dynamic_state_parts)).encode("utf-8")).hexdigest()
        return ""

    def _normalize_call_args(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Bind ``(args, kwargs)`` to the function signature and apply defaults.

        Collapses logically-identical calls written in different forms - ``f(1)``
        vs ``f(1, y=10)`` (the default) vs ``f(x=1, y=10)``, and kwargs in any
        order - to one canonical argument shape so they share a cache key
        instead of producing wasteful misses.

        Best-effort: any introspection or bind failure (builtins with no
        signature, ``*args`` calls that don't match, deliberately mismatched
        calls) returns the inputs unchanged, so behavior never regresses.
        """
        func = self.functions.get(func_name)
        cached = self._signatures.get(func_name)
        # Re-read the signature when the name has been rebound to a different
        # function object: a notebook cell re-run with an edited default keeps
        # the qualname but changes what `apply_defaults()` must fold.
        if cached is not None and cached[0] is func:
            sig = cached[1]
        else:
            try:
                sig = inspect.signature(func) if func is not None else None
            except (ValueError, TypeError):
                sig = None
            self._signatures[func_name] = (func, sig)
        if sig is None:
            return args, kwargs
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            # The call doesn't match the signature (the function itself would
            # raise when invoked). Leave the raw form untouched.
            return args, kwargs
        # ``bound.arguments`` is ordered by parameter definition, so the result
        # is canonical regardless of how the caller wrote the call. Re-express
        # named params as kwargs; keep *args positional; sort **kwargs so its
        # order doesn't leak into the key. (We only build a payload to hash, so
        # routing named params through kwargs is purely for determinism.)
        canon_args: list[Any] = []
        canon_kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                continue
            val = bound.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                canon_args.extend(val)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                # Under its own name: a `**kwargs` entry may be called after a
                # parameter, and writing both into one dict let it overwrite
                # that parameter's value. `def request(url, /, **params)`
                # called as `request("/a", url="x")` then keyed on the kwargs
                # `url` alone, so every such call shared one entry and
                # `request("/b", url="x")` was served `GET /a` (found
                # attacking the decorator before round 26).
                for k in sorted(val):
                    canon_kwargs[f"{name}:{k}"] = val[k]
            else:
                canon_kwargs[name] = val
        return tuple(canon_args), canon_kwargs

    _ARG_HASH_MEMO_CAP = 1024

    def _memo_arg_hash(self, arg: Any, lineage: str, content_hash: str) -> None:
        """Record ``id(arg) -> (weakref, lineage, content_hash)`` for the session,
        bounded so a long session can't grow the memo without limit. When full,
        drop it wholesale: the memo is a pure speedup, so an occasional cold
        start just re-hashes. Values that cannot be weak-referenced are skipped
        (they simply keep full-hashing).
        """
        try:
            wref = weakref.ref(arg)
        except TypeError:
            return
        memo = self._arg_hash_memo
        if len(memo) >= self._ARG_HASH_MEMO_CAP:
            memo.clear()
        memo[id(arg)] = (wref, lineage, content_hash)

    _FRAME_MEMO_CAP = 256

    def _frozen_array_hash(self, arr: Any) -> str | None:
        """The content hash of a frozen function's numpy result, computed once.

        Valid while the array is still that object and still read-only; an
        array made writeable again (``a.flags.writeable = True``) is keyed by
        content from then on.
        """
        entry = self._frozen_arrays.get(id(arr))
        if entry is None:
            return None
        wref, _producer, content_hash = entry
        if wref() is not arr or getattr(arr, "flags", None) is None or arr.flags.writeable:
            self._frozen_arrays.pop(id(arr), None)
            return None
        if content_hash is None:
            content_hash = builtin_hash(arr)
            entry[2] = content_hash
        return content_hash

    #: Frozen list/tuple/dict results held at once. Past it the oldest goes.
    _FROZEN_CONTAINERS_MAX = 256

    def _remember_frozen_container(self, obj: Any, producer: str, lineage: str) -> None:
        """Key a frozen function's list, tuple or dict by its producer's lineage.

        A list of two million parsed rows, passed on to two cached consumers,
        was pickled in full for every call -- warm runs about 9x slower than
        uncached -- and ``frozen=True`` on the parser changed nothing: its fast
        path covered numpy arrays alone, and a list cannot carry a tag (round
        19). Such a result is now keyed like a frozen frame: by the lineage of
        the call that produced it, audited now and then (`_audit_frozen`'s
        schedule).

        It has no weakref either, so the object is held here -- and let go
        again once nothing else holds it, swept on each new entry.
        """
        table = self._frozen_containers
        # What "held by the table alone" reads as, measured the same way: the
        # count differs between Python versions (3.14 borrows references).
        probe = [None, None, None, None, None]
        probe[0] = object()
        alone = sys.getrefcount(probe[0])
        for key, entry in list(table.items()):
            if sys.getrefcount(entry[0]) <= alone:
                table.pop(key, None)
        while len(table) >= self._FROZEN_CONTAINERS_MAX:
            table.pop(next(iter(table)))
        table[id(obj)] = [obj, producer, f"frozen:{lineage}", 0, None, self._frozen_shape(obj)]

    def _frozen_container_hash(self, obj: Any) -> str | None:
        """The lineage a frozen list/tuple/dict is keyed by, or None once it
        has been seen to change (KEY-FROZEN-MUTATED, as for a frozen frame)."""
        entry = self._frozen_containers.get(id(obj))
        if entry is None or entry[0] is not obj:
            return None
        entry[3] += 1
        uses = entry[3]
        shape = self._frozen_shape(obj)
        if len(entry) > 5 and entry[5] is not None and shape != entry[5]:
            self._frozen_containers.pop(id(obj), None)
            self._warn_frozen_mutated(obj, entry[1])
            return None
        due = (
            (self.debug or os.environ.get("CASH_DEBUG"))
            or uses == FROZEN_AUDIT_FIRST
            or (uses > FROZEN_AUDIT_FIRST and uses % FROZEN_AUDIT_EVERY == 0)
        )
        if due:
            try:
                digest = hashlib.sha256(pickle.dumps(obj)).hexdigest()
            except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
                digest = None
            if digest is not None:
                if entry[4] is None:
                    entry[4] = digest
                elif entry[4] != digest:
                    self._frozen_containers.pop(id(obj), None)
                    warn_diagnostic(
                        CashImpurityWarning,
                        "KEY-FROZEN-MUTATED",
                        f"a {type(obj).__name__} returned by {entry[1]}, which is "
                        f"declared @cash.cache(frozen=True), has been modified since "
                        f"it was returned. Calls that received it before the change "
                        f"may have been served results for the unmodified object; "
                        f"from now on it is keyed by its contents.",
                        f"take frozen=True off {entry[1]} if its result is meant to "
                        f"be modified, or modify a copy (`obj = copy.deepcopy(obj)`) "
                        f"instead.",
                    )
                    return None
        return entry[2]

    def _warn_frozen_has_no_effect(self, func_name: str, result: Any) -> None:
        """Say so when ``frozen=True`` cannot apply to what the function returned."""
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "frozen_no_effect",
            f"@cash.cache(frozen=True) on {func_name}: it returned a "
            f"{type(result).__name__}, which cash cannot mark, so frozen=True has "
            f"no effect on it -- a call that receives it still hashes it in full. "
            f"frozen=True applies to a numpy array, a pandas/polars/modin frame, "
            f"a pyarrow table, a list, tuple or dict, and any object that takes "
            f"an attribute.",
            code="KEY-FROZEN-NO-EFFECT",
            fix="return one of those types, or take frozen=True off; "
            "cash.register_hasher gives the type a cheap identity instead.",
        )

    def _warn_frozen_mutated(self, obj: Any, producer: Any = None) -> None:
        """KEY-FROZEN-MUTATED: a result declared frozen is not what it was."""
        producer = producer or getattr(obj, "_cash_lineage_producer", None) or "a frozen=True function"
        try:
            obj._cash_lineage_src = LINEAGE_SRC_DECORATOR
        except (AttributeError, TypeError):
            pass
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {producer}, which is declared "
            f"@cash.cache(frozen=True), has been modified since it was returned. "
            f"Calls that received it before the change may have been served "
            f"results for the unmodified object; from now on it is keyed by its "
            f"contents.",
            f"take frozen=True off {producer} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )

    #: How many of a container's elements the cheap audit measures.
    _FROZEN_SHAPE_SAMPLE = 8

    def _frozen_shape(self, obj: Any) -> tuple | None:
        """What *obj* is shaped like, in O(1)-ish work, or ``None``.

        The full audit hashes every byte, so it runs rarely -- the baseline at
        the 8th use and a comparison every 64th after that. That left the
        ordinary shape unprotected: produce a result, change it, pass it again.
        A length, a frame's shape and dtypes, and the lengths of a few elements
        cost nothing to read on EVERY use, and they move for the changes a
        caller actually makes (``model["w"].append(...)``). A change they
        cannot see -- a value overwritten in place, same length -- is still
        caught by the full audit.
        """
        try:
            shape = getattr(obj, "shape", None)
            if shape is not None:
                dtypes = getattr(obj, "dtypes", None)
                dtype = tuple(str(d) for d in dtypes) if dtypes is not None else str(getattr(obj, "dtype", ""))
                return ("shaped", tuple(shape), dtype)
            if isinstance(obj, (str, bytes)):
                return None
            values = list(obj.values())[: self._FROZEN_SHAPE_SAMPLE] if isinstance(obj, dict) else None
            if values is None and isinstance(obj, (list, tuple)):
                values = list(obj[: self._FROZEN_SHAPE_SAMPLE])
            inner = tuple(len(v) for v in values or () if isinstance(v, Sized))
            return ("sized", len(obj), inner) if isinstance(obj, Sized) else None
        except Exception:  # noqa: BLE001 - no cheap signal is not a failure
            return None

    def _audit_frozen(self, obj: Any) -> bool:
        """Is a frozen=True result still what it was? False once it is not.

        The declaration is trusted, and checked now and then: at the object's
        8th use as an argument and every 64th after that, and at every use
        under CASH_DEBUG. The first check records a baseline -- the content
        hash for a type cash content-hashes, a digest of the pickle otherwise
        -- and each later one compares. On a change, KEY-FROZEN-MUTATED names
        the producer, the object's tag stops being trusted, and it is keyed by
        its content from then on. An object that cannot be pickled cannot be
        audited, and stays trusted.
        """
        key = id(obj)
        entry = self._frozen_uses.get(key)
        if entry is None or entry[0]() is not obj:
            try:
                wref = weakref.ref(obj, lambda _r, k=key, m=self._frozen_uses: m.pop(k, None))
            except TypeError:
                return True
            entry = [wref, 0, None, self._frozen_shape(obj)]
            if len(self._frozen_uses) >= 4096:
                self._frozen_uses.clear()
            self._frozen_uses[key] = entry
        entry[1] += 1
        uses = entry[1]
        shape = self._frozen_shape(obj)
        if entry[3] is not None and shape != entry[3]:
            self._frozen_uses.pop(key, None)
            self._warn_frozen_mutated(obj)
            return False
        due = (
            (self.debug or os.environ.get("CASH_DEBUG"))
            or uses == FROZEN_AUDIT_FIRST
            or (uses > FROZEN_AUDIT_FIRST and uses % FROZEN_AUDIT_EVERY == 0)
        )
        if not due:
            return True
        try:
            digest = builtin_hash(obj)
            if digest is None:
                digest = hashlib.sha256(pickle.dumps(obj)).hexdigest()
        except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
            return True
        if entry[2] is None:
            entry[2] = digest
            return True
        if entry[2] == digest:
            return True
        producer = getattr(obj, "_cash_lineage_producer", None) or "a frozen=True function"
        try:
            obj._cash_lineage_src = LINEAGE_SRC_DECORATOR
        except (AttributeError, TypeError):
            pass
        self._frozen_uses.pop(key, None)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {producer}, which is declared "
            f"@cash.cache(frozen=True), has been modified since it was returned. "
            f"Calls that received it before the change may have been served "
            f"results for the unmodified object; from now on it is keyed by its "
            f"contents.",
            f"take frozen=True off {producer} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )
        return False

    @staticmethod
    def _frame_signature(obj: Any) -> tuple:
        """What must stay the same for a pandas object's content hash to hold.

        Under copy-on-write, a frame whose data another frame also references
        cannot be written in place: every write path (``loc``/``iloc``/``at``,
        column assignment, ``inplace=True`` methods, ``update``, ``insert``,
        ``pop``) first gives the written frame NEW block arrays, and writes
        through ``.values`` / ``to_numpy()`` raise (the arrays are read-only).
        So the identities of the block arrays, the manager and the axes are an
        exact change signal -- measured on 17 mutation forms, pandas 3.0.3. The
        axis NAMES are compared by value, because ``df.index.name = ...``
        renames the same Index object and the content hash includes them.
        """
        mgr = obj._mgr
        blocks = tuple(id(block.values) for block in mgr.blocks)
        if hasattr(obj, "columns"):
            return (id(mgr), blocks, id(obj.columns), tuple(obj.columns.names), id(obj.index), tuple(obj.index.names))
        return (id(mgr), blocks, id(obj.index), tuple(obj.index.names), obj.name)

    @staticmethod
    def _frame_borrows_its_data(obj: Any, held: Any = None) -> bool:
        """Whether *obj*'s blocks sit on memory something else may write.

        Copy-on-write is what makes the block identities an exact change
        signal, and it only governs writes through PANDAS. ``pd.DataFrame(arr,
        copy=False)`` keeps the caller's ndarray, and ``arr[0, 0] = 100`` goes
        straight past pandas: same blocks, changed data. The memo answered 10.0
        where the frame really summed to 109.0 (found attacking the decorator
        before round 26). Such a frame is re-hashed on every call.

        *held* is the memo's own shallow copy of *obj*. Its blocks are views
        whose ``base`` is *obj*'s array, one reference each. Those references
        are cash's, not an outside writer's, so they are not counted against
        the baseline; counting them made every memoised frame look borrowed,
        and it was re-hashed on every call.
        """
        try:
            ours = Cash._held_block_refs(held) if held is not None else {}
            for block in obj._mgr.blocks:
                # Counted before this loop binds the array to a name of its
                # own, exactly as the baseline was measured.
                refcount = Cash._block_refcount(block)
                values = block.values
                base = getattr(values, "base", None)
                if base is not None or not getattr(getattr(values, "flags", None), "owndata", True):
                    return True
                # A 1-D block IS the caller's array (`pd.Series(arr,
                # copy=False)`), with no base and owning its data -- only the
                # extra reference the caller still holds tells them apart. A
                # count above the baseline can only make cash re-hash a frame
                # it could have memoised: slower, never wrong.
                if refcount > Cash._block_refcount_baseline() + ours.get(id(values), 0):
                    return True
                del values, base
        except Exception:  # noqa: BLE001 - a pandas internals change: keep the memo
            return False
        return False

    @staticmethod
    def _held_block_refs(held: Any) -> dict[int, int]:
        """``{id(array): n}``: the references *held*'s blocks keep to arrays.

        A function of its own so that no loop variable outlives it: one left
        pointing at an array would itself be a reference over the baseline.
        """
        refs: dict[int, int] = {}
        for block in held._mgr.blocks:
            values = block.values
            for ref in (values, getattr(values, "base", None)):
                if ref is not None:
                    refs[id(ref)] = refs.get(id(ref), 0) + 1
        return refs

    @staticmethod
    def _block_refcount(block: Any) -> int:
        """``sys.getrefcount`` of *block*'s array, taken the same way for the
        baseline and for every check."""
        return sys.getrefcount(block.values)

    @staticmethod
    def _block_refcount_baseline() -> int:
        """What `_block_refcount` reads for an array only its block holds.

        Measured rather than written down: what ``sys.getrefcount`` counts
        besides the holders varies across Python versions (3.14 counts one
        fewer), and a baseline one too high lets a caller's array through
        as the frame's own -- the stale answer this check exists to stop.
        """
        baseline = Cash._BLOCK_REFCOUNT_BASELINE
        if baseline is None:
            import pandas as pd

            probe = pd.Series([0.0, 1.0, 2.0])
            baseline = Cash._BLOCK_REFCOUNT_BASELINE = Cash._block_refcount(probe._mgr.blocks[0])
        return baseline

    #: See ``_block_refcount_baseline``; anything above it means something
    #: outside can write to the array, see ``_frame_borrows_its_data``.
    _BLOCK_REFCOUNT_BASELINE: int | None = None

    def _frame_memo_lookup(self, obj: Any) -> str | None:
        """The content hash recorded for *obj*, if *obj* has not changed since."""
        entry = self._frame_memo.get(id(obj))
        if entry is None:
            return None
        wref, held, signature, content_hash = entry
        if self._frame_borrows_its_data(obj, held):
            self._frame_memo.pop(id(obj), None)
            return None
        try:
            if wref() is obj and self._frame_signature(obj) == signature:
                return content_hash
        except Exception:  # noqa: BLE001 - a pandas internals change: just re-hash
            pass
        self._frame_memo.pop(id(obj), None)
        return None

    def _frame_memo_store(self, obj: Any, content_hash: str) -> None:
        """Remember *obj*'s content hash, and hold a shallow copy of it.

        The shallow copy shares the data and is what makes the signature
        exact: while cash references the blocks, pandas must copy before any
        write. Cost: the first in-place write to each block afterwards copies
        that block, once. The entry, copy included, goes when *obj* is
        collected, or when the memo fills.
        """
        try:
            held = obj.copy(deep=False)
            signature = self._frame_signature(obj)
            memo = self._frame_memo
            key = id(obj)
            wref = weakref.ref(obj, lambda _ref, key=key, memo=memo: memo.pop(key, None))
        except Exception:  # noqa: BLE001 - the memo is a speedup; hash every time
            return
        if len(self._frame_memo) >= self._FRAME_MEMO_CAP:
            self._frame_memo.clear()
        self._frame_memo[key] = (wref, held, signature, content_hash)

    def _hash_arg_payload(self, args: tuple, kwargs: dict) -> str:
        """Hash one concrete ``(args, kwargs)`` form. May raise on unpicklable
        values; the caller decides whether to retry with a different form."""

        def get_arg_hash(arg):
            # Content-authoritative builtin hashers FIRST. pandas /
            # numpy / polars / pyarrow / modin / dask hash the argument's
            # *content*, which is byte-stable across processes and kernel
            # restarts. The notebook's in-memory ``_cash_lineage_hash`` (checked
            # next) is recomputed per session and is NOT reproducible across a
            # restart -- keying a persisted @cash.cache entry on it makes the
            # decorator miss after a restart even though the argument is
            # byte-identical (re-training the model the docs promise survives a
            # restart). A value that has a content hash must key on content so
            # the entry survives; the modest extra hashing cost is the price of
            # the flagship "restart-and-run-all in seconds" guarantee. Mirrors
            # principle: the reproducible signal, not the volatile
            # in-memory one, is authoritative.
            # Fast path: skip re-hashing a possibly-huge argument we already
            # content-hashed this session, when it is provably the SAME,
            # unmutated object. Keyed on ``id`` (NOT lineage): two *different*
            # objects that happen to share a lineage string must still be
            # distinguished by content -- an explicit invariant
            # (test_arg_hash_restart_stable) -- and distinct live objects have
            # distinct ids. The entry is validated on read by BOTH a weakref
            # identity check (guards id reuse after GC) AND the object's
            # ``_cash_lineage_hash`` being unchanged (cash's own mutation signal,
            # the same one it trusts to cache every notebook statement). The
            # stored value is still the reproducible content hash, so the cache
            # key is byte-identical and restart-safe; the memo is a pure
            # within-session speedup, empty after a restart.
            #
            # Trusted only where something KEEPS it current: the notebook's
            # statement layer re-tags a variable on every assignment and
            # mutation. The decorator also tags what it returns, and nothing
            # ever moves that tag -- in a script, `q.F = 0.03; run(q)` or
            # `df.loc[0, "a"] = 100` left it as it was, and both the memo below
            # and the tag-as-identity shortcut further down served the result
            # for the unmutated object (rounds 17-18).
            # The instance's OWN tag: one inherited from a tagged class made
            # every instance key alike (see cash.lineage_tag).
            lineage = own_tag(arg)
            if lineage is not None:
                src = own_tag(arg, "_cash_lineage_src")
                if src == LINEAGE_SRC_FROZEN:
                    if not self._audit_frozen(arg):
                        lineage = None
                elif src != LINEAGE_SRC_STATEMENT:
                    lineage = None
            if self._frozen_arrays and id(arg) in self._frozen_arrays:
                frozen_hash = self._frozen_array_hash(arg)
                if frozen_hash is not None:
                    return frozen_hash
            if self._frozen_containers and id(arg) in self._frozen_containers:
                frozen_hash = self._frozen_container_hash(arg)
                if frozen_hash is not None:
                    return frozen_hash
            if lineage is not None:
                entry = self._arg_hash_memo.get(id(arg))
                if entry is not None:
                    wref, memo_lineage, content_hash = entry
                    if memo_lineage == lineage and wref() is arg:
                        return content_hash
            # pandas >= 3 copy-on-write: an exact "has this frame changed?"
            # check instead of a trusted tag. See `_frame_memo_lookup`.
            frame_memo = lineage is None and is_cow_pandas(arg)
            if frame_memo:
                content_hash = self._frame_memo_lookup(arg)
                if content_hash is not None:
                    return content_hash

            # Overriding hashers, ahead of everything cash would do itself.
            # The user has said their identity for this type beats content
            # hashing, which is the only way to stop re-reading a 800MB array
            # on every call. Guarded by the emptiness check so the ordinary
            # case pays one dict truth test, not a loop.
            if self._override_hashers:
                for type_, (hasher_fn, src_hash) in self._override_hashers.items():
                    if isinstance(arg, type_):
                        return f"{src_hash}:{hasher_fn(arg)}"

            content_digest = builtin_hash(arg)
            if content_digest is not None:
                if lineage is not None:
                    self._memo_arg_hash(arg, lineage, content_digest)
                elif frame_memo:
                    self._frame_memo_store(arg, content_digest)
                return content_digest
            # Notebook lineage hash: the authoritative, cheap identity for
            # values that carry NO content hasher (custom objects). Kept ahead
            # of registered hashers so a lineage-carrying object short-circuits
            # its (possibly expensive) registered hasher within a session
            # (test_hasher_priority_cash_hash_first).
            if lineage is not None:
                return lineage
            for type_, (hasher_fn, src_hash) in self._type_hashers.items():
                if isinstance(arg, type_):
                    # Embed the hasher source hash so that changing the
                    # hasher's body invalidates dependent cache entries
                    # even when the hasher's output coincidentally matches.
                    return f"{src_hash}:{hasher_fn(arg)}"
            return arg

        # Timed per argument -- two clock reads each -- so that a
        # CACHE-NET-LOSS verdict can name the argument that costs the time.
        costliest: tuple | None = None

        def timed(label: str, value: Any) -> Any:
            nonlocal costliest
            t0 = _perf_counter()
            digest = get_arg_hash(value)
            seconds = _perf_counter() - t0
            if costliest is None or seconds > costliest[1]:
                producer = getattr(value, "_cash_lineage_producer", None)
                if producer is None and self._frozen_arrays and id(value) in self._frozen_arrays:
                    producer = self._frozen_arrays[id(value)][1]
                if producer is None and self._frozen_containers and id(value) in self._frozen_containers:
                    producer = self._frozen_containers[id(value)][1]
                old_pandas = (
                    type(value).__name__ in ("DataFrame", "Series")
                    and (type(value).__module__ or "").startswith("pandas")
                    and not is_cow_pandas(value)
                )
                costliest = (label, seconds, type(value).__name__, producer, old_pandas)
            return digest

        hashed_args = tuple(timed(f"#{i}", a) for i, a in enumerate(args))
        hashed_kwargs = {k: timed(k, v) for k, v in kwargs.items()}
        # An argument with no hasher of its own goes into the payload AS IS,
        # and its cost is the walk and the pickle below, not the lookup timed
        # above -- so CACHE-NET-LOSS named a 2M-row list as taking "about 0ms
        # to hash" (round 19). The payload's time is charged to the largest
        # such argument.
        raw = [
            (label, value)
            for (label, value), digest in zip(
                [(f"#{i}", a) for i, a in enumerate(args)] + list(kwargs.items()),
                list(hashed_args) + list(hashed_kwargs.values()),
            )
            if digest is value and type(value) not in CODELESS_PRIMS
        ]
        payload_t0 = _perf_counter()

        # One canonical form (`stable_key_repr`): sets and dicts in a stable
        # order, every container tagged with its type.
        payload = stable_key_repr(
            (tuple(map(plain_key_part, hashed_args)), {k: plain_key_part(v) for k, v in hashed_kwargs.items()})
        )
        args_bytes = _plain_data.key_dumps(payload)
        if raw:
            payload_seconds = _perf_counter() - payload_t0
            if costliest is None or payload_seconds > costliest[1]:
                label, value = max(raw, key=lambda r: len(r[1]) if hasattr(r[1], "__len__") else sys.getsizeof(r[1]))
                producer = getattr(value, "_cash_lineage_producer", None)
                if producer is None and self._frozen_containers and id(value) in self._frozen_containers:
                    producer = self._frozen_containers[id(value)][1]
                costliest = (label, payload_seconds, type(value).__name__, producer, False)
        ARG_COST.last = costliest
        return hashlib.sha256(args_bytes).hexdigest()

    def _serialize_args(
        self, func_name: str, args: tuple, kwargs: dict, normalized: tuple[tuple, dict] | None = None
    ) -> str | None:
        """Hash the arguments, canonicalised.

        *normalized* lets a caller that has ALREADY canonicalised pass the
        result in rather than have it recomputed. That is not an optimisation:
        the code channel (`_fold_code_args`) and this value channel must key
        off the SAME bound arguments, or `f()` and `f(<the default>)` -- the
        same logical call -- disagree in one channel and split into two cache
        entries. One canonicalisation, shared, is the only way that invariant
        holds by construction rather than by two call sites staying in step.
        """
        if normalized is None:
            normalized = self._normalize_call_args(func_name, args, kwargs)
        try:
            return self._hash_arg_payload(*normalized)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            # Normalization can fold a default value into the payload (so
            # f(1) keys identically to f(1, y=<default>)). If that default is
            # unpicklable it must not make a call that hashed fine before stop
            # caching - retry with the raw, un-normalized form first.
            if normalized[0] is not args or normalized[1] is not kwargs:
                try:
                    return self._hash_arg_payload(args, kwargs)
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    pass
            # Pickle failure here is surfaced via CashCacheIneffectiveWarning in
            # _resolve_cache_key (which sees the None return). Keep this log at
            # debug level so it's available when explicitly enabled but doesn't
            # double-warn.
            logger.debug("Could not serialize arguments for %s: %s", func_name, e)
            return None

    @staticmethod
    def builtin_hashed_family(type_: type) -> str | None:
        """Which built-in content hasher claims *type_*, or ``None``.

        Tells a user at ``register_hasher`` time that the hasher they just
        handed over would never be consulted -- the moment they can still do
        something about it. See `cash.object_hashing.builtin_hash_family`.
        """
        return builtin_hash_family(type_)

    def _compute_with_lock(self, spec: CallSpec, call: Call, compute: Callable[[], Any]) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error.

        Acquiring the lock is best-effort: if *any* backend raises while taking
        it (a Redis ``LockError`` on contention/timeout, a dropped connection,
        an OSError on a file lock), we degrade to an unlocked compute rather than
        crash the user's call. Acquisition, compute, and release are separated so
        a release failure can't re-run the compute, and a compute exception
        propagates normally (it is not mistaken for a lock failure). Under the
        lock the key is looked up again by `_reread`, the same test as the first
        lookup."""
        lock_cm = self.backend.lock(call.cache_key)
        try:
            lock_cm.__enter__()
        except Exception as e:  # noqa: BLE001 - any acquisition failure -> unlocked
            self._warn_lock_failed(spec.func_name, e)
            return compute()
        try:
            hit = self._reread(spec, call)
            if hit is not CACHE_MISS:
                return hit
            return compute()
        finally:
            try:
                lock_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - releasing failed; compute already done
                logger.debug("lock release failed for %s", spec.func_name)

    def _compute_cache_key(self, func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
        return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"

    def _validate_ttl(self, metadata: CacheMetadata | None, ttl: int | None) -> None:
        if metadata and ttl_expired(metadata.timestamp, ttl):
            raise CacheExpiredError("Cache expired")

    @staticmethod
    def _lineage_hash(cache_key: str, auto_file_deps: dict | None) -> str:
        """The lineage hash a result carries downstream.

        It is the producer's ``cache_key`` PLUS a fingerprint of the files the
        producer read. The cache key alone omits file state (files invalidate
        via a freshness re-stat, not via the key), so without this a downstream
        function keyed on the lineage hash would return a STALE result after an
        upstream file changed - the producer recomputes, but its new output
        carries the same lineage hash as the old one. Folding the file deps in
        gives a changed file a distinct lineage. No deps -> unchanged key.

        The fingerprint is built from the recorded content ``hash`` plus the
        size, NOT the mtime. Content is the authoritative freshness
        signal everywhere else, and mtime is the untrustworthy one: keying
        lineage on it would hand a touched-but-identical file a new lineage and
        needlessly recompute every downstream consumer, while a same-size edit
        under an indistinguishable mtime would reuse the old lineage and serve
        stale. A snapshot with no ``hash`` falls back to the mtime so the
        entry still keeps a stable lineage.
        """
        if not auto_file_deps:
            return cache_key
        fp = hashlib.sha256(
            repr(sorted((p, d.get("hash") or d.get("mtime"), d.get("size")) for p, d in auto_file_deps.items())).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"{cache_key}:fdeps:{fp}"

    def _attach_lineage(
        self,
        result: Any,
        cache_key: str,
        auto_file_deps: dict | None = None,
        ttl: int | None = None,
        func_name: str | None = None,
    ) -> None:
        """Attach lineage hash to result if it supports attribute setting.

        Works with pandas DataFrame/Series, polars DataFrame/Series, PyArrow
        Table, modin DataFrame, and any object that allows setting attributes.

        Skipped when the producer has a ``ttl``: a TTL'd value's identity is not
        captured by its cache key (the value changes over time while the key
        stays the same), so a downstream cached function keyed on the lineage
        hash would return a stale result after the upstream's TTL refresh. With
        no lineage hash, the downstream content-hashes the actual current value
        instead - correct, just without the large-value short-circuit.
        """
        if ttl is not None:
            return
        if type(result) in IMMUTABLE_PRIMS:
            # Nothing to attach and nothing worth sparing a hash of: under
            # CASH_DEBUG every int result logged "Cannot attach
            # _cash_lineage_hash to int" (round 19).
            return
        frozen = func_name is not None and func_name in self._frozen_funcs
        if frozen and type(result) in (list, tuple, dict):
            self._remember_frozen_container(result, func_name, self._lineage_hash(cache_key, auto_file_deps))
            return
        if frozen and type(result).__name__ == "ndarray" and (type(result).__module__ or "").startswith("numpy"):
            # An array cannot carry a tag, and read-only is a promise numpy
            # enforces: a write raises instead of going stale.
            try:
                result.flags.writeable = False
                self._frozen_arrays[id(result)] = [
                    weakref.ref(result, lambda _r, k=id(result), m=self._frozen_arrays: m.pop(k, None)),
                    func_name,
                    None,
                ]
            except (AttributeError, TypeError, ValueError):
                pass
            return
        if not frozen and type(result) in UNTAGGABLE_TYPES:
            return
        lineage = self._lineage_hash(cache_key, auto_file_deps)
        try:
            # Say who wrote it: nothing will move this tag when the value is
            # mutated, so `_hash_arg_payload` must not take it for the content
            # -- unless the function was declared frozen=True.
            try:
                result._cash_lineage_src = LINEAGE_SRC_FROZEN if frozen else LINEAGE_SRC_DECORATOR
                if func_name is not None:
                    # Named in CACHE-NET-LOSS and KEY-FROZEN-MUTATED.
                    result._cash_lineage_producer = func_name
            except (AttributeError, TypeError):
                if frozen:
                    self._warn_frozen_has_no_effect(func_name, result)
            type_name = type(result).__name__
            module = type(result).__module__ or ""

            # pandas DataFrame / Series (has attrs dict)
            if module.startswith("pandas") and type_name in ("DataFrame", "Series"):
                result._cash_lineage_hash = lineage
                return

            # polars DataFrame / Series
            if module.startswith("polars") and type_name in ("DataFrame", "Series"):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to polars %s", type_name)
                return

            # modin DataFrame / Series
            if module.startswith("modin") and type_name in ("DataFrame", "Series"):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to modin %s", type_name)
                return

            # PyArrow Table
            if module.startswith("pyarrow") and type_name in ("Table", "RecordBatch"):
                try:
                    result._cash_lineage_hash = lineage
                except (AttributeError, TypeError):
                    logger.debug("Cannot attach _cash_lineage_hash to PyArrow %s", type_name)
                return

            # Generic: try setting on DataFrame-like objects with attrs
            if type_name == "DataFrame" and hasattr(result, "attrs"):
                result._cash_lineage_hash = lineage
                return

            # Generic custom objects: any instance that accepts attribute
            # assignment can carry the lineage hash, so a custom result short-
            # circuits downstream content-hashing the same way a DataFrame does.
            # Builtins (list/dict/tuple/str/numbers) and __slots__ objects with
            # no matching slot reject the assignment - caught below, harmless
            # skip - so those keep content-hashing (a hard Python limitation).
            try:
                result._cash_lineage_hash = lineage
            except (AttributeError, TypeError):
                # Once per type, then never tried again: it logged on every
                # call returning a dict or an array, and meant nothing to the
                # user reading CASH_DEBUG (round 20).
                UNTAGGABLE_TYPES.add(type(result))
                logger.debug(
                    "results of type %s cannot carry a lineage tag, so a cached function taking one hashes its content",
                    type_name,
                )

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
        miss_detail: str = "",
        body_seconds: float | None = None,
        cash_seconds: float | None = None,
        file_deps: dict | None = None,
    ) -> None:
        """Record a decorator call event for notebook integration.

        Thread-safe: uses a lock to protect concurrent appends.
        The notebook ``StatementProcessor`` drains this log after each
        statement execution to include decorator call metrics in the badge;
        it keeps the last ``_CALL_LOG_MAX`` events, since nothing drains it
        outside a notebook. The entry also goes to the running call's
        `CALL_ENTRY` slot, which is what ``cache_info()`` counts.

        ``execution_time`` is the wall-time of *this* operation - a lookup on a
        hit, the compute on a miss. ``time_saved`` is the compute a hit
        *avoided* (the originally-measured execution time stored with the
        cached entry), and 0.0 on a miss. They are distinct: a hit's
        ``execution_time`` is microseconds, but its ``time_saved`` is the full
        compute it stood in for. ``cache_info()['total_time_saved']`` sums the
        latter - summing ``execution_time`` (the old behaviour) under-reported
        savings by orders of magnitude.
        """
        # What cash spent on this call rather than the body: the whole of a
        # hit, and what the miss path measured around a body. Added to the
        # caller's tally when this call is nested in another cached call's
        # body (`NESTED_CASH_SECONDS`).
        if cash_seconds is None:
            cash_seconds = execution_time if cache_hit else 0.0
        nested = NESTED_CASH_SECONDS.get()
        if nested is not None:
            nested[0] += cash_seconds
        entry = {
            "func_name": func_name,
            "cache_hit": cache_hit,
            "execution_time": execution_time,
            "body_seconds": body_seconds,
            "time_saved": time_saved,
            "cash_seconds": cash_seconds,
            "args_hash": args_hash,
            "cache_key": cache_key,
            "timestamp": time.time(),
        }
        outcome: dict[str, Any] = {}
        if not cache_hit:
            if args_hash == "unhashable":
                reason = (MISS_UNHASHABLE, "an argument could not be hashed, so there is no key to look up")
            elif args_hash == "error":
                reason = (MISS_KEY_FAILED, "building the key raised")
            elif args_hash == "unkeyable":
                reason = (MISS_MOCKED, f"{miss_detail}, which has no code to key, so the call ran uncached")
            elif args_hash == "raised":
                reason = (MISS_RAISED, miss_detail)
            else:
                reason = self._pending_miss.pop(cache_key, None) or (MISS_FIRST, "")
            entry["miss_reason"] = reason
            outcome = self._store_outcomes.get(cache_key) or {}
            # Only this call's own outcome. A streamed result is logged before
            # it is stored, and must not borrow the previous call's verdict.
            if outcome.get("at", 0) < entry["timestamp"] - execution_time:
                outcome = {}
            entry["not_persisted"] = outcome.get("not_persisted")
            entry["not_stored"] = outcome.get("not_stored")
        with self._decorator_call_log_lock:
            self._decorator_call_log.append(entry)
        slot = CALL_ENTRY.get()
        if slot is not None:
            slot[0] = entry
        if self._per_call_lines():
            if file_deps:
                # Only for the line: a hit pays nothing for it otherwise.
                entry["sampled_files"] = tuple(path for path, rec in file_deps.items() if is_sampled_dep(rec))
            calls_logger.info("%s", self._describe_call(entry))

    def _per_call_lines(self) -> bool:
        """Is the one-line-per-call log on? Asked for, not merely permitted: an
        application that turned the `cash` logger up to INFO did not ask for a
        line per call."""
        return bool(self.verbose or self.debug or getattr(self.config, "verbose", False)) and calls_logger.isEnabledFor(
            logging.INFO
        )

    @staticmethod
    def _describe_call(entry: dict[str, Any]) -> str:
        """One line for the per-call log: what happened, and on a miss, why."""
        name = entry["func_name"]
        # The id `cash inspect` lists and `cash clear --entry` takes, so a log
        # line can be matched to an entry on disk.
        key = entry.get("cache_key") or ""
        tag = f"  [{entry_id_of(key)}]" if key else ""
        if entry["cache_hit"]:
            saved = entry.get("time_saved") or 0.0
            lookup = entry.get("execution_time") or 0.0
            # What the hit cost, when it is not small: the summary said "time
            # saved" while warm runs were 9x slower than uncached (round 19).
            if lookup >= 0.01 and lookup >= 0.1 * saved:
                verdict = "; a net loss" if lookup > saved else ""
                line = f"HIT  {name}{tag}  (saved {saved:.2f}s; the lookup took {lookup:.2f}s{verdict})"
            else:
                line = f"HIT  {name}{tag}  (saved {saved:.2f}s)"
            sampled = entry.get("sampled_files")
            if sampled:
                # Larger than file_hash_full_max_bytes: the HIT rests on the
                # timestamps, and "when it does not recompute I need to be sure
                # it was right not to" had no way to see that (round 20).
                shown = ", ".join(os.path.basename(p) for p in sampled[:3])
                more = f" and {len(sampled) - 3} more" if len(sampled) > 3 else ""
                line += f"  -- trusts the timestamps of {shown}{more} (sampled: larger than file_hash_full_max_bytes)"
            return line
        kind, detail = entry.get("miss_reason") or (MISS_FIRST, "")
        if kind == MISS_RAISED:
            return f"RAISE {name}  {detail}; nothing stored  (ran {entry['execution_time']:.2f}s)"
        line = f"MISS {name}{tag}  {kind}" + (f": {detail}" if detail else "")
        # The body's own time: the persistence floor named beside it is judged
        # on that, and the call's time -- key, analysis, lookup -- made "ran
        # 0.20s ... under the 0.1s floor" read as a contradiction (round 19).
        ran = entry.get("body_seconds")
        line += f"  (ran {entry['execution_time'] if ran is None else ran:.2f}s"
        if entry.get("not_stored"):
            line += f"; not stored: {entry['not_stored']}"
        elif entry.get("not_persisted"):
            line += f"; kept in RAM only -- {entry['not_persisted']} -- so another process will recompute it"
        return line + ")"

    def _log_raised(self, func_name: str, exc: BaseException, call_start: float) -> None:
        """Record a call whose body raised: nothing is stored, and it counts.

        Such a call produced no line at all, and a run that crashed half-way
        summarised as "5 of 5 calls restored" (round 18).
        """
        self._log_decorator_call(
            func_name,
            cache_hit=False,
            execution_time=_perf_counter() - call_start,
            args_hash="raised",
            cache_key="",
            miss_detail=f"{type(exc).__name__}: {str(exc)[:80]}",
        )

    def _warn_cache_if_raised(
        self,
        func_name: str,
        error: BaseException,
        *,
        stacklevel: int | None = None,
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
            f"{type(error).__name__} ({error}), so the result is returned "
            f"un-cached and every later call recomputes.",
            code="CACHE-IF-RAISED",
            fix="make the predicate total -- it must handle every shape the "
            "result can take -- or drop cache_if= to restore caching.",
            stacklevel=stacklevel,
        )

    def _warn_metadata_invalid(
        self,
        func_name: str,
        error: BaseException,
        *,
        stacklevel: int | None = None,
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
            f"could not be validated ({type(error).__name__}: {error}), so "
            f"cash treated the entry as absent and recomputed.",
            code="STORE-METADATA-INVALID",
            fix="nothing, for a one-off; if it keeps appearing, run "
            "f.cache_clear() so the unreadable records are replaced.",
            stacklevel=stacklevel,
        )

    def _warn_lock_failed(
        self,
        func_name: str,
        error: BaseException,
        *,
        stacklevel: int | None = None,
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
            f"({type(error).__name__}: {error}), so cash proceeded without the "
            f"lock and concurrent calls with the same args may compute "
            f"redundantly.",
            code="STORE-LOCK-FAILED",
            fix="investigate the backend the exception names -- a full disk, a "
            "stale lock file, or a cache_dir on a filesystem where locking "
            "does not work.",
            stacklevel=stacklevel,
        )

    def _code_functions(self, func: Callable, func_name: str) -> list[Any]:
        """The functions whose code a call of *func_name* runs, as far as cash
        follows it: its own, its helpers', and those of the cached functions it
        depends on, transitively."""
        found: list[Any] = []
        seen_names: set[str] = set()
        stack: list[tuple[str, Any]] = [(func_name, func)]
        while stack:
            name, fn = stack.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            if fn is not None:
                found.append(fn)
            report = self._purity_reports.get(name)
            if report is not None:
                for ref in report.helper_objects.values():
                    helper = ref()
                    if helper is not None:
                        found.append(helper)
                for module_name, chain in report.helper_resolution_paths.values():
                    target = resolve_binding(module_name, chain)
                    if callable(target):
                        found.append(target)
            for dep in self.graph.get_dependencies(name):
                if dep in self.functions and dep not in seen_names:
                    stack.append((dep, self.functions[dep]))
        return found

    def _credit_remembered_reads(self, func_name: str, tracker: Any, args: tuple, kwargs: dict) -> None:
        """Add the files a helper read in an EARLIER call to this call's inputs.

        A parse memoised with ``functools.lru_cache`` or a module dict: the
        first cached consumer read the file and recorded it; the second got
        the memoised rows, read nothing, and stored ``file_deps: None`` -- so
        after the file changed it kept serving the old total (round 19).

        For each function this call's code reaches that did NOT read a file in
        this call, its remembered files are added (`credited_reads`). A memo
        keyed by a path the call was given (``parse(path)``) adds only that
        path when it is among them; a memo of a fixed file adds what it read.
        The cached function's own history is left out -- it is per argument --
        and so is a function that read too many files to attribute.

        A remembered read also says which version of the file it was. When the
        file has changed since, the memo handed this call the OLD version's
        data -- right for this process until it refills, but not an answer
        for the file as it is now, which is what the entry would be stored
        against (round 20). Such a path goes into ``stale_memo_reads``, and the
        store is refused.
        """

        func = self.functions.get(func_name)
        if func is None or tracker is None:
            return
        live = getattr(tracker, "reading_codes", set())
        own = getattr(func, "__code__", None)
        have = tracker.get_accessed_files()
        arg_paths: set[str] | None = None
        for fn in self._code_functions(func, func_name):
            code = getattr(fn, "__code__", None)
            if code is None or code is own or code in live:
                continue
            remembered = credited_reads(code)
            if not remembered or remembered.keys() <= have:
                continue
            if arg_paths is None:
                arg_paths = self._argument_paths(args, kwargs)
            # Chosen BEFORE what is already tracked is taken away: `have` grows
            # as files are added, and a remainder that misses the arguments
            # would read as a memo of a fixed file.
            chosen = (remembered.keys() & arg_paths) or set(remembered)
            for path in sorted(chosen - have):
                tracker.add_tracked(path)
                then = remembered[path]
                if then is not None and tracker.read_stats.get(path, then) != then:
                    tracker.stale_memo_reads.add(path)

    @staticmethod
    def _argument_paths(args: tuple, kwargs: dict) -> set[str]:
        """The resolved paths among a call's arguments, one container deep."""

        values: list[Any] = [*args, *kwargs.values()]
        for value in list(values):
            if isinstance(value, (list, tuple)) and len(value) <= 64:
                values.extend(value)
        found: set[str] = set()
        for value in values:
            if isinstance(value, os.PathLike) or (
                isinstance(value, str) and 0 < len(value) < 1024 and "\n" not in value
            ):
                try:
                    found.add(normalize_path(os.path.realpath(os.fspath(value))))
                except (TypeError, ValueError, OSError):
                    continue
        return found

    def _code_moved_since_keyed(self, func: Callable, func_name: str) -> bool:
        """Did a file this call's code came from change after its key was read?

        A function is keyed by the text of its file, read once per process;
        the code that runs is what the process loaded -- or, for a worker a
        pool starts during the call, whatever the file holds THEN. Edited in
        between, the result of one version was stored under the other's key,
        and a later process running the first version was served the second
        one's numbers (round 19: a helper edited while a pooled call ran, and a
        deploy that replaced a helper under a running job). Nothing can say
        which version the result came from, so it is returned and not stored.

        Only THIS code's text counts: a file edited elsewhere -- another
        function, a comment -- runs the same code in a new worker, and the
        entry is still right.

        One ``os.stat`` per code file, on a miss only; the text is re-read only
        for a file that moved.
        """
        stats: dict[str, tuple[int, int] | None] = {}
        moved: list[str] = []
        for fn in self._code_functions(func, func_name):
            code = getattr(fn, "__code__", None)
            rec = CODE_KEYED_STATS.get(id(code)) if code is not None else None
            if rec is None or rec[0] is not code:
                continue
            path = rec[1]
            if path not in stats:
                try:
                    st = os.stat(path)
                    stats[path] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    stats[path] = None
            now = stats[path]
            if now is None or now == (rec[2], rec[3]) or path in moved:
                continue
            # None -- the function is gone from the file -- is not the same text.
            same_text = own_source_digest(fn) == rec[4]
            if same_text:
                CODE_KEYED_STATS[id(code)] = (code, path, *now, rec[4])
            else:
                moved.append(path)
        if not moved:
            return False
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        self._warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "code_changed",
            f"@cash.cache on {func_name}: {shown} changed on disk after this "
            f"process read the code it keys {func_name} by. The result was "
            f"returned but not cached: a worker process started now runs the "
            f"file's new code, this process runs the old, and nothing can say "
            f"which one produced it.",
            code="STORE-CODE-CHANGED",
            fix="restart the process to run -- and cache -- the new code. A "
            "deploy that replaces files under a running job opens this window.",
        )
        return True

    def _inputs_moved_during_call(self, func_name: str, tracker: Any) -> bool:
        """Did a file this call read change before the call returned?

        The entry's file fingerprints are taken when it is STORED. A file
        rewritten after the body read it but before it returned was
        fingerprinted in its new state, so the entry matched the new file
        and served the old answer on every later call (CAS-109, round 17:
        a sync job overlapping a long pipeline; and, one level up, an outer
        aggregate re-fingerprinting a file its inner call had already read).
        The documented mitigation -- write to a temp file and rename -- did
        not help, because the rename lands before the store.

        The result is still returned: it is what the body computed. It is
        only not cached, because nothing can say which content it came from.
        """
        moved_fn = getattr(tracker, "inputs_changed_since_read", None)
        if moved_fn is None:
            return False
        try:
            moved = moved_fn()
        except Exception:  # noqa: BLE001 - never let the check break a call
            return False
        if not moved:
            return False
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        self._warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "input_changed",
            f"@cash.cache on {func_name}: {shown} changed while the call was "
            f"running, after it had been read. The result was returned but not "
            f"cached, because it cannot be told which version of the file it "
            f"was computed from.",
            code="STORE-INPUT-CHANGED",
            fix="nothing, if something else writes these files while this runs "
            "-- the next call reads the settled file and caches normally. If "
            "the function writes a file it also reads, that is why: split the "
            "read and the write.",
        )
        return True

    def _warn_unseeded_randomness(
        self,
        func: Callable,
        func_name: str,
        allow_random: bool,
    ) -> None:
        """Warn once if *func*'s source draws from an unseeded RNG.

        The decorator used to be completely silent here while the notebook path
        warned, so ``@cash.cache`` would freeze a non-deterministic result
        forever with nothing on screen to say so. The two paths now share ONE
        detector — :class:`~cash.tracking.randomness.RandomnessDetector`, reused
        verbatim — so "what counts as unseeded" cannot drift between them.

        Runs at DECORATION time, once per function. The analysis is a pure
        function of the source, so there is no reason to pay for it per call,
        and ``cache()`` already reads the source anyway (``_register_func`` ->
        ``callable_identity``), which warms ``linecache`` for us.

        A fresh detector is used per function rather than one shared across the
        instance. The detector's seed-tracking is *session*-scoped, which is
        right for a notebook (cells run top-to-bottom in one namespace) but
        wrong here: decoration order is not call order, so letting a
        ``np.random.seed(0)`` inside function A silence function B would be
        unsound. Per-function analysis keeps the verdict a property of the
        source we are actually looking at.

        Silent when:

        * ``allow_random=True``, or the notebook's ``# @cash:allow-random``
          appears in the function's own source (same directive vocabulary,
          parsed by the same ``parse_annotation_line``);
        * the RNG is seeded — the whole point, and the reason a seeded draw
          must not be flagged;
        * the source cannot be read (``exec``/REPL-defined functions). The
          purity analyzer has the identical blind spot and treats it the same
          way: no source, no claim.
        """
        if allow_random:
            return

        try:
            src_lines, first_lineno = inspect.getsourcelines(func)
        except SOURCE_RETRIEVAL_ERRORS:
            # No retrievable source (exec'd, REPL, C function). Staying silent
            # is the conservative choice: we cannot see a draw, so we cannot
            # honestly claim there is one.
            return
        src = textwrap.dedent("".join(src_lines))

        # Honour the notebook's in-source opt-out too. Users coming from
        # ``%cash_on`` reach for the comment, and the source is already in hand.
        for line in src.splitlines():
            ann = parse_annotation_line(line)
            if ann is not None and ann.allow_random:
                return

        # A seeding call fed by a PARAMETER -- `default_rng(seed)` -- counts as
        # seeded to the detector, but whether it is depends on the call:
        # `def simulate(params, seed=None)` draws from OS entropy whenever the
        # caller leaves the seed out, and R Monte Carlo replicates came back
        # identical with nothing said (CAS-116). Note which parameters, and
        # check their bound value per call.
        seed_params = seed_parameters(src)
        if seed_params:
            self._seed_params[func_name] = seed_params

        try:
            unseeded, _messages, _has_seed = RandomnessDetector().analyze_code(src)
        except Exception:  # pragma: no cover - detector must never break caching
            logger.debug("randomness scan failed for %s", func_name, exc_info=True)
            return

        if not unseeded:
            return

        call = unseeded[0]
        extra = ""
        if len(unseeded) > 1:
            extra = f" (+{len(unseeded) - 1} more unseeded call(s) in this function)"
        # ``call.lineno`` is relative to the source we handed the detector, which
        # starts at the function's first line. Rebase it onto the file so the
        # number in the message matches what the user's editor shows.
        # ``getsourcelines`` returns 0 for sources it cannot place; keep the
        # relative number rather than reporting a nonsense negative line.
        abs_lineno = call.lineno + first_lineno - 1 if first_lineno else call.lineno

        # ASCII only: this lands in a terminal whose codepage may not be UTF-8.
        message = (
            f"@cash.cache on {func_name}: Unseeded randomness detected: "
            f"{describe_random_call(call)} at line {abs_lineno}{extra}. "
            f"The first call's result is cached and replayed on every later "
            f"call - the RNG is never consulted again, so the value is frozen "
            f"and not reproducible across a cleared cache."
        )
        # ``_warn_once`` keys on (category, func_name, "") -> one warning per
        # decorated function for the life of this Cash instance, and it also
        # files the message into ``f.cache_info()['warnings']`` so it stays
        # discoverable if the user missed the stderr emission.
        self._warn_once(
            CashRandomnessWarning,
            func_name,
            "",
            message,
            code="RANDOM-UNSEEDED",
            fix="seed the RNG to make the value reproducible, leave the "
            "function undecorated for a genuinely fresh draw, or pass "
            "@cash.cache(allow_random=True) to keep it frozen on purpose.",
            # 4, not 3: the chain from ``warnings.warn`` is
            # ``warn_diagnostic_message -> _warn_once ->
            # _warn_unseeded_randomness -> cache -> user``, so 3 blamed
            # ``cache`` itself and printed a line inside core.py. Measured
            # against a decoration on a known line; a reader whose warning
            # points into Cash cannot act on it, which is the whole point of
            # this diagnostic. Single caller, so the depth is fixed.,
        )

    def _warn_if_seed_is_none(self, func: Callable, func_name: str, args: tuple, kwargs: dict) -> None:
        """RANDOM-UNSEEDED for a seed that is None in THIS call.

        The seed may be a parameter (CAS-116) or read from one or from a module
        global: ``default_rng(settings.seed)`` with the field None froze one
        draw across processes and said nothing (round 18), while the bare
        ``seed=None`` parameter warned.
        """
        bound = None
        g = getattr(func, "__globals__", None) or {}
        for expr, (call, root, is_param, path) in sorted(self._seed_params.get(func_name, {}).items()):
            if is_param:
                if bound is None:
                    try:
                        bound = inspect.signature(func).bind(*args, **kwargs)
                        bound.apply_defaults()
                    except (TypeError, ValueError):
                        return
                if root not in bound.arguments:
                    continue
                value = read_seed(bound.arguments[root], path)
                origin = f"the parameter '{root}'" if not path else f"'{expr}'"
            else:
                if root not in g:
                    continue
                value = read_seed(g[root], path)
                origin = f"'{expr}'"
            if value is not None:
                continue
            fix = (
                f"pass a seed: {root}=i per replicate keeps each one reproducible and cacheable."
                if is_param and not path
                else f"set {expr} to an integer, or pass the seed as an argument."
            )
            self._warn_once(
                CashRandomnessWarning,
                func_name,
                f"seed-param:{expr}",
                f"@cash.cache on {func_name}: {call} is seeded from "
                f"{origin}, which is None in this call, so the RNG "
                f"draws from OS entropy. The first call's result is cached and "
                f"replayed on every later call with the same arguments -- "
                f"repeated calls return the same 'random' value, and it is "
                f"not reproducible across a cleared cache.",
                code="RANDOM-UNSEEDED",
                fix=f"{fix} Or @cash.cache(allow_random=True) to keep the value frozen on purpose.",
            )
            return

    def _warn_unseeded_estimator_result(
        self,
        func_name: str,
        result: Any,
        allow_random: bool,
    ) -> None:
        """Warn when a cached function RETURNS an unseeded fitted estimator.

        ``_warn_unseeded_randomness`` reads the source, and
        ``decorator.md`` is right that this hazard is invisible to it:
        randomness inside sklearn's compiled ``.fit()`` is not in any AST. The
        notebook's statement path solves that by asking the LIVE object
        (``get_params()['random_state'] is None``) rather than the source; the
        decorator path had no equivalent, so the recommended way to cache a fit
        was also the silent one.

        Reported in round 14: three runs returned the identical model (first
        tree's `random_state` 1200527474), no warning, no badge marker, using
        the docs' own recipe. The tester's words for the harm are the reason
        this exists -- "I would have written 'the model is completely stable
        across random seeds' in a report."

        Same verdict rule as ``_unseeded_estimator_fits``: unseeded iff
        ``get_params()`` HAS ``random_state`` and it is ``None``. A seed of any
        kind, or no such parameter at all (``LinearRegression``), is silent.
        Any failure is silent too -- an advisory must never break a call.
        """
        if allow_random:
            return
        # This runs on EVERY call, hits included, so it must stay cheap once it
        # has had its say. `_warn_once` would dedupe the emission but not the
        # `get_params()` that precedes it, and sklearn's `get_params` walks the
        # signature -- a per-hit cost on exactly the functions people cache to
        # avoid paying for a fit. Check the same key first and leave.
        if (CashRandomnessWarning, func_name, "_estimator_result") in self._warning_keys_seen:
            return
        get_params = getattr(result, "get_params", None)
        if get_params is None or not callable(get_params):
            return
        try:
            params = get_params()
            if params.get("random_state", "absent") is not None:
                return
        except Exception:  # noqa: BLE001 - advisory only; never break a call
            return

        self._warn_once(
            CashRandomnessWarning,
            func_name,
            "_estimator_result",
            f"@cash.cache on {func_name}: returns a fitted estimator with "
            f"random_state=None. cash caches it, so every later call replays "
            f"that one fit - the model is frozen, not stable. Two genuine fits "
            f"would differ, and comparing runs cannot tell you otherwise.",
            code="RANDOM-UNSEEDED",
            fix="pass random_state=<int> to the estimator for a reproducible "
            "fit, leave the function undecorated for a genuinely fresh "
            "one, or pass @cash.cache(allow_random=True) to keep it frozen "
            "on purpose.",
        )

    def _warn_once(
        self,
        category: type[Warning],
        func_name: str,
        arg_type_name: str,
        message: str,
        *,
        code: str,
        fix: str,
        stacklevel: int | None = None,
        once_per_version: bool = False,
    ) -> None:
        """Emit a coded diagnostic at most once per
        ``(category, func_name, arg_type_name)`` for this Cash instance.

        ``once_per_version``: and once per CACHE for the same text -- which
        names the lines and the code it found them in -- so a later process
        records it in ``cache_info()['warnings']`` without printing it. For
        the static findings a source reading makes, which were the same 32
        lines in a nightly job's log every night (round 20); an edit that
        changes what they say shows them again.

        ``message`` is one sentence of *what happened*; ``fix`` is one
        imperative sentence; ``code`` is the diagnostic code from
        ``cash.diagnostics`` that names the section of ``docs/warnings.md``
        expanding both. The three are rendered together by
        :func:`~cash.diagnostics.format_diagnostic`, and the rendered text is
        what reaches BOTH stderr and ``cache_info()['warnings']`` -- the log
        and the terminal must not drift apart, since the log is where people
        look once the stderr line has scrolled away.

        ``arg_type_name`` is the empty string for warnings that do not
        attach to a specific arg type (e.g. store-failed). The seen-set
        key still distinguishes by func_name.

        **Do not pass ``stacklevel``.** The blamed frame is resolved at emit
        time by walking out to the nearest frame outside ``cash/`` -- see
        :func:`~cash.diagnostics._stacklevel_of_first_user_frame`. This used to
        be a per-caller constant, documented here as 5 by default with 6 and 3
        for the deeper and shallower chains, and four separate diagnostics
        shipped pointing at a line inside ``core.py`` anyway. A constant cannot
        be right for a helper reached at two different depths, and an over-deep
        one reports ``<sys>:0`` rather than clamping, so the failure was silent
        in both directions. The parameter survives only as an override for a
        site that needs one; none does.
        """
        if _EXPLAINING.get():
            return
        rendered = format_diagnostic(code, message, fix)  # raises on a bad code
        key = (category, func_name, arg_type_name)
        with self._decorator_call_log_lock:
            if key in self._warning_keys_seen:
                return
            self._warning_keys_seen.add(key)
            # Also record in per-function rolling log so the warning is
            # discoverable after the fact via ``f.cache_info()['warnings']``
            # - even if the user missed the stderr emission. The code goes in
            # as its own field as well as inside the text, so a reader of the
            # log can branch on it the way a warning handler branches on
            # ``w.message.code``.
            entry = {
                "category": category.__name__,
                "code": code,
                "message": rendered,
                "timestamp": time.time(),
            }
            log = self._func_warnings.setdefault(func_name, [])
            log.append(entry)
            if len(log) > self._func_warnings_max:
                del log[: len(log) - self._func_warnings_max]
        if once_per_version and not self._first_showing(func_name, rendered):
            entry["shown_by_an_earlier_run"] = True
            return
        warn_diagnostic_message(
            category, code, rendered, stacklevel=stacklevel, fallback=self._definition_site(func_name)
        )

    def _first_showing(self, func_name: str, rendered: str) -> bool:
        """Has no earlier run on this cache shown *rendered*? Records that one
        has. True whenever the cache keeps no record -- when in doubt, show."""
        try:
            self.backend  # the first call is about to build it for its lookup anyway
        except Exception:  # noqa: BLE001 - no backend, no record: show it
            return True
        if self._stored_keys_path(func_name) is None:
            return True
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        if digest in self._stored_doc(func_name).get("warned", {}):
            return False
        # Written with the record's next write, never now: this runs before
        # the first call's lookup, and creating (and stamping) the cache
        # directory here reordered the stamps the clear check reads -- a
        # clear under a process that started cold went unnoticed on Linux.
        with self._ram_only_lock:
            self._warned_pending.setdefault(func_name, {})[digest] = time.time()
        return True

    def _definition_site(self, func_name: str) -> tuple[str, int] | None:
        """Where *func_name* is defined: what a warning blames when the call
        runs on a pool thread, whose stack holds nothing of the user's."""
        fn = self.functions.get(func_name)
        try:
            code = getattr(inspect.unwrap(fn), "__code__", None) if fn is not None else None
        except ValueError:  # a wrapper chain that loops
            return None
        return (code.co_filename, code.co_firstlineno) if code is not None else None

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

    #: Types whose code must not participate in any cache key. Process-wide,
    #: not per-instance: a marker is a property of the type, and a user who
    #: marks it once should not have to repeat it per Cash instance.
    #:
    #: Holds STRONG references deliberately, so a registered class can never
    #: be garbage collected. Considered and rejected a WeakSet: opaque types
    #: are registered by hand, at import time, in the tens at most for any
    #: real user -- not generated in volume -- so the leak this trades away
    #: has no realistic scale to bite at. A WeakSet would also silently
    #: un-register a type the moment nothing else references it, which is
    #: the opposite of "mark it once and forget about it."
    _OPAQUE_TYPES: set = set()

    @staticmethod
    def mark_opaque(*types_: type) -> None:
        """Exclude *types_* from code-surface hashing: what ``cash.opaque`` records."""
        Cash._OPAQUE_TYPES.update(types_)

    @staticmethod
    def _is_opaque(obj: Any) -> bool:
        """True when *obj* -- a class, or an instance of one -- must not have
        its code hashed into a cache key.

        The type itself must be in ``_OPAQUE_TYPES`` (``cash.opaque``); a
        subclass of an opaque class is not covered. It may carry its own
        freshly-written methods the user actively edits, and inheriting the
        mark would silently exempt that code from ever invalidating the cache.
        A subclass that wants the same treatment is marked itself (pinned by
        ``test_a_subclass_of_an_opaque_class_does_not_inherit_opacity``).

        Never raises. Measured, not assumed: a metaclass that defines
        ``__eq__`` without ``__hash__`` makes the CLASS ITSELF unhashable
        (Python's data-model default, not just its instances), so
        ``target in Cash._OPAQUE_TYPES`` can raise ``TypeError`` on a real,
        if unusual, class shape. An opacity check must not be the thing
        that breaks an otherwise-cacheable call.
        """
        try:
            if isinstance(obj, functools.partial):
                # A partial is the function it wraps plus arguments, both of
                # which are keyed now. `cash.opaque(functools.partial)` was the
                # old advice for silencing KEY-OPAQUE-CALLABLE, and it silenced
                # EVERY partial in the process, including ones over code the
                # user then edited (round 18).
                return False
            target = obj if isinstance(obj, type) else type(obj)
            return target in Cash._OPAQUE_TYPES
        except Exception as e:  # noqa: BLE001 - opacity check must never break a call
            logger.debug("[CORE] opacity check failed for %r: %s", obj, e)
            return False

    def _learn_mutating_captures(self, func: Callable, func_name: str, watched: dict[str, tuple[str, str]]) -> None:
        """Demote any provisional global this call was OBSERVED to mutate.

        A global merely *passed to a call* (`sum(G)`, `model.predict(G)`) used to
        be dropped from the key outright, on the theory that the callee might
        mutate it. That silently served stale values forever (CAS-270). Those
        names are folded now, and confirmed here: hash them again once the body
        has run and compare against the hash the key already needed.

        Changed across the call => calling this function is what moves the value,
        so folding it would key the entry on the function's own output and miss
        forever. Stop folding that ONE name; the function keeps caching on
        everything else.

        Two things worth knowing:

        * The entry just written stays valid -- it is keyed on the PRE-call
          state, which is what produced it. The next call keys without this
          name, misses once, and thereafter behaves as it did before CAS-270.
        * A change *between* calls (`G = [...]` anywhere) is invisible to this
          window by construction, which is correct: that is precisely what
          folding is for, and it needs no detection.

        Only runs on a miss -- the body has to execute for there to be anything
        to observe -- so the cost is one hash on the path that just paid for a
        real computation.
        """
        if not watched:
            return
        code = getattr(func, "__code__", None)
        if code is None:
            return
        own_globals = getattr(func, "__globals__", None)
        cells = dict(zip(getattr(code, "co_freevars", ()) or (), getattr(func, "__closure__", ()) or ()))
        for name, (before, scope, owner, reader) in watched.items():
            try:
                if scope == "closure":
                    cell = cells.get(name)
                    if cell is None:
                        continue
                    after = self._hash_arg_payload((cell.cell_contents,), {})
                elif scope == "carrier":
                    mapping, key = owner
                    if key not in mapping:
                        continue
                    after = self._carried_global_hash(mapping[key], getattr(func, "__module__", None))
                else:
                    # The mapping the BEFORE hash came from -- a helper's
                    # module, when this entry was folded on a helper's behalf.
                    g = owner if isinstance(owner, dict) else own_globals
                    if not isinstance(g, dict) or name not in g:
                        continue
                    after = self._hash_arg_payload(
                        (self._stabilize_for_global_hash(g[name], self._data_callable_identity),), {}
                    )
            except Exception:  # noqa: BLE001 - unhashable NOW; treat as unchanged
                continue
            if after == before:
                continue
            if scope == "carrier":
                # The library's own state (a generator advanced, a cache
                # filled), not a mutation the user wrote: stop folding it.
                self._mutating_globals.setdefault((code, "global"), set()).add(name)
                logger.debug("[CORE] %s: stopped keying what %s carries; calling it changes it", func_name, name)
                continue
            self._mutating_globals.setdefault((code, scope), set()).add(name)
            if scope == "closure":
                where = f"variable it captures '{name}'"
            else:
                module = owner.get("__name__") if isinstance(owner, dict) else None
                if module in MAIN_MODULE_NAMES and reader is not None:
                    module = resolve_main_module(reader)
                where = f"module global '{module}.{name}'" if module else f"module global '{name}'"
            site = Cash._describe_scope_use(reader, name, func)
            self._warn_once(
                CashImpurityWarning,
                func_name,
                name,
                f"@cash.cache on {func_name}: calling it modifies the {where}"
                f"{site}, so '{name}' can no longer be tracked for "
                f"invalidation and a cache hit will not repeat that change.",
                code="IMPURE-SCOPE-MUTATION",
                fix="pass the value in as an argument and return the new one, "
                "instead of reaching out and rewriting it; or, if a cache hit "
                "may skip the change (a bill, a log), put "
                "`# @cash:assume-safe` on the line named.",
            )

    @staticmethod
    def _describe_scope_use(reader: Any, name: str, cached: Any) -> str:
        """`` -- through `X.f()` in mod.helper (file:line)`` for the warning, or ``""``.

        *reader* is the function whose read of *name* was watched: the cached
        function, or a helper several calls below it. The move itself may be
        deeper still (a method of the object), but this is the line in code the
        user wrote that reaches it -- the first use that may mutate, the same
        rule that made the name provisional. Only runs when warning.
        """
        if reader is None:
            return ""
        try:
            lines, first = inspect.getsourcelines(reader)
            filename = inspect.getsourcefile(reader) or inspect.getfile(reader)
            source = textwrap.dedent("".join(lines))
            tree = ast.parse(source)
        except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError):
            return ""
        best = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Call, ast.Assign, ast.AugAssign, ast.Delete)):
                continue
            if name not in Cash._unsafe_uses_of(node, {name}):
                continue
            if best is None or (node.lineno, node.col_offset) < (best.lineno, best.col_offset):
                best = node
        if best is None:
            return ""
        lineno = max(first, 1) - 1 + best.lineno
        text = " ".join((ast.get_source_segment(source, best) or "").split())
        if len(text) > 80:
            text = text[:77] + "..."
        inside = "" if reader is cached else f" in {Cash.get_func_key(reader)}"
        return f" -- through `{text}`{inside} ({filename}:{lineno})"

    def _refuses_identity_coupled(self, func_name: str, result: Any) -> bool:
        """True when *result* must never be stored, because storing it would
        detach a library's global registry from the object the caller holds.

        The statement path (``statement/processor.py``) and call interception
        (``call_unit.py``) have gated on ``identity_coupled_reason`` for a
        while; the decorator did not.  So ``@cash.cache`` on a function
        returning a ``Figure`` hijacked ``plt.gcf()`` -- on the FIRST call,
        during the *store*, because the RAM tier deep-copies and
        ``Figure.__setstate__`` re-registers the copy as pyplot's current
        figure.  The user then draws on their figure while ``plt.savefig()``
        writes the cache's private snapshot (CAS-245).

        Checked here rather than inside ``_store_in_cache`` so the refusal
        lands beside ``cache_if``, BEFORE ``_attach_lineage``: a value that is
        not stored must not carry a lineage hash pointing at an entry that was
        never written.

        KNOWN BOUNDARY: called at all four store sites (sync/async x
        non-iterator/single-chunk), which is every site where the value is in
        hand before anything is written.  A *multi*-chunk iterator is not
        covered -- ``_stream_and_store`` has already written earlier chunks by the
        time any item could be inspected, so gating there would mean aborting
        mid-write and reclaiming them.  Reaching it needs a generator yielding
        enough Figures to cross ``chunk_max_bytes`` (or a million of them),
        which no reported case comes near.  Widen this if one ever does.
        """

        # ``func_name`` is already in the message prefix, so name the slot
        # rather than repeating the qualified path inside the reason.
        reason = identity_coupled_reason("the returned value", result)
        if reason is None:
            return False
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: result not cached. {reason}",
            code="CACHE-IDENTITY-COUPLED",
            fix="split the function: cache the part that computes the numbers, "
            "and draw the figure from them in an uncached function.",
        )
        return True

    def _note_arg_cost(self, func_name: str) -> None:
        """Keep the costliest argument to hash seen for *func_name*.

        Only its description is kept -- parameter, type, seconds, the cached
        function that produced it -- never the value, which may be large.
        """
        cost = getattr(ARG_COST, "last", None)
        ARG_COST.last = None
        if cost is None:
            return
        label, seconds, type_name, producer, old_pandas = cost
        known = self._arg_costs.get(func_name)
        if known is not None and known[2] >= seconds:
            return
        if known is None and len(self._arg_costs) >= 1024:
            return
        self._arg_costs[func_name] = (label, type_name, seconds, producer, old_pandas)

    def _note_effectiveness(
        self,
        func_name: str,
        overhead_seconds: float,
        *,
        body_seconds: float | None,
        was_hit: bool,
    ) -> None:
        """Account for one call and warn if caching has become a net loss.

        Informational only: the decorator caches because the user asked it
        to, and deciding otherwise is the notebook cost model's job, not
        this one. See ``cash.effectiveness`` for when it speaks up.

        ``overhead_seconds`` is everything cash did around the body: the key and
        lookup, and on a miss the store as well. The store used to be left out
        as a once-per-key cost, but a result kept in RAM only is copied in
        every process that computes it, and a 2.5M-row parse cost 4x its body
        that way with nothing reporting it (round 20).
        """
        culprit = self._arg_costs.get(func_name)
        if culprit is not None and culprit[3] and culprit[3] in self._frozen_funcs:
            # Already frozen: advising frozen=True on it (round 20) is noise.
            culprit = (*culprit[:3], None, *culprit[4:])
        try:
            verdict = self._effectiveness.record(
                func_name,
                overhead_seconds=overhead_seconds,
                body_seconds=body_seconds,
                was_hit=was_hit,
                culprit=culprit,
            )
        except Exception:  # noqa: BLE001 - accounting must never break a call
            return
        # The warn is deliberately OUTSIDE that guard. Under ``-W error`` it
        # raises into the caller -- which is the shape this project spent
        # 8b47cc4 removing from the backend, so it is worth being explicit
        # that it is different here: there, cash raised on its own initiative
        # over a failure the user had not asked to hear about. Here the user
        # configured warnings-as-errors and is entitled to have that honoured.
        # Swallowing it would silently override their filter, which is worse.
        if verdict:
            what, fix = verdict
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "CACHE-NET-LOSS",
                what,
                fix,
            )

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
        body_seconds: float | None = None,
        saves_seconds: float | None = None,
        rng_replay: dict[str, Any] | None = None,
    ) -> None:
        try:
            serializer = get_serializer(result)

            # What a later hit saves is the BODY's time. The wall-clock cost
            # also holds cash's own work -- the first call's analysis, the key
            # -- which the next process pays again whether this entry exists
            # or not. Judged on wall-clock, a function that returns at once was
            # persisted whenever a busy machine made that first-call work cross
            # the 0.1s floor (Windows CI, round 18).
            if body_seconds is not None:
                execution_time = body_seconds
            # The ttl the entry is WRITTEN with, a tier's default included: a
            # backend drops an expired entry on read, so the next miss can only
            # say "expired" -- rather than "evicted or cleared" -- if this
            # process and the stored-key record know it (round 19).
            ttl_declared = ttl is not None or None
            if ttl is None:
                ttl = self._tier_default_ttl()
            meta = CacheMetadata(
                key=cache_key,
                func_name=func_name,
                timestamp=time.time(),
                # The body's measured cost. ``TieredBackend`` reads this to
                # decide whether the value is expensive enough to promote past
                # RAM (otherwise the smart-persistence policy gates everything
                # at the 0.1s floor, and script runs that recompute the same
                # cheap value forever).
                execution_time=execution_time,
                # The body's own cost, so a later HIT can tell what it
                # actually saved. execution_time cannot answer that: it is
                # measured from the top of the wrapper and includes the
                # key hashing whose worth is the question.
                body_seconds=body_seconds,
                saves_seconds=saves_seconds,
                serializer_cls=type(serializer),
                ttl=ttl,
                ttl_declared=ttl_declared,
                args_hash=args_hash,
                state_hash=state_hash,
                # Each entry: path -> {'mtime': float, 'size': int}.
                # Validated on subsequent get() via _auto_file_deps_fresh.
                auto_file_deps=auto_file_deps or None,
                rng_replay=rng_replay or None,
                # Decorating a function IS the decision to cache it, however
                # quick it is. The compute floor belongs to the notebook, where
                # cash caches every statement by itself; here it meant a script
                # run twice recomputed everything, which reads as "cash does
                # not cache" (found attacking the decorator before round 26).
                # Size caps and the tiers' own refusals still apply.
                decorator_entry=True,
                # A SEPARATE promise: the stored value is what the next call
                # hands back, so the RAM tier refuses one it cannot copy rather
                # than sharing it. Not for a `frozen=True` function: declaring
                # a result frozen says it is not modified, and handing the same
                # object back is what that promises for a result no pickle can
                # copy at all. This used to ride on `decorator_entry`, which
                # made `frozen=True` look undecorated to the rate ceiling and
                # cost it disk persistence entirely.
                copy_required=func_name not in self._frozen_funcs,
            )

            # Kept, not a temporary: TieredBackend writes back where the value
            # landed, and "RAM only" is the answer to the next process's miss.
            meta_dict = meta.to_dict()
            self.backend.set(cache_key, result, meta_dict, serializer=serializer)
            # A tiered backend catches each tier's failure so one bad tier
            # cannot break a call; it reports them here instead, and a result
            # nothing could store is a STORE-FAILED like any other.
            store_errors = meta_dict.get("store_errors")
            if store_errors and not [t for t in (meta_dict.get("storage") or []) if t != "RAM"]:
                raise CacheBackendError("; ".join(str(e) for e in store_errors))
            not_persisted = self._not_persisted_reason(meta_dict, execution_time)
            self._remember_outcome(
                cache_key,
                {
                    "stored_at": time.time(),
                    "ttl": ttl,
                    "not_persisted": not_persisted,
                },
            )
            if not_persisted is None:
                self._record_stored_key(func_name, cache_key, ttl)
            else:
                self._remember_ram_only(func_name, cache_key, not_persisted)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError, CacheBackendError) as e:
            self._note_not_stored(cache_key, "the backend refused the write")
            backend_name = type(self.backend).__name__
            self._warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed to store "
                f"result ({type(e).__name__}: {e}). Compute succeeded, nothing was "
                f"stored, and the next call recomputes.",
                code="STORE-FAILED",
                fix=STORE_FAILED_FIX,
            )

    def _warn_cache_if_bypassed(
        self, func_name: str, chunk_max_items: int, chunk_max_bytes: int, stacklevel: int | None = None
    ) -> None:
        """One-shot: the result outgrew a single chunk, so cache_if cannot run.

        Applying it would mean materializing every chunk back into memory,
        undoing the bound that chunking exists to provide.

        The fix line says **raise** the thresholds, and that direction is
        load-bearing. ``cache_if`` is consulted only in the ``chunk_index == 0``
        branch of ``_stream_and_store`` -- the whole result fit one chunk -- and
        this fires at ``chunk_index == 1``, once it did not. The message used to
        advise *lowering* the thresholds, which produces more chunks and so
        guarantees the very bypass it is warning about.
        """
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: the result exceeded a single chunk "
            f"(chunk_max_items={chunk_max_items}, "
            f"chunk_max_bytes={chunk_max_bytes}), so it was cached without "
            f"cache_if ever being consulted.",
            code="CACHE-IF-BYPASSED",
            fix="raise chunk_max_items / chunk_max_bytes above the size this "
            "result reaches, or return a list instead of an iterator, so "
            "the whole result arrives in one piece for the predicate to "
            "see.",
            stacklevel=stacklevel,
        )

    def _stream_and_store(
        self,
        source,
        *,
        cache_key,
        func_name,
        tracker,
        observer,
        rng_new,
        args,
        kwargs,
        args_hash,
        current_state_hash,
        ttl,
        cache_if,
        chunk_max_items,
        chunk_max_bytes,
        code_module=None,
    ):
        """Yield the producer's items as they come, and cache once it ends.

        Three things have to hold at once, and they are why this is not simply
        a `yield` added to a drain-then-store loop:

        **The tracker watches production, never consumption.** A generator's
        file reads happen while it is consumed, so the tracker has to be live
        across `next()`; leaving it live across the caller's `yield` would
        attribute the CALLER's reads to this function. It is entered once and
        suspended around each yield, which records the producer's lazy reads
        and nothing else.

        **Nothing is stored unless the producer ran to exhaustion.** A caller
        that breaks out, or a producer that raises, leaves no entry -- a
        truncated result under the full result's key is a wrong answer, not a
        slow one. Chunks already flushed are removed on the way out. The cost
        is real and accepted: draining eagerly used to leave a complete entry
        behind even when the caller took two items, and it no longer does.

        **Time is the producer's, not the wall clock.** Only the spans inside
        `next()` are summed, so a slow consumer cannot inflate the number the
        persistence decision reads.
        """

        buffer: list[Any] = []
        buffer_bytes = 0
        chunk_index = 0
        total_items = 0
        produced_seconds = 0.0
        committed = False

        try:
            # Entered ONCE. Per item we only suspend around the `yield`, which
            # is a ContextVar swap; `__enter__` reinstalls patches and rechecks
            # the import hook, and paying that per item cost 5.1us each --
            # measured 294ms -> 1521ms on a 200k-item iterator before this.
            with tracker, observer:
                while True:
                    started = _perf_counter()
                    try:
                        item = next(source)
                    except StopIteration:
                        produced_seconds += _perf_counter() - started
                        break
                    produced_seconds += _perf_counter() - started

                    buffer.append(item)
                    buffer_bytes += estimate_object_size(item)
                    total_items += 1
                    if len(buffer) >= chunk_max_items or buffer_bytes >= chunk_max_bytes:
                        if chunk_index == 1 and cache_if is not None:
                            self._warn_cache_if_bypassed(func_name, chunk_max_items, chunk_max_bytes)
                        self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl, execution_time=produced_seconds)
                        buffer = []
                        buffer_bytes = 0
                        chunk_index += 1

                    # The caller's own reads and effects are its own.
                    tracker_token = tracker.suspend()
                    observer_token = observer.suspend()
                    try:
                        yield item
                    finally:
                        observer.resume(observer_token)
                        tracker.resume(tracker_token)

            self._check_argument_mutation(func_name, args, kwargs, args_hash, observer)
            self._report_observed_effects(func_name, observer)
            self._credit_remembered_reads(func_name, tracker, args, kwargs)
            auto_file_deps = self._snapshot_tracked_deps(tracker, code_module)

            if chunk_index == 0:
                # Everything fit in one chunk, so cache_if can still see the
                # whole result -- it gates STORAGE, never what the caller
                # already received.
                refusal = self._store_refusal(None, func_name, buffer, rng_new, cache_if, tracker, observer=observer)
                if refusal is not None:
                    self._note_not_stored(cache_key, refusal)
                else:
                    if buffer:
                        self._write_one_chunk(cache_key, 0, buffer, ttl=ttl, execution_time=produced_seconds)
                    # An empty iterator still gets a zero-chunk manifest, so a
                    # hit returns empty instead of recomputing.
                    self._store_chunked_manifest(
                        cache_key,
                        func_name,
                        {"n_chunks": 1 if buffer else 0, "total_items": total_items},
                        ttl,
                        current_state_hash,
                        args_hash,
                        produced_seconds,
                        auto_file_deps,
                    )
            else:
                if buffer:
                    if chunk_index == 1 and cache_if is not None:
                        self._warn_cache_if_bypassed(func_name, chunk_max_items, chunk_max_bytes)
                    self._write_one_chunk(cache_key, chunk_index, buffer, ttl=ttl, execution_time=produced_seconds)
                    chunk_index += 1
                self._store_chunked_manifest(
                    cache_key,
                    func_name,
                    {"n_chunks": chunk_index, "total_items": total_items},
                    ttl,
                    current_state_hash,
                    args_hash,
                    produced_seconds,
                    auto_file_deps,
                )

            committed = True
        finally:
            if not committed:
                # Abandoned or failed: the chunks written so far are
                # unreferenced (no manifest names them). Best effort -- a
                # killed process can still leave some behind.
                for index in range(chunk_index):
                    try:
                        self.backend.delete(f"{cache_key}:chunk_{index}")
                    except Exception:  # noqa: BLE001 - cleanup must not raise
                        logger.debug("[CORE] could not drop orphan chunk %d", index)

    def _write_one_chunk(
        self,
        cache_key: str,
        chunk_index: int,
        chunk_buffer: list[Any],
        ttl: int | None = None,
        execution_time: float = 0.0,
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
            # The PRODUCER's time, not 0. A chunk is not an independent result
            # whose worth gets decided on its own -- it is the payload of an
            # entry whose durability was already decided. Writing 0 here sent
            # every chunk under the smart-persistence compute floor, so chunks
            # stayed RAM-only while the manifest (carrying the real time) went
            # to disk. A fresh process then found a manifest with no chunks
            # behind it and, because a missing chunk terminates iteration,
            # returned an EMPTY iterator. Silent data loss, and the cross-
            # process hit is the entire point of caching a generator.
            execution_time=execution_time,
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
                # NOT "you will get a truncated iterator". ``_chunks_are_intact``
                # probes every chunk and turns a manifest with a hole into a
                # MISS, on both read paths, so the cost is a permanent recompute
                # rather than a short answer. The locked re-read skipped that
                # guard until 2026-09-06; do not re-introduce the truncation
                # language here without first checking it is true again.
                f"@cash.cache: backend {backend_name} failed to store "
                f"chunk {chunk_index} of {cache_key} ({type(e).__name__}: {e}), "
                f"so the entry can never be read back and every later call "
                f"recomputes it.",
                code="STORE-CHUNK-FAILED",
                fix="clear the entry with f.cache_clear() before anything reads "
                "it, then fix the write the exception names.",
            )

    def _store_chunked_manifest(
        self,
        cache_key: str,
        func_name: str,
        manifest_data: dict[str, Any],
        ttl: int | None,
        state_hash: str,
        args_hash: str,
        execution_time: float,
        auto_file_deps: dict[str, dict[str, float]] | None,
    ) -> None:
        """Write the manifest entry for a chunked iterator at *cache_key*.

        The value stored at the key is the manifest dict (``n_chunks``,
        ``total_items``). The metadata flags this entry as chunked so
        the hit path knows to use ``ChunkedCachedIterator``.
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
                f"Compute succeeded, nothing was stored, and the next call "
                f"recomputes.",
                code="STORE-FAILED",
                fix=STORE_FAILED_FIX,
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
        rows = [(name, s) for name, s in self._function_stats.items() if s["hits"] or s["misses"]]
        bypassed = sum(s.get("bypassed", 0) for s in self._function_stats.values())
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
        # a function that never hit as costing nothing (round 20).
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
            # meant. A tester spent a round on a cache directory that was not
            # the one they thought they had set.
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
            out.append(f"      {MISS_CODE} ({n}x): {what}")
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
                # JSON response -- and a summary landing in it broke all three
                # for round-17 testers. ONE write: pool workers exiting together
                # interleaved print()'s separate writes mid-line (round 18).
                # Into the application's log when it will print it: a service
                # whose output goes through dictConfig never saw the summary.
                # Otherwise to stderr -- once: both, with cash's own handler
                # passing it on as well, printed it three times (round 19).
                # Through the application's handlers whenever it has one that
                # takes INFO -- past the level filters, as CASH_DEBUG's lines
                # are: CASH_SUMMARY asked for it. Gated on the levels, the
                # block came through the app's formatter at INFO and raw at
                # WARNING, two shapes for a log shipper to parse (round 20).
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
        func_name = self.get_func_key(func)
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

        Under ``self._analysis_lock`` because the CACHE KEY is built from what
        this populates. Concurrent first calls otherwise resolved two different
        keys for one call -- the threads that arrived mid-population, and the
        one doing it -- which is what made ``use_locking=True`` look like it
        admitted exactly two threads at every thread count. Both paths into
        this must hold the lock: the wrapper's one-time analysis AND
        ``explain()``, which populates the closure directly and would otherwise
        race it back apart.
        """
        with self._analysis_lock:
            self._ensure_closure_analyzed_locked(func)

    def _ensure_closure_analyzed_locked(self, func: Callable[..., Any]) -> None:
        """The body of ``_ensure_closure_analyzed``, with the lock already held."""
        stack = [func]
        seen: set[str] = set()
        while stack:
            f = stack.pop()
            fname = self.get_func_key(f)
            if fname in seen:
                continue
            seen.add(fname)
            if fname not in self._populated:
                self._populate_analysis(f, fname)
            for dep in self.graph.get_dependencies(fname):
                dep_func = self.functions.get(dep)
                if dep_func is not None:
                    stack.append(dep_func)

    def _refresh_helper_bindings(self, func: Callable[..., Any], func_name: str) -> str | None:
        """Re-analyse any report whose call-site bindings moved; name a mock.

        Walks *func* and its cached-dependency closure. A report is built once
        per function, from whatever its call sites' names held at that moment,
        so a helper patched before the first call stayed "the helper" after
        the real one came back, and the helpers below a patched one were never
        walked. When a binding no longer holds the object analysed, that
        function is analysed again from the current bindings (the analyzer's
        own cache applies the same check).

        Returns a description when the tree reaches a mock -- there is no code
        to key and its answer is whatever the test configured, so the caller
        runs this call uncached -- and None otherwise. Never raises: a failure
        here leaves the reports as they were.

        Per call: one ``sys.modules`` lookup, an attribute chain and an
        identity test per binding, no hashing.
        """
        try:
            reason: str | None = None
            stack: list[tuple[Callable[..., Any], str]] = [(func, func_name)]
            seen: set[str] = set()
            while stack:
                f, name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                report = self._purity_reports.get(name)
                if report is not None and report.helper_bindings and bindings_changed(report):
                    with self._analysis_lock:
                        self._populate_analysis(f, name)
                    report = self._purity_reports.get(name)
                if report is not None and report.unkeyable and reason is None:
                    reason = report.unkeyable[0]
                for dep in self.graph.get_dependencies(name):
                    dep_func = self.functions.get(dep)
                    if dep_func is not None:
                        stack.append((dep_func, dep))
            return reason
        except Exception:  # noqa: BLE001 - never break a call over this
            logger.debug("[CORE] binding refresh failed for %s", func_name, exc_info=True)
            return None

    def _populate_analysis(self, func: Callable[..., Any], func_name: str) -> None:
        """Record *func*'s cached-call graph edges and purity report (no
        surfacing). The analyzer caches by source hash globally, so this is
        cheap on repeated registrations.
        """
        self._populated.add(func_name)
        called_names = CodeAnalyzer.find_called_functions(func, self.functions, include_references=True)
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

    #: A re-hash costing more than this marks the function as not worth
    #: verifying again. Measured: ~7.4 ms for a 16 MB ndarray, so this is
    #: roughly a 100 MB argument. The first miss still gets checked -- the
    #: budget only stops a large argument from being re-hashed on every
    #: subsequent miss.
    _MUTATION_CHECK_BUDGET_S = 0.05

    def _argument_snapshot(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, str] | None:
        """``{parameter: hash}`` of the arguments that CAN change, before the body.

        An int, a str, a tuple of them: rebinding one inside the body (``n -=
        1``) is invisible to the caller, so they are left out, and most calls
        snapshot nothing. What remains lets `_check_argument_mutation` name the
        argument that moved. None when the check has been retired as too
        costly for this function, or nothing could be hashed.
        """
        if func_name in self._mutation_check_too_costly:
            return None
        # The key was hashed a moment ago, on this thread: if that already cost
        # more than the check may, the check is retired before it pays -- a
        # miss on two million rows hashed them three times, once for the key,
        # once here and once after the body (round 19). Read from what
        # `_note_arg_cost` kept: it has already taken `ARG_COST.last`.
        cost = self._arg_costs.get(func_name)
        if cost is not None and cost[2] > self._MUTATION_CHECK_BUDGET_S:
            self._mutation_check_too_costly.add(func_name)
            return None
        started = _perf_counter()
        try:
            canon_args, canon_kwargs = self._normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            return None
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        snapshot: dict[str, str] = {}
        immutable = IMMUTABLE_VALUE_TYPES
        candidates = [
            (name, value)
            for name, value in named
            if not (self._is_immutable_capture(value) or isinstance(value, immutable))
        ]
        if len(candidates) == 1:
            # The only argument that can change is the one that did: named by
            # elimination, with no hash of its own.
            return {candidates[0][0]: ""}
        for name, value in candidates:
            try:
                snapshot[name] = self._hash_arg_payload((value,), {})
            except Exception:  # noqa: BLE001 - unhashable: the whole-args check still runs
                continue
        if _perf_counter() - started > self._MUTATION_CHECK_BUDGET_S:
            self._mutation_check_too_costly.add(func_name)
        return snapshot

    def _argument_identities(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, tuple[Any, list]]:
        """``{parameter: (value, identity snapshot)}`` for the plain lists and
        tuples a call receives, before the body runs.

        The hash snapshot below is retired for a big argument and never covers
        a frozen one, which is exactly where ``rows.sort()`` on a million
        parsed rows, or a field rewritten in every row of a frozen result, got
        stored (round 20). Identities cost a fraction of a hash, so they are
        taken whatever the size (`_plain_data.identity_snapshot`).
        """
        try:
            canon_args, canon_kwargs = self._normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            return {}
        found: dict[str, tuple[Any, list]] = {}
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        for name, value in named:
            if type(value) is list or type(value) is tuple:
                snapshot = _plain_data.identity_snapshot(value)
                if snapshot is not None and any(level is not None for level in snapshot):
                    found[name] = (value, snapshot)
        return found

    def _forget_frozen_container(self, obj: Any) -> None:
        """Stop trusting a frozen result a call was just seen to change."""
        entry = self._frozen_containers.get(id(obj))
        if entry is None or entry[0] is not obj:
            return
        self._frozen_containers.pop(id(obj), None)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {entry[1]}, which is declared "
            f"@cash.cache(frozen=True), was modified in place by a cached call. "
            f"From now on it is keyed by its contents.",
            f"take frozen=True off {entry[1]} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )

    def _check_argument_mutation(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        args_hash: str | None,
        observer: Any,
    ) -> None:
        """Did the body change the arguments it was handed?

        The static analyzer sees `rows.append(x)` written in the function, but
        not `vendor.normalize(rows)` sorting the list inside a library it does
        not walk into. Measured: of four state mutations that reached past the
        analyzer, three left an observable change in the arguments -- so the
        cheapest way to find them is to look.

        The argument hash is already computed to build the cache key, so this
        re-runs exactly that and compares. `_serialize_args` canonicalises
        (kwargs order included) and is deterministic on unchanged input, which
        is what makes a difference mean *mutation* rather than noise.

        Runs only on a miss, and only while it stays cheap: a re-hash over the
        budget above retires the check for that function rather than taxing
        every later miss.
        """
        if args_hash is None or observer is None:
            return
        identities = getattr(observer, "arg_identities", None)
        if identities:
            moved = [
                name for name, (value, snapshot) in identities.items() if _plain_data.identity_changed(value, snapshot)
            ]
            if moved:
                for name in moved:
                    self._forget_frozen_container(identities[name][0])
                observer.mutated_args = moved
                observer.record(
                    "argument mutation",
                    f"the call changed {', '.join(repr(n) for n in moved)} in place -- "
                    f"the result was not stored, so this call runs every time",
                )
                return
        if func_name in self._mutation_check_too_costly:
            return
        started = _perf_counter()
        try:
            after = self._serialize_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001
            # Hashing is best-effort here. An argument that hashed once and
            # not twice (a generator drained by the body, say) is not evidence
            # of mutation, and must not be reported as such.
            return
        if _perf_counter() - started > self._MUTATION_CHECK_BUDGET_S:
            self._mutation_check_too_costly.add(func_name)
        if after is None or after == args_hash:
            return
        names = self._mutated_argument_names(func_name, args, kwargs, observer)
        observer.mutated_args = names or ["an argument"]
        shown = ", ".join(repr(n) for n in names) if names else "an argument"
        observer.record(
            "argument mutation",
            f"the call changed {shown} in place -- the result was not stored, so this call runs every time",
        )

    def _mutated_argument_names(self, func_name: str, args: tuple, kwargs: dict, observer: Any) -> list[str]:
        """The parameters whose value moved across the call, by name."""
        before = getattr(observer, "arg_snapshot", None)
        if not before:
            return []
        if len(before) == 1:
            return list(before)
        now = self._argument_snapshot(func_name, args, kwargs) or {}
        return [name for name, digest in before.items() if now.get(name) != digest]

    def _make_effect_observer(self) -> Any:
        """An :class:`EffectObserver` scoped to this instance's cache dir.

        Excluding the cache directory is load-bearing: cash writes the entry
        it is computing, and without the exclusion every cached function would
        be observed writing a file and every one of them would warn.
        """

        cache_dir = getattr(self.config, "cache_dir", None)
        return EffectObserver(exclude_under=cache_dir)

    def _report_observed_effects(self, func_name: str, observer: Any) -> None:
        """Warn once when the first call did something a hit will not do.

        Silent when:

        * ``assume_safe=True`` -- the user audited this function and said so;
          ``# @cash:assume-safe`` on a line that led to an effect waives that
          effect alone (see ``EffectObserver.record_effect``).
        * the static findings already name that KIND of effect -- a write the
          analyzer listed is not news when the observer sees it too. Only the
          kinds they cover are dropped. The whole warning used to be, so a
          static finding about a log line hid a network read in the same
          function: never reported in 30 starts (round 18).
        * nothing was observed -- which is *not* proof of purity. Only the
          path this call took was watched, so an effect behind a branch that
          did not run is unobserved. That is why this supplements the static
          pass rather than replacing it.
        """
        if observer is None or not observer.effects:
            return
        if self._purity_modes.get(func_name, "warn") == "silent":
            return
        covered: set[str] = set()
        if func_name in self._purity_static_flagged:
            covered = static_effect_kinds(self._purity_reports.get(func_name))
        effects = [(kind, detail) for kind, detail in observer.effects if kind not in covered]
        if not effects:
            return
        summary = "\n".join(dict.fromkeys(f"  {kind}: {detail}" for kind, detail in effects))
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "observed_effect",
            f"@cash.cache on {func_name}: the first call had effects that "
            f"static analysis did not see -- they happen inside library code "
            f"cash does not walk into. Every cache HIT from here on returns "
            f"the stored value WITHOUT repeating them.\n{summary}",
            code="IMPURE-OBSERVED-EFFECTS",
            fix="split the function if an effect is part of the result -- an "
            "'argument mutation' line means an object the CALLER still "
            "holds stops being changed. If it is incidental, put "
            "`# @cash:assume-safe` on the line named (any line of yours on "
            "the way to it counts); @cash.cache(assume_safe=True) waives "
            "the whole function instead, including effects added later.",
        )

    def _mutable_global_is_keyed(self, func_name: str, report: PurityReport, issue: Any) -> bool:
        """Is this ``mutable_global`` finding about a global the key already
        folds by value on every call?

        Then "cached results won't reflect changes to it" is false: a setter
        rebinding it, or a test patching it, makes the next call a new entry.
        Measured in round 18 -- a `configure()`-set module flag re-ran the
        function each time it changed, 0 diffs against a no-cache oracle --
        while the warning, of the kind the docs say never to ignore, said
        otherwise. Kept for what the fold leaves out: callables, modules,
        classes (tracked their own way, or not at all), and a global the
        function itself writes.
        """

        name = getattr(issue, "subject", "")
        if getattr(issue, "kind", None) != ISSUE_MUTABLE_GLOBAL or not name:
            return False
        func = self.functions.get(func_name)
        reader: Any = func
        where = getattr(issue, "where", "")
        if where in report.helper_resolution_paths:
            reader = resolve_binding(*report.helper_resolution_paths[where])
        elif where in report.helper_objects:
            reader = report.helper_objects[where]()
        module_ns = getattr(reader, "__globals__", None)
        if not isinstance(module_ns, dict) or name not in module_ns:
            return False
        if reader is not func and self._helper_mutates_global(reader, name):
            # The loader of a lazily filled settings dict (`_CFG.clear();
            # _CFG.update(...)`) "reads" it only to fill it: its writes are
            # findings of their own, and the dict is not its input. Reported
            # as a stale-result risk, it was one more false alarm on the most
            # common settings pattern there is (round 19).
            return True
        try:
            if name not in self._read_global_data_names(reader):
                return False
        except Exception:  # noqa: BLE001
            return False
        value = module_ns[name]
        if isinstance(value, types.ModuleType):
            # `conf.RATE` reads of a module of the user's are folded by value
            # (`_module_attr_parts`); "mutated elsewhere" is `conf.RATE = ...`.
            return self._is_user_module(value, self._own_package(reader))
        if isinstance(value, type):
            return False
        return not (callable(value) and not isinstance(value, (dict, list, tuple, set)))

    @staticmethod
    def _helper_mutates_global(fn: Any, name: str) -> bool:
        """Does *fn*'s own body rebind *name* or change it in place?"""
        code = getattr(fn, "__code__", None)
        if code is None:
            return False

        for scope in Cash._iter_code_scopes(code):
            for instr in dis.get_instructions(scope):
                if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL") and instr.argval == name:
                    return True
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
            return False
        return name in Cash._unsafe_uses_of(tree, frozenset({name}), bare_args=False, mutating_methods_only=True)

    def _surface_purity(
        self,
        func_name: str,
        report: PurityReport,
        mode: str,
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
        issues = [i for i in report.issues if not self._mutable_global_is_keyed(func_name, report, i)]
        if any(getattr(i, "kind", None) == ISSUE_NETWORK_READ for i in issues):
            # Named statically, so the observer does not report the same read
            # as a connection -- whether or not the advisory below is shown.
            self._purity_static_flagged.add(func_name)
            if self._effective_ttl(func_name, self._func_ttls.get(func_name)) is not None:
                # `ttl=` is the answer to "how old may a fetched answer be":
                # once one is set, the question has been answered.
                issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
        if mode == "strict" and report.opaque_callees:
            opaque_list = ", ".join(report.opaque_callees[:5])
            if len(report.opaque_callees) > 5:
                opaque_list += f", ... +{len(report.opaque_callees) - 5} more"
            issues.append(make_opaque_issue(func_name, opaque_list))

        if not issues:
            return
        if mode == "silent":
            return

        summary = format_issues_summary(func_name, issues)

        # Untrackable-dependency patterns (eval/exec/compile, getattr(obj,name)()
        # dynamic dispatch, importlib.import_module) RAISE by default, even in
        # the ordinary "warn" mode: cash cannot see an edit to a dependency it
        # resolves from a runtime value, so a cached result can go silently
        # stale, and caching correctness can no longer be guaranteed. The user
        # must acknowledge the risk with assume_safe=True (the ``silent`` mode
        # handled above) to cache anyway.
        untrackable = [i for i in issues if getattr(i, "kind", None) == ISSUE_UNTRACKABLE_DEP]
        if untrackable and mode != "strict":
            untrackable_summary = format_issues_summary(func_name, untrackable)
            raise CashImpureFunctionError(
                f"@cash.cache on {func_name}: a dependency is resolved from a "
                f"runtime value, so cash cannot tell when it changes and a cached "
                f"result could be silently stale. Caching correctness cannot be "
                f"guaranteed for this function.\nPut `# @cash:assume-safe` on "
                f"the line named below to accept the risk for that statement "
                f"alone, pass @cash.cache(assume_safe=True) to waive the whole "
                f"function, or refactor to a statically-named "
                f"call.\n{untrackable_summary}"
            )

        # Ambient reads get their own warning, not the side-effects one. The
        # hazard is the opposite shape -- nothing is skipped, a hidden INPUT is
        # frozen -- and so is the fix: pass the value in as an argument, where
        # it reaches the key. Filing them under "likely side effects" told the
        # user to audit for writes that are not there, and left the actual
        # failure (a nightly job whose `date.today()` is the night it first
        # ran) unnamed.
        # strict=True keeps them in the one exception it raises: there, every
        # issue is a hard stop and splitting the report would hide half of it.
        ambient = [i for i in issues if getattr(i, "kind", None) == ISSUE_AMBIENT_READ]
        if ambient and mode != "strict":
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_AMBIENT_READ]
            self._purity_static_flagged.add(func_name)
            self._warn_once(
                CashImpurityWarning,
                func_name,
                "ambient",
                f"@cash.cache on {func_name}: the body reads ambient state "
                f"(the clock, the environment, the working directory, a fresh "
                f"UUID). That value is not part of the cache key, so the first "
                f"call's answer is what every later call gets back -- in this "
                f"process and in every process "
                f"after it.\n{format_issues_summary(func_name, ambient)}",
                code="KEY-AMBIENT-READ",
                fix="pass the value in as an argument -- `f(now=datetime.now())` "
                "-- so it reaches the cache key and a new value means a new "
                "entry. If freezing it is what you want, say so with "
                "`# @cash:assume-safe` on that line.",
                once_per_version=True,
            )
        # A network or database read gets its own advisory too, for the same
        # reason as an ambient read: nothing is skipped, an input the key
        # cannot see is frozen. Unlike the clock it has a knob made for it,
        # `ttl=`, which silences it (above). Under strict=True it raises with
        # the other issues unless a ttl= is set.
        remote = [i for i in issues if getattr(i, "kind", None) == ISSUE_NETWORK_READ]
        if remote and mode != "strict":
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
            self._warn_once(
                CashImpurityWarning,
                func_name,
                "network_read",
                f"@cash.cache on {func_name}: the result depends on what a "
                f"server or a database returned, and that answer is not part "
                f"of the cache key. The first call's answer is what every later call gets "
                f"back -- in this process and in every process after it -- "
                f"until something changes the key.\n{format_issues_summary(func_name, remote)}",
                code="KEY-NETWORK-READ",
                fix="bound how old a served answer may be with ttl= -- "
                "`@cash.cache(ttl=3600)` -- or pass what makes the answer new "
                "(a date, a version) as an argument, so it reaches the key. If "
                "the answer never changes, say so with `# @cash:assume-safe` "
                "on that line.",
                once_per_version=True,
            )
        if not issues:
            return
        summary = format_issues_summary(func_name, issues)

        self._purity_static_flagged.add(func_name)

        if mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: purity issues "
                f"detected. Fix the function, mark callees with "
                f"@pure / @stateful, put `# @cash:assume-safe` on the lines you "
                f"have audited, or relax to assume_safe=True.\n{summary}"
            )
        # mode == "warn"
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "purity",
            f"@cash.cache on {func_name}: reading the source found likely "
            f"side effects or scope mutations, so cached results may not "
            f"reflect what the body does.\n{summary}",
            code="IMPURE-SIDE-EFFECTS",
            fix=(
                (
                    "for a line that changes an argument in place, return a "
                    "modified copy instead -- the caller keeps its object "
                    "whether the call hits or misses. "
                    if "changes the argument" in summary or "element of the argument" in summary
                    else ""
                )
                + "go down the list and put `# @cash:assume-safe` on each line "
                "you have audited, or refactor; @cash.cache(assume_safe=True) "
                "waives the whole function instead, including anything added "
                "to it later. The first annotation changes the function's key "
                "once: @cash: directives are part of its source identity."
            ),
            once_per_version=True,
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

        file_registry().register(module_name, func_name, handler_factory)

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
        backend = getattr(self, "_backend", None)
        effectiveness = getattr(self, "_effectiveness", None)
        if effectiveness is not None:
            try:
                verdicts = effectiveness.final_verdicts()
            except Exception:  # noqa: BLE001 - a notice must never block shutdown
                verdicts = []
            for what, fix in verdicts:
                try:
                    warn_diagnostic(CashCacheIneffectiveWarning, "CACHE-NET-LOSS", what, fix)
                except Exception:  # noqa: BLE001 - -W error at exit, or teardown
                    pass
        if backend is not None:
            if getattr(self, "_ram_only_pending", None) or getattr(self, "_warned_pending", None):
                self._flush_ram_only_keys()
            backend.shutdown()
