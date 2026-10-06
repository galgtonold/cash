"""One cached call, from its key to its result: build the key, look it up,
run the body on a miss, and hand back what to return."""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import hashlib
import logging
import os
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

from .._active import EXPLAINING as _EXPLAINING
from .._clock import perf_counter as _perf_counter
from ..analysis.purity_report import PurityReport
from ..backends._base import ttl_expired, written_at
from ..dependency_state import STATE_LEDGER, ledger_note
from ..exceptions import CashCacheIneffectiveWarning
from ..tracking.file_tracker import FileAccessTracker
from ..tracking.randomness import (
    capture_argument_carrier_states,
    capture_reachable_carrier_states,
    carriers_put_back,
    moved_carriers,
    replay_argument_carriers,
    replayable,
)
from .arg_hashing import PLAIN_CENSUS
from .arg_key import keyed_arguments
from .cache_metadata import CacheMetadata
from .cached_function import CachedFunction
from .call_state import (
    CACHE_MISS,
    CAPTURE_WATCH,
    NESTED_CASH_SECONDS,
    THREADS_IN_CALLS,
    BodyRun,
    BuiltKey,
    Call,
    KeyBuildFailed,
    UnhashableArgs,
    UnhashableDefault,
    run_to_completion,
)
from .class_data import CLASSES_FOLDED
from .closure_fold import defaults_of
from .explain import MissKind, MissReason, describe_stale_files
from .file_deps import note_unentered_body, propagate_file_deps_to_active_tracker, snapshot_tracked_deps
from .function_identity import func_key, hash_callable_source
from .globals_fold import READS_FOLDED
from .iterators import ChunkedCachedIterator, StreamingCachedIterator, chunk_prefix, is_one_shot_iterator
from .registry import resolve_dynamic_dependencies
from .rng import capture_rng_pre_state, replay_rng_state
from .store import StoreRequest

if TYPE_CHECKING:
    from ..dependency_state import DependencyStateHasher
    from .arg_hashing import ArgHasher
    from .backend_slot import BackendSlot
    from .closure_fold import ClosureFold
    from .code_args import CodeArgs
    from .environment_fold import EnvironmentFold
    from .explain import MissHistory
    from .file_deps import FileDeps
    from .function_identity import OwnSourcePins
    from .globals_fold import GlobalsFold
    from .method_deps import MethodClassDeps
    from .purity_checks import PurityChecks
    from .registry import FunctionRegistry
    from .reporting import CallLog, Notices
    from .rng import RngWatch
    from .store import ResultStore

logger = logging.getLogger(__name__)


#: Cached functions whose `KeyBuilder.callee_state` is being built on this
#: thread, so one that reaches itself folds its name instead of recursing.
_CALLEE_STATES: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "_cash_callee_states", default=frozenset()
)


class Unkeyable(NamedTuple):
    """A call that gets no key, and the miss it is logged as."""

    reason: MissReason


_UNHASHABLE = MissReason(MissKind.UNHASHABLE, "an argument could not be hashed, so there is no key to look up")
_KEY_FAILED = MissReason(MissKind.KEY_FAILED, "building the key raised")


def _unhashable_callee_default(func_name: str) -> KeyBuildFailed:
    """The failure of a caller's key when cached *func_name*'s default cannot be hashed."""
    return KeyBuildFailed(
        "KEY-UNHASHABLE-DEFAULT",
        f"@cash.cache on {func_name}: a parameter default of this cached function, "
        f"which another cached function calls, could not be hashed, so the caller "
        f"cannot tell whether it changed and ran uncached.",
        "give the default a hashable value, or register a hasher for its type with cash.register_hasher.",
    )


def decorator_key(func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
    return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"


class KeyBuilder:
    """The cache key of a call: the state segment folded from the function's
    code, closure, defaults, globals, files, RNG epoch and environment, then
    the dynamic dependencies and the arguments."""

    def __init__(
        self,
        registry: FunctionRegistry,
        args: ArgHasher,
        method_deps: MethodClassDeps,
        pins: OwnSourcePins,
        files: FileDeps,
        closures: ClosureFold,
        globals_fold: GlobalsFold,
        environment: EnvironmentFold,
        rng: RngWatch,
        code_args: CodeArgs,
        state_hasher: DependencyStateHasher,
        misses: MissHistory,
        notices: Notices,
    ) -> None:
        self._registry = registry
        self._args = args
        self._method_deps = method_deps
        self._pins = pins
        self._files = files
        self._closures = closures
        self._globals = globals_fold
        self._environment = environment
        self._rng = rng
        self._code_args = code_args
        self._state_hasher = state_hasher
        self._misses = misses
        self._notices = notices

    def resolve(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> tuple[BuiltKey | Unkeyable, dict]:
        """The key for a real call, or why it has none; and its `CAPTURE_WATCH`.

        `KeyBuilder.build`, with a ledger of what the state segment is made of
        (`STATE_LEDGER`). A call with no key -- a mocked helper, an argument
        or default that cannot be hashed, a key build that raised -- has
        been warned about once; the caller runs it uncached.
        """
        unkeyable = self._registry.refresh_helper_bindings(func, func_name)
        if unkeyable is not None:
            return Unkeyable(unkeyable), {}
        ledger: dict = {}
        ledger_token = STATE_LEDGER.set(ledger)
        watch: dict = {}
        watch_token = CAPTURE_WATCH.set(watch)
        try:
            built = self.build(func, func_name, dynamic_depends_on, args, kwargs)
        except UnhashableDefault:
            # `ClosureFold.fold_defaults` has warned: an unhashable default means cash
            # cannot tell whether it changed, so caching at all risks a stale
            # result.
            return Unkeyable(_UNHASHABLE), watch
        except UnhashableArgs as e:
            hashed_args, hashed_kwargs = e.keyed if e.keyed is not None else (args, kwargs)
            self._args.warn_unhashable_args(func_name, hashed_args, hashed_kwargs, e.args[0] if e.args else None)
            return Unkeyable(_UNHASHABLE), watch
        except KeyBuildFailed as e:
            self._notices.warn_once(CashCacheIneffectiveWarning, func_name, e.code, e.message, code=e.code, fix=e.fix)
            return Unkeyable(_KEY_FAILED), watch
        except Exception as e:  # noqa: BLE001 - any failure building the key means no key
            self._args.warn_key_build_failed(func_name, args, kwargs, e)
            return Unkeyable(_KEY_FAILED), watch
        finally:
            CAPTURE_WATCH.reset(watch_token)
            STATE_LEDGER.reset(ledger_token)
        if ledger:
            slot = (func_name, built.state_hash)
            if not self._misses.has_ledger(slot):
                self._misses.keep_state_ledger(slot, ledger)
        return built, watch

    def _code_state(self, func: Callable, func_name: str, chain: list[str], *, note: bool = False) -> str:
        """The state segment before anything the arguments decide: *func*'s
        code, helpers, cached callees, declared files, closure, defaults,
        globals, RNG epoch and environment. Appends each stage to *chain*.

        Raises `UnhashableDefault` when a default cannot be hashed.
        """
        state_hash = self._state_hasher.compute(
            func_name,
            own_source_override=self._pins.pin_own_source(func),
            own_report=self._registry.report_for(func, func_name),
            note=note,
        )
        state_hash = self._fold_callee_bindings(func_name, state_hash)
        state_hash = self._files.fold_declared_files(func_name, state_hash)
        chain.append(state_hash)
        state_hash = self._closures.fold_closure(func, func_name, state_hash)
        chain.append(state_hash)
        folded_defaults = self._closures.fold_defaults(func, func_name, state_hash)
        if folded_defaults is None:
            raise UnhashableDefault
        state_hash = folded_defaults
        chain.append(state_hash)
        state_hash = self._closures.fold_bound_self(func, func_name, state_hash)
        state_hash = self._closures.fold_bound_partial(func, func_name, state_hash)
        chain.append(state_hash)
        state_hash = self._globals.fold_read_globals(func, func_name, state_hash)
        state_hash = self._globals.fold_helper_read_globals(func, func_name, state_hash)
        state_hash = self._globals.fold_dependency_read_globals(func, func_name, state_hash)
        chain.append(state_hash)
        state_hash = self._rng.fold_rng_epoch(func_name, state_hash)
        chain.append(state_hash)
        state_hash = self._environment.fold_environment(func, func_name, state_hash)
        chain.append(state_hash)
        spec = self._registry.cached.get(func_name)
        if spec is not None and spec.arg_key is not None:
            # What decides the argument part is code too: a caller that
            # reaches this function keys it as well (`callee_state`).
            state_hash = self._fold_key_function(spec, state_hash)
        return state_hash

    def _fold_callee_bindings(self, func_name: str, state_hash: str) -> str:
        """Fold what each cached function *func_name* calls is bound to: its
        captured variables and its parameter defaults, transitively.

        The dependency state keys a cached callee by its code, helpers and
        callees. Its captures and defaults live on the function object, not in
        its code: ``def clip(x, limit=THRESHOLD)`` with ``THRESHOLD`` edited in
        another module, or ``add = make(n)`` built with another ``n``, served
        the caller's old result while the callee called directly recomputed.
        They are folded by the same rules as the caller's own
        (`ClosureFold.fold_closure`, `ClosureFold.fold_defaults`). A callee
        another registry describes (`FunctionRegistry.reached_callee`) is
        keyed whole by its own state already.

        Raises `KeyBuildFailed` when a callee's default cannot be hashed.
        """
        parts: list[str] = []
        graph = self._registry.graph
        visited = {func_name}
        stack = [(func_name, dep) for dep in sorted(graph.get_dependencies(func_name), reverse=True)]
        # A callee's mutable capture is not this call's to watch.
        watch_token = CAPTURE_WATCH.set(None)
        try:
            while stack:
                node, dep = stack.pop()
                if dep in visited:
                    continue
                visited.add(dep)
                dep_func = self._registry.functions.get(dep)
                if dep_func is None or self._registry.reached_callee(node, dep) is not None:
                    continue
                stack.extend((dep, d) for d in sorted(graph.get_dependencies(dep), reverse=True))
                if not getattr(dep_func, "__closure__", None) and defaults_of(dep_func) == ((), {}):
                    continue
                bound = self._closures.fold_closure(dep_func, dep, "")
                bound = self._closures.fold_defaults(dep_func, dep, bound)
                if bound is None:
                    raise _unhashable_callee_default(dep)
                ledger_note(("captures and defaults of cached function", dep), bound)
                parts.append(f"{dep}={bound}")
        finally:
            CAPTURE_WATCH.reset(watch_token)
        if not parts:
            return state_hash
        return hashlib.sha256(f"{state_hash}:callee-bindings:{':'.join(parts)}".encode("utf-8")).hexdigest()

    def callee_state(self, func: Callable, func_name: str) -> str:
        """What a call of cached *func_name* depends on besides its arguments,
        as ONE digest: for a cached function another one reaches without a
        graph edge of its own registry.

        A cached function on another `Cash`, one passed in as an argument or
        held in a dict, a list or a closure, or one a caller still holds after
        ``importlib.reload`` put a new function under its name: its code was
        keyed by cash's own wrapper, or by the new function, and the globals
        and environment it reads not at all, so an edit to any of them served
        the caller's old result. This is the key's own state segment, built by
        the instance that owns *func*, for the function object the caller
        reaches.

        A cycle (a cached function that reaches itself through a table)
        contributes its name. The ledger and the capture watch of the key
        being built are the caller's, so they are set aside meanwhile.

        Raises `KeyBuildFailed` when the state cannot be built.
        """
        active = _CALLEE_STATES.get()
        if func_name in active:
            return f"cycle:{func_name}"
        token = _CALLEE_STATES.set(active | {func_name})
        ledger_token = STATE_LEDGER.set(None)
        watch_token = CAPTURE_WATCH.set(None)
        # The callee's digest folds its own classes, whatever the caller folded.
        classes_token = CLASSES_FOLDED.set(set())
        reads_token = READS_FOLDED.set({})
        try:
            if self._registry.functions.get(func_name) is func:
                self._registry.ensure_closure_analyzed(func)
            return self._code_state(func, func_name, [])
        except UnhashableDefault:
            raise _unhashable_callee_default(func_name) from None
        finally:
            READS_FOLDED.reset(reads_token)
            CLASSES_FOLDED.reset(classes_token)
            CAPTURE_WATCH.reset(watch_token)
            STATE_LEDGER.reset(ledger_token)
            _CALLEE_STATES.reset(token)

    def build(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> BuiltKey:
        """The cache key for calling *func* with these arguments.

        The ONE key build: a real call (`KeyBuilder.resolve`) and ``explain()``
        (`Explainer.explain`) both use it, so the key explain() predicts is the key
        the call looks up.

        Raises when there is no key: `UnhashableDefault`, `UnhashableArgs`,
        `KeyBuildFailed`, or whatever else a step raised. Never keys the call
        without a part that failed. Warnings from the steps are silent while
        `_EXPLAINING` is set.
        """
        # One plain-data census per argument, shared across the key
        # (`plain_census`).
        previous = getattr(PLAIN_CENSUS, "memo", None)
        PLAIN_CENSUS.memo = {}
        # Each class's data is folded once per key (`ClassDataFold.class_parts`).
        classes_token = CLASSES_FOLDED.set(set())
        # Each function's globals are folded once per key (`READS_FOLDED`).
        reads_token = READS_FOLDED.set({})
        try:
            # The state after each fold, in `_STATE_STAGES` order: when no
            # named part moved, the first stage whose output did is the one
            # that changed (`describe_state_change`).
            chain: list[str] = []
            ledger_note("@chain", chain)
            state_hash = self._code_state(func, func_name, chain, note=True)
            state_hash = self._method_deps.fold_method_class_deps(func, args, state_hash)
            chain.append(state_hash)
            # ONE canonicalisation, fed to both the code channel and the value
            # channel. `CodeArgs.fold_code_args` on the RAW arguments saw a class
            # passed explicitly but not the identical class arriving as a
            # parameter DEFAULT, so `build()` and `build(Schema)` -- the same
            # logical call -- produced two cache keys and two executions.
            normalized_args = self._args.normalize_call_args(func_name, args, kwargs)
            spec = self._registry.cached[func_name]
            if spec.seed_params:
                self._rng.warn_if_seed_is_none(func, func_name, args, kwargs)
            # What the argument part of the key is made of: every argument,
            # or what ``key=`` / the ignored parameters leave of them. Both
            # channels read it, the code one and the value one.
            keyed = normalized_args
            if spec.arg_key is not None:
                keyed = self.key_arguments(spec, args, kwargs, normalized_args)
            state_hash = self._code_args.fold_code_args(
                *keyed, state_hash, func_name=func_name, owner_code=getattr(func, "__code__", None)
            )
            chain.append(state_hash)
            dynamic_state_hash = resolve_dynamic_dependencies(func_name, dynamic_depends_on, args, kwargs)
            failure: list = []
            if keyed is normalized_args:
                args_hash = self._args.serialize_args(
                    func_name, args, kwargs, normalized=normalized_args, failure=failure
                )
            else:
                args_hash = self._keyed_args_hash(keyed, failure)
            self._args.note_arg_cost(func_name)
        finally:
            PLAIN_CENSUS.memo = previous
            READS_FOLDED.reset(reads_token)
            CLASSES_FOLDED.reset(classes_token)
        if args_hash is None:
            raise UnhashableArgs(*failure[:1], keyed=None if keyed is normalized_args else keyed)
        cache_key = decorator_key(func_name, state_hash, dynamic_state_hash, args_hash)
        call_args_hash = args_hash if spec.arg_key is None else None
        return BuiltKey(cache_key, state_hash, args_hash, normalized_args, call_args_hash)

    def key_arguments(self, spec: CachedFunction, args: tuple, kwargs: dict, normalized: tuple[tuple, dict]) -> tuple:
        """`keyed_arguments` for *spec*'s ``key=`` or ignored parameters.

        The first time a key function runs, it runs under a file tracker of
        its own: a file it reads is an input the key cannot see, and the
        static check (`key_function_impurities`) only finds the readers it
        can name. Raises `KeyBuildFailed` when the key function raises.
        """
        arg_key = spec.arg_key
        if arg_key.key_fn is None or spec.key_reads_checked or _EXPLAINING.get():
            return keyed_arguments(arg_key, spec.signature, spec.name, args, kwargs, normalized)
        spec.key_reads_checked = True
        tracker = FileAccessTracker()
        with tracker:
            keyed = keyed_arguments(arg_key, spec.signature, spec.name, args, kwargs, normalized)
        read = sorted(tracker.get_accessed_files() | tracker.get_accessed_remote_urls())
        if read and spec.purity != "silent":
            self._notices.key_function_impure(spec.name, [f"reads {os.path.normpath(path)}" for path in read[:3]])
        return keyed

    def _fold_key_function(self, spec: CachedFunction, state_hash: str) -> str:
        """Fold the ``key=`` function's code into the state, as a function
        passed as an argument is folded: its code, the user code it reaches
        and the globals that code reads. An edit to it re-keys every call."""
        key_fn = spec.arg_key.key_fn
        if key_fn is None:
            part = f"ignore:{sorted(spec.arg_key.ignored)}"
            ledger_note(("ignored parameters", ", ".join(sorted(spec.arg_key.ignored))), part)
        else:
            parts = [f"{func_key(key_fn)}:{hash_callable_source(key_fn)}"]
            parts.extend(
                self._code_args.carrier_parts(key_fn, spec.name, "key", owner_code=getattr(spec.func, "__code__", None))
            )
            part = "keyfn:" + hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
            ledger_note(("key= function", getattr(key_fn, "__qualname__", type(key_fn).__name__)), part)
        return hashlib.sha256(f"{state_hash}:{part}".encode("utf-8")).hexdigest()

    def _keyed_args_hash(self, keyed: tuple[tuple, dict], failure: list) -> str | None:
        """`ArgHasher.hash_payload` of what ``key=`` / the ignored parameters
        left, or None (with what raised in *failure*). No retry with the raw
        arguments, as `ArgHasher.serialize_args` does for a default: that
        would key the parameters the user left out."""
        try:
            return self._args.hash_payload(*keyed)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            failure.append(e)
            return None

    def call_args_hash(self, spec: CachedFunction, args: tuple, kwargs: dict, built: BuiltKey) -> str | None:
        """The hash of every argument of the call, whatever ``key=`` or the
        ignored parameters left out: what the argument-mutation check compares
        after the body and what an entry records, so explain() can say a hit
        was matched by ``key=`` / ``ignore``. None when they cannot all be
        hashed."""
        if built.call_args_hash is not None or spec.arg_key is None:
            return built.call_args_hash
        try:
            return self._args.serialize_args(spec.name, args, kwargs, normalized=built.normalized_args)
        except Exception:  # a hash for the record only, never the key
            logger.debug("[CORE] could not hash every argument of %s", spec.name, exc_info=True)
            return None


class CallRunner:
    """The steps of a cached call, shared by the sync and async wrappers: the
    lookup, the body's scope, what a miss does after it, and the locked and
    single-flight paths of ``use_locking``."""

    def __init__(
        self,
        registry: FunctionRegistry,
        backend_slot: BackendSlot,
        keys: KeyBuilder,
        store: ResultStore,
        calls: CallLog,
        files: FileDeps,
        purity: PurityChecks,
        rng: RngWatch,
        misses: MissHistory,
        notices: Notices,
    ) -> None:
        self._registry = registry
        self._backend_slot = backend_slot
        self._keys = keys
        self._store = store
        self._calls = calls
        self._files = files
        self._purity = purity
        self._rng = rng
        self._misses = misses
        self._notices = notices
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

    def _run_uncached(self, spec: CachedFunction, call: Call, why: MissReason) -> Any:
        """Run a call that has no key, log it as a miss, and hand back its result."""
        note_unentered_body(spec.name)
        result = spec.func(*call.args, **call.kwargs)
        self._calls.log(
            spec.name,
            cache_hit=False,
            execution_time=_perf_counter() - call.call_start,
            args_hash="",
            cache_key="",
            miss=why,
        )
        return result

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

        Whether a found entry is served is `CallRunner.entry_verdict`.
        """
        if metadata is None:
            self._misses.note_miss(func_name, cache_key, self._misses.absent_entry_reason(func_name, cache_key))
            return CACHE_MISS
        try:
            verdict = self.entry_verdict(cache_key, metadata, ttl)
            if verdict is not None:
                self._misses.note_miss(func_name, cache_key, verdict)
                return CACHE_MISS
            ttl = self._backend_slot.entry_ttl(ttl, metadata)
            # If this hit happens *inside* another cached function's
            # computation, replay the files this entry depends on into the
            # enclosing tracker, so the outer function records them too.
            # Without this, a dependency that was already cached before the
            # consumer's first run hides its file deps behind a cache hit
            # and the consumer never invalidates when that file changes.
            propagate_file_deps_to_active_tracker(metadata, func_name)
            # Re-attach the lineage hash to the restored value. It's a plain
            # attribute that doesn't survive pickling, so a value restored
            # from disk would otherwise lose it - and a downstream cached
            # function would fall back to content-hashing under a DIFFERENT
            # key than when the upstream was freshly computed, recomputing
            # needlessly. The hash is deterministic from (cache_key,
            # auto_file_deps), both available here.
            cached_data = self._store.restore_identity(func_name, metadata, cached_data)
            self._store.attach_lineage(cached_data, cache_key, metadata.auto_file_deps, ttl=ttl, func_name=func_name)
            replay_rng_state(metadata)
            self._registry.cached[func_name].last_key = cache_key
            self._calls.log(
                func_name,
                cache_hit=True,
                execution_time=_perf_counter() - call_start,
                args_hash=args_hash,
                cache_key=cache_key,
                time_saved=(metadata.saves_seconds if metadata.saves_seconds is not None else metadata.execution_time)
                or 0.0,
            )
            return cached_data
        except (TypeError, KeyError) as e:
            self._notices.metadata_invalid(func_name, e)
            self._misses.note_miss(
                func_name, cache_key, MissReason(MissKind.INCOMPLETE, "the stored entry's metadata did not validate")
            )
        return CACHE_MISS

    def entry_verdict(
        self, cache_key: str, metadata: CacheMetadata, ttl: int | None, *, quiet: bool = False
    ) -> MissReason | None:
        """Why the entry found under *cache_key* is not served, or None to
        serve it: its age against the ttl, its recorded files, its chunks.

        The one judgement a lookup (`CallRunner._try_get_cached`) and
        ``explain()`` both apply, so a check added here holds for both.
        *quiet* leaves out the warnings the file check gives, for
        ``explain()``. Raises ``TypeError``/``KeyError`` on metadata that
        does not validate.
        """
        ttl = self._backend_slot.entry_ttl(ttl, metadata)
        if ttl_expired(written_at(metadata), ttl):
            age = time.time() - (written_at(metadata) or 0)
            return MissReason(MissKind.TTL, f"the entry is {age:.1f}s old and ttl={ttl}s")
        if not self._files.auto_file_deps_fresh(metadata, quiet=quiet):
            return MissReason(MissKind.FILE, describe_stale_files(metadata))
        if not self._chunks_are_intact(cache_key, metadata):
            return MissReason(MissKind.INCOMPLETE, "a chunk of the stored result is missing")
        return None

    def _chunks_are_intact(self, cache_key: str, metadata: CacheMetadata) -> bool:
        """True unless this is a chunked manifest missing some of its chunks.

        A manifest can outlive its chunks -- eviction reaches them separately
        -- and served as it is, such an entry returns FEWER items than it
        stored, or none at all, without a word. A truncated answer is worse
        than a slow one, so the entry is treated as absent and recomputed.

        Metadata-only reads: the point is to check presence, not to load the
        payload and undo the laziness chunking exists for.

        Scope, because the docs depend on it: BOTH read paths apply this --
        ``CallRunner._try_get_cached`` for the default one, and the double-checked re-read
        inside ``CallRunner.compute_with_lock`` for ``use_locking=True``, since both go
        through `CallRunner._try_get_cached`. Both paths are pinned by
        ``tests/test_core/results/test_iterator_caching.py``.
        """
        if getattr(metadata, "iterator_storage", None) != "chunked":
            return True
        try:
            prefix = chunk_prefix(cache_key, metadata.chunk_stream)
            for index in range(metadata.n_chunks or 0):
                if self._backend_slot.backend.get_metadata(f"{prefix}:chunk_{index}") is None:
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

    def _wrap_iterator_hit(self, call: Call, metadata: CacheMetadata | None, hit: Any) -> Any:
        """Wrap a cache-hit value in the right iterator class.

        Iterators (including the single-chunk case) are stored under
        an ``iterator_storage='chunked'`` manifest plus N chunk
        entries; on hit they're returned as a fresh
        ``ChunkedCachedIterator``, which recomputes the rest from the
        function if a chunk is gone. Non-iterator return types live as
        a single blob and are returned as *hit* directly.

        Every hit path goes through here -- the first lookup, the locked
        re-read (`CallRunner.compute_with_lock`) and the async single-flight follower
        (`CallRunner.single_flight`) -- and all of them pass the call's `recompute`, so a
        missing chunk is recomputed on every path rather than raised.
        """
        if metadata and metadata.iterator_storage == "chunked":
            n_chunks = metadata.n_chunks or 0
            prefix = chunk_prefix(call.cache_key, metadata.chunk_stream)
            returned = hit.get("return_value") if isinstance(hit, dict) else None
            return ChunkedCachedIterator(self._backend_slot, prefix, n_chunks, call.recompute, returned)
        return hit

    def _analyze_dependencies(self, func: Callable[..., Any]) -> None:
        """Populate analysis for *func* + its transitive cached-dependency
        closure, then surface *func*'s own purity issues.

        Populating the WHOLE closure (not just *func*) before the first cache
        key is computed is what keeps the key stable from the very first call.
        The state hash folds in each dependency's purity-report
        ``helper_source_hashes``; filled lazily on each dependency's own first
        call, the key would deepen only after the chain warmed, and a fresh
        process would miss the first call to every cached function even with a
        valid entry on disk.

        Surfacing stays per-function: each dependency warns/raises on its OWN
        first direct call, not here, so eager population doesn't change which
        warnings fire or when.
        """
        self._registry.ensure_closure_analyzed(func)
        func_name = func_key(func)
        report = self._registry.report_for(func, func_name) or PurityReport()
        mode = self._registry.purity_mode(func_name)
        self._purity.surface_purity(func_name, report, mode, per_report=bool(getattr(func, "__closure__", None)))

    def lookup(self, spec: CachedFunction, args: tuple, kwargs: dict, *, async_body: bool) -> Call:
        """Everything a call does before the body: analysis, key, lookup.

        Returns the call's state. ``call.outcome`` is what the wrapper returns
        now -- a hit, or the result of a call that has no key -- or
        ``CACHE_MISS`` when the body has to run.
        """
        func, func_name = spec.func, spec.name
        call = Call(args, kwargs)
        call.call_start = _perf_counter()
        if self._registry.needs_surfacing(func, func_name):
            # Double-checked under a per-function lock: the key is built
            # from what this populates, so two threads must not race it.
            # Once per name, and once per closure: another closure from a
            # factory already called is keyed by, and checked against, the
            # helpers it captures, not its sibling's.
            with self._registry.analysis_lock:
                if self._registry.needs_surfacing(func, func_name):
                    self._analyze_dependencies(func)
                    self._registry.mark_surfaced(func, func_name)
        # Inherit the shortest TTL of any TTL'd dependency (computed after
        # analysis populates the graph).
        call.ttl = self._registry.effective_ttl(func_name, spec.ttl)

        def recompute() -> Any:
            # The body runs here with no entry of its own (a stream whose
            # chunk went), under whatever tracker is active.
            note_unentered_body(func_name)
            if async_body:
                return run_to_completion(lambda: func(*args, **kwargs))
            return func(*args, **kwargs)

        call.recompute = recompute

        # Everything from here to the hit/miss verdict is cash's own cost,
        # not the user's work. Two perf_counter pairs cost under 1% of the
        # cheapest possible cached call, so this is not gated behind a
        # heuristic.
        overhead_t0 = _perf_counter()
        # Outside the key build, which turns any exception into "no key": an
        # exception from the body of an uncached call must propagate, not run
        # the body a second time.
        built, call.capture_watch = self._keys.resolve(func, func_name, spec.dynamic_depends_on, args, kwargs)
        if isinstance(built, Unkeyable):
            call.outcome = self._run_uncached(spec, call, built.reason)
            return call
        call.cache_key, call.state_hash, call.args_hash = built.cache_key, built.state_hash, built.args_hash

        raw_metadata, cached_data = self._backend_slot.read(call.cache_key)
        call.metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        hit = self._try_get_cached(
            call.cache_key, call.metadata, cached_data, call.call_start, call.args_hash, func_name, call.ttl
        )
        call.cash_overhead = _perf_counter() - overhead_t0
        if hit is not CACHE_MISS:
            self._replay_argument_rng(func_name, args, kwargs, call.metadata)
            self._calls.note_effectiveness(
                func_name,
                call.cash_overhead,
                body_seconds=getattr(call.metadata, "body_seconds", None),
                was_hit=True,
            )
            call.outcome = self._wrap_iterator_hit(call, call.metadata, hit)
        else:
            # Before the body runs: the mutation check compares against it.
            call.call_args_hash = self._keys.call_args_hash(spec, args, kwargs, built)
        return call

    def _reread(self, spec: CachedFunction, call: Call) -> Any:
        """Look the key up again (another caller may have stored it meanwhile):
        the hit, wrapped like any other, or ``CACHE_MISS``.

        The SAME validity test as the first lookup, by calling the same
        function -- not a hand-rolled subset of it, which would lose a check
        (``CallRunner._chunks_are_intact``, ``FileDeps.auto_file_deps_fresh``) as they are added.
        One function decides whether an entry may be served.
        """
        raw_metadata, cached_data = self._backend_slot.read(call.cache_key)
        if raw_metadata is None:
            return CACHE_MISS
        metadata = CacheMetadata.from_dict(raw_metadata)
        hit = self._try_get_cached(
            call.cache_key, metadata, cached_data, call.call_start, call.args_hash, spec.name, call.ttl
        )
        if hit is CACHE_MISS:
            return CACHE_MISS
        self._replay_argument_rng(spec.name, call.args, call.kwargs, metadata)
        return self._wrap_iterator_hit(call, metadata, hit)

    def _named_args(self, func_name: str, args: tuple, kwargs: dict) -> list[tuple[str, Any]]:
        """``(parameter, value)`` for each argument of the call, bound to the
        signature as the key binds it, so two spellings of one call agree."""
        try:
            canon_args, canon_kwargs = self._args.normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - unbindable: positions are all there is
            canon_args, canon_kwargs = args, kwargs
        return [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())

    def _replay_argument_rng(self, func_name: str, args: tuple, kwargs: dict, metadata: Any) -> None:
        """Move a generator handed in as an argument to where the computed
        call left it (`RngWatch.replay_parts`)."""
        moved = (getattr(metadata, "rng_replay", None) or {}).get("arg_carriers")
        if moved:
            replay_argument_carriers(moved, dict(self._named_args(func_name, args, kwargs)))

    @contextlib.contextmanager
    def body_scope(self, spec: CachedFunction, call: Call) -> Iterator[BodyRun]:
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
        func, func_name, args, kwargs = spec.func, spec.name, call.args, call.kwargs
        run = BodyRun()
        run.tracker = FileAccessTracker(getattr(func, "__globals__", None), propagate_to_parent=True, hash_on_read=True)
        # Watch for side effects the STATIC analyzer cannot see, which is
        # anything happening inside an installed library. Only on this
        # (missing) path: a hit runs no body, so there is nothing to observe
        # and nothing to pay for.
        run.observer = self._purity.make_effect_observer()
        run.observer.arg_snapshot = self._purity.argument_snapshot(func_name, args, kwargs)
        run.observer.arg_identities = self._purity.argument_identities(func_name, args, kwargs)
        # Watch the global RNG across the call: a draw inside the body is an
        # input the key cannot see statically.
        run.rng_pre = capture_rng_pre_state()
        # And the generators it can reach through its globals and closure
        # (`rng = np.random.default_rng(42)` at module level, drawn from
        # inside): the module channel above cannot see those move.
        run.carriers_pre = capture_reachable_carrier_states(func)
        run.carriers_moved = []
        # Or handed in as an argument (`boot(x, rng)`): a hit must move the
        # caller's generator on just the same.
        run.arg_carriers_pre = capture_argument_carrier_states(self._named_args(func_name, args, kwargs))
        run.arg_carriers_moved = []
        with run.tracker, run.observer:
            threads_at_start = THREADS_IN_CALLS[0]
            body_t0 = _perf_counter()
            nested = [0.0]
            nested_token = NESTED_CASH_SECONDS.set(nested)
            try:
                self._files.track_declared_files(run.tracker, func_name)
                yield run
            except Exception as exc:  # the user's body can raise anything; logged, then re-raised
                self._calls.log_raised(func_name, exc, call.call_start)
                raise
            finally:
                NESTED_CASH_SECONDS.reset(nested_token)
            # The user's own work, isolated. Everything cash does sits outside
            # this pair, which is the whole point: it is the only number that
            # can answer "did caching pay?".
            run.body_seconds = max(0.0, _perf_counter() - body_t0 - run.tracker.read_hash_seconds - nested[0])
            run.saves_seconds = run.body_seconds / max(threads_at_start, THREADS_IN_CALLS[0], 1)
            run.rng_new = self._rng.note_draw(func_name, run.rng_pre)
            run.carriers_moved = moved_carriers(run.carriers_pre)
            run.arg_carriers_moved = moved_carriers(run.arg_carriers_pre)

    def finish_miss(self, spec: CachedFunction, call: Call, run: BodyRun) -> Any:
        """Everything a missed call does after its body: check, store, log."""
        func, func_name, args, kwargs = spec.func, spec.name, call.args, call.kwargs
        res = run.res
        # A generator is handed straight back, wrapped, and cached only once
        # the caller has drained it. Draining it here instead meant a streamed
        # response arrived in one lump after the full latency, so
        # `@cash.cache` changed how the function behaved. `ResultStore.stream_and_store`
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
            self._calls.log(
                func_name,
                cache_hit=False,
                execution_time=_perf_counter() - call.call_start,
                args_hash=call.args_hash,
                cache_key=call.cache_key,
            )
            return StreamingCachedIterator(self._store.stream_and_store(res, spec, call, run))

        # A generator argument the body drew from is moved on again by a hit
        # (`replay_argument_carriers`), so that draw is not a mutation the
        # caller would lose: the check sees it where the call found it.
        with carriers_put_back(run.arg_carriers_pre if run.arg_carriers_moved else []):
            self._purity.check_argument_mutation(func_name, args, kwargs, call.call_args_hash, run.observer)
        self._purity.report_observed_effects(func_name, run.observer)
        self._files.credit_remembered_reads(func_name, run.tracker, args, kwargs)
        auto_file_deps = snapshot_tracked_deps(run.tracker, func.__module__)
        execution_time = _perf_counter() - call.call_start

        self._purity.warn_shared_result(func, func_name, res, args, kwargs)
        refusal = self._store.refusal(
            func, func_name, res, run.rng_new, spec.cache_if, run.tracker, call.capture_watch, observer=run.observer
        )
        if refusal is None and not replayable(run.carriers_moved + run.arg_carriers_moved):
            # A hit must leave the generator where the body did, or the
            # caller's next draw repeats what this call drew. One only a
            # closure holds cannot be found again by a later process.
            refusal = "it drew from a random generator a cached result cannot advance"
        if refusal is not None:
            self._misses.note_not_stored(call.cache_key, refusal)
        else:
            # Attach lineage only when the value is actually stored: a lineage
            # hash points downstream at THIS cache entry, so a cache_if-rejected
            # (uncached) value must not carry one - it would reference an entry
            # that was never written.
            self._store.attach_lineage(res, call.cache_key, auto_file_deps, ttl=call.ttl, func_name=func_name)
            self._store.store(
                StoreRequest(
                    call,
                    func_name,
                    execution_time=execution_time,
                    auto_file_deps=auto_file_deps,
                    body_seconds=run.body_seconds,
                    saves_seconds=run.saves_seconds,
                    rng_replay=self._rng.replay_parts(
                        bool(self._registry.cached[func_name].rng_modules),
                        run.rng_pre,
                        run.carriers_moved,
                        run.arg_carriers_moved,
                    ),
                ),
                res,
            )
        # A result the disk cap had evicted, computed again: say what that cost.
        if self._misses.pending_eviction(call.cache_key) is not None:
            self._notices.evicted_recompute(func_name, run.body_seconds)
        # Everything that was not the body: the key and lookup before it, the
        # checks and the store after it.
        miss_overhead = max(call.cash_overhead, _perf_counter() - call.call_start - run.body_seconds)
        self._calls.log(
            func_name,
            cache_hit=False,
            execution_time=execution_time,
            args_hash=call.args_hash,
            cache_key=call.cache_key,
            body_seconds=run.body_seconds,
            cash_seconds=miss_overhead,
        )
        self._calls.note_effectiveness(func_name, miss_overhead, body_seconds=run.body_seconds, was_hit=False)
        return res

    async def single_flight(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
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

    def compute_with_lock(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error.

        Acquiring the lock is best-effort: if *any* backend raises while taking
        it (a Redis ``LockError`` on contention/timeout, a dropped connection,
        an OSError on a file lock), we degrade to an unlocked compute rather than
        crash the user's call. Acquisition, compute, and release are separated so
        a release failure can't re-run the compute, and a compute exception
        propagates normally (it is not mistaken for a lock failure). Under the
        lock the key is looked up again by `CallRunner._reread`, the same test as the first
        lookup."""
        lock_cm = self._backend_slot.backend.lock(call.cache_key)
        try:
            lock_cm.__enter__()
        except Exception as e:  # noqa: BLE001 - any acquisition failure -> unlocked
            self._notices.lock_failed(spec.name, e)
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
                logger.debug("lock release failed for %s", spec.name)
