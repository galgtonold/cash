"""One cached call, from its key to its result: build the key, look it up,
run the body on a miss, and hand back what to return."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

from .._clock import perf_counter as _perf_counter
from ..backends import CacheMetadata
from ..backends._base import ttl_expired
from ..dependency_state import STATE_LEDGER, ledger_note
from ..exceptions import CashCacheIneffectiveWarning
from ..purity_analyzer import PurityReport
from ..tracking.file_tracker import FileAccessTracker
from .arg_hashing import PLAIN_CENSUS
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
from .code_identity import func_key
from .explain import MissKind, MissReason, describe_stale_files
from .file_deps import propagate_file_deps_to_active_tracker, snapshot_tracked_deps
from .iterators import ChunkedCachedIterator, StreamingCachedIterator, is_one_shot_iterator
from .registry import resolve_dynamic_dependencies
from .rng import capture_rng_pre_state, replay_rng_state

logger = logging.getLogger(__name__)


class Unkeyable(NamedTuple):
    """A call that gets no key, and the miss it is logged as."""

    reason: MissReason


_UNHASHABLE = MissReason(MissKind.UNHASHABLE, "an argument could not be hashed, so there is no key to look up")
_KEY_FAILED = MissReason(MissKind.KEY_FAILED, "building the key raised")


def entry_expired(metadata: CacheMetadata, ttl: int | None) -> bool:
    """Is the entry older than *ttl* (``ttl=0``: always)? The one TTL rule
    every cache path shares, `ttl_expired`."""
    return ttl_expired(metadata.timestamp, ttl)


def compute_cache_key(func_name: str, state_hash: str, dynamic_hash: str, args_hash: str) -> str:
    return f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"


class RuntimeMixin:
    """The steps of a cached call, shared by the sync and async wrappers."""

    def _resolve_cache_key(
        self,
        func: Callable,
        func_name: str,
        dynamic_depends_on: Callable[..., Any] | list[Callable[..., Any]] | None,
        args: tuple,
        kwargs: dict,
    ) -> tuple[BuiltKey | Unkeyable, dict]:
        """The key for a real call, or why it has none; and its `CAPTURE_WATCH`.

        `_build_key`, with a ledger of what the state segment is made of
        (`STATE_LEDGER`). A call with no key -- a mocked helper, an argument
        or default that cannot be hashed, a key build that raised -- has
        been warned about once; the caller runs it uncached.
        """
        mocked = self._registry.refresh_helper_bindings(func, func_name)
        if mocked is not None:
            return Unkeyable(
                MissReason(MissKind.MOCKED, f"{mocked}, which has no code to key, so the call ran uncached")
            ), {}
        ledger: dict = {}
        ledger_token = STATE_LEDGER.set(ledger)
        watch: dict = {}
        watch_token = CAPTURE_WATCH.set(watch)
        try:
            built = self._build_key(func, func_name, dynamic_depends_on, args, kwargs)
        except UnhashableDefault:
            # `ClosureFold.fold_defaults` has warned: an unhashable default means cash
            # cannot tell whether it changed, so caching at all risks a stale
            # result.
            return Unkeyable(_UNHASHABLE), watch
        except UnhashableArgs:
            self._args.warn_unhashable_args(func_name, args, kwargs)
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

    def _run_uncached(self, spec: CachedFunction, call: Call, why: MissReason) -> Any:
        """Run a call that has no key, log it as a miss, and hand back its result."""
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
        try:
            # The state after each fold, in `_STATE_STAGES` order: when no
            # named part moved, the first stage whose output did is the one
            # that changed (`describe_state_change`).
            chain: list[str] = []
            ledger_note("@chain", chain)
            state_hash = self._state_hasher.compute(
                func_name,
                own_source_override=self._code.pin_own_source(func),
                note=True,
            )
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
            chain.append(state_hash)
            state_hash = self._globals.fold_read_globals(func, func_name, state_hash)
            state_hash = self._globals.fold_helper_read_globals(func, func_name, state_hash)
            state_hash = self._globals.fold_dependency_read_globals(func, func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._rng.fold_rng_epoch(func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._globals.fold_environment(func_name, state_hash)
            chain.append(state_hash)
            state_hash = self._code.fold_method_class_deps(func, args, state_hash)
            chain.append(state_hash)
            # ONE canonicalisation, fed to both the code channel and the value
            # channel. `CodeArgs.fold_code_args` on the RAW arguments saw a class
            # passed explicitly but not the identical class arriving as a
            # parameter DEFAULT, so `build()` and `build(Schema)` -- the same
            # logical call -- produced two cache keys and two executions.
            normalized_args = self._args.normalize_call_args(func_name, args, kwargs)
            if self._registry.cached[func_name].seed_params:
                self._rng.warn_if_seed_is_none(func, func_name, args, kwargs)
            state_hash = self._code_args.fold_code_args(*normalized_args, state_hash, func_name=func_name)
            chain.append(state_hash)
            dynamic_state_hash = resolve_dynamic_dependencies(func_name, dynamic_depends_on, args, kwargs)
            args_hash = self._args.serialize_args(func_name, args, kwargs, normalized=normalized_args)
            self._args.note_arg_cost(func_name)
        finally:
            PLAIN_CENSUS.memo = previous
        if args_hash is None:
            raise UnhashableArgs
        cache_key = compute_cache_key(func_name, state_hash, dynamic_state_hash, args_hash)
        return BuiltKey(cache_key, state_hash, args_hash, normalized_args)

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
            self._misses.note_miss(func_name, cache_key, self._misses.absent_entry_reason(func_name, cache_key))
            return CACHE_MISS
        ttl = self._entry_ttl(ttl, metadata)
        try:
            if entry_expired(metadata, ttl):
                age = time.time() - (metadata.timestamp or 0)
                self._misses.note_miss(
                    func_name, cache_key, MissReason(MissKind.TTL, f"the entry is {age:.1f}s old and ttl={ttl}s")
                )
                return CACHE_MISS
            if not self._files.auto_file_deps_fresh(metadata):
                self._misses.note_miss(func_name, cache_key, MissReason(MissKind.FILE, describe_stale_files(metadata)))
                return CACHE_MISS
            if not self._chunks_are_intact(cache_key, metadata):
                self._misses.note_miss(
                    func_name, cache_key, MissReason(MissKind.INCOMPLETE, "a chunk of the stored result is missing")
                )
                return CACHE_MISS
            # If this hit happens *inside* another cached function's
            # computation, replay the files this entry depends on into the
            # enclosing tracker, so the outer function records them too.
            # Without this, a dependency that was already cached before the
            # consumer's first run hides its file deps behind a cache hit
            # and the consumer never invalidates when that file changes.
            propagate_file_deps_to_active_tracker(metadata)
            # Re-attach the lineage hash to the restored value. It's a plain
            # attribute that doesn't survive pickling, so a value restored
            # from disk would otherwise lose it - and a downstream cached
            # function would fall back to content-hashing under a DIFFERENT
            # key than when the upstream was freshly computed, recomputing
            # needlessly. The hash is deterministic from (cache_key,
            # auto_file_deps), both available here.
            self._attach_lineage(cached_data, cache_key, metadata.auto_file_deps, ttl=ttl, func_name=func_name)
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
                file_deps=metadata.auto_file_deps,
            )
            return cached_data
        except (TypeError, KeyError) as e:
            self._notices.metadata_invalid(func_name, e)
            self._misses.note_miss(
                func_name, cache_key, MissReason(MissKind.INCOMPLETE, "the stored entry's metadata did not validate")
            )
        return CACHE_MISS

    def _tier_default_ttl(self) -> int | None:
        """The ``default_ttl`` of the first tier that has one, as configured now."""
        backend = self._backend_slot.built
        return backend.default_ttl if backend is not None else None

    def _entry_ttl(self, ttl: int | None, metadata: Any) -> int | None:
        """The ttl a stored entry is judged by.

        The decorator's ``ttl=`` when it has one -- a per-function setting,
        applied as it stands now, in both directions. Otherwise the SHORTER of
        the ttl the entry was written with and the tier's ``default_ttl`` as
        configured now: lowering a tier's default from a day to 5 seconds left
        every entry written under the day being served,
        while lowering a decorator's ttl took effect at once.
        """
        if ttl is not None:
            return ttl
        found = [t for t in (getattr(metadata, "ttl", None), self._tier_default_ttl()) if t is not None]
        return min(found) if found else None

    def _chunks_are_intact(self, cache_key: str, metadata: CacheMetadata) -> bool:
        """True unless this is a chunked manifest missing some of its chunks.

        A manifest can outlive its chunks -- eviction reaches them separately
        -- and served as it is, such an entry returns FEWER items than it
        stored, or none at all, without a word. A truncated answer is worse
        than a slow one, so the entry is treated as absent and recomputed.

        Metadata-only reads: the point is to check presence, not to load the
        payload and undo the laziness chunking exists for.

        Scope, because the docs depend on it: BOTH read paths apply this --
        ``_try_get_cached`` for the default one, and the double-checked re-read
        inside ``_compute_with_lock`` for ``use_locking=True``, since both go
        through `_try_get_cached`. Both paths are pinned by
        ``tests/test_core/test_iterator_caching.py``.
        """
        if getattr(metadata, "iterator_storage", None) != "chunked":
            return True
        try:
            for index in range(metadata.n_chunks or 0):
                if self._backend_slot.backend.get_metadata(f"{cache_key}:chunk_{index}") is None:
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
        re-read (`_compute_with_lock`) and the async single-flight follower
        (`_single_flight`) -- and all of them pass the call's `recompute`, so a
        missing chunk is recomputed on every path rather than raised.
        """
        if metadata and metadata.iterator_storage == "chunked":
            n_chunks = metadata.n_chunks or 0
            return ChunkedCachedIterator(self, call.cache_key, n_chunks, call.recompute)
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
        report = self._registry.purity_reports.get(func_name) or PurityReport()
        mode = self._registry.purity_mode(func_name)
        self._purity.surface_purity(func_name, report, mode)

    def _lookup(self, spec: CachedFunction, args: tuple, kwargs: dict, *, async_body: bool) -> Call:
        """Everything a call does before the body: analysis, key, lookup.

        Returns the call's state. ``call.outcome`` is what the wrapper returns
        now -- a hit, or the result of a call that has no key -- or
        ``CACHE_MISS`` when the body has to run.
        """
        func, func_name = spec.func, spec.name
        call = Call(args, kwargs)
        call.call_start = _perf_counter()
        if func_name not in self._registry.analyzed:
            # Double-checked under a per-function lock: the key is built
            # from what this populates, so two threads must not race it.
            with self._registry.analysis_lock:
                if func_name not in self._registry.analyzed:
                    self._analyze_dependencies(func)
                    self._registry.analyzed.add(func_name)
        # Inherit the shortest TTL of any TTL'd dependency (computed after
        # analysis populates the graph).
        call.ttl = self._registry.effective_ttl(func_name, spec.ttl)
        if async_body:
            call.recompute = lambda: run_to_completion(lambda: func(*args, **kwargs))
        else:
            call.recompute = lambda: func(*args, **kwargs)

        # Everything from here to the hit/miss verdict is cash's own cost,
        # not the user's work. Two perf_counter pairs measured at 196ns
        # against a 25.5us floor for the cheapest possible cached call --
        # 0.8%, so this is not gated behind a heuristic.
        overhead_t0 = _perf_counter()
        # Outside the key build, which turns any exception into "no key": an
        # exception from the body of an uncached call must propagate, not run
        # the body a second time.
        built, call.capture_watch = self._resolve_cache_key(func, func_name, spec.dynamic_depends_on, args, kwargs)
        if isinstance(built, Unkeyable):
            call.outcome = self._run_uncached(spec, call, built.reason)
            return call
        call.cache_key, call.state_hash, call.args_hash = built.cache_key, built.state_hash, built.args_hash

        raw_metadata, cached_data = self._backend_slot.backend.get(call.cache_key)
        call.metadata = CacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        hit = self._try_get_cached(
            call.cache_key, call.metadata, cached_data, call.call_start, call.args_hash, func_name, call.ttl
        )
        call.cash_overhead = _perf_counter() - overhead_t0
        if hit is not CACHE_MISS:
            self._calls.note_effectiveness(
                func_name,
                call.cash_overhead,
                body_seconds=getattr(call.metadata, "body_seconds", None),
                was_hit=True,
            )
            call.outcome = self._wrap_iterator_hit(call, call.metadata, hit)
        return call

    def _reread(self, spec: CachedFunction, call: Call) -> Any:
        """Look the key up again (another caller may have stored it meanwhile):
        the hit, wrapped like any other, or ``CACHE_MISS``.

        The SAME validity test as the first lookup, by calling the same
        function -- not a hand-rolled subset of it, which would lose a check
        (``_chunks_are_intact``, ``FileDeps.auto_file_deps_fresh``) as they are added.
        One function decides whether an entry may be served.
        """
        raw_metadata, cached_data = self._backend_slot.backend.get(call.cache_key)
        if raw_metadata is None:
            return CACHE_MISS
        metadata = CacheMetadata.from_dict(raw_metadata)
        hit = self._try_get_cached(
            call.cache_key, metadata, cached_data, call.call_start, call.args_hash, spec.name, call.ttl
        )
        if hit is CACHE_MISS:
            return CACHE_MISS
        return self._wrap_iterator_hit(call, metadata, hit)

    @contextlib.contextmanager
    def _body_scope(self, spec: CachedFunction, call: Call) -> Iterator[BodyRun]:
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
        with run.tracker, run.observer:
            threads_at_start = THREADS_IN_CALLS[0]
            body_t0 = _perf_counter()
            nested = [0.0]
            nested_token = NESTED_CASH_SECONDS.set(nested)
            try:
                self._files.track_declared_files(run.tracker, func_name)
                yield run
            except Exception as exc:  # noqa: BLE001 - the user's body can raise anything; logged, then re-raised
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

    def _finish_miss(self, spec: CachedFunction, call: Call, run: BodyRun) -> Any:
        """Everything a missed call does after its body: check, store, log."""
        func, func_name, args, kwargs = spec.func, spec.name, call.args, call.kwargs
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
            self._calls.log(
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
                    spec=spec,
                    tracker=run.tracker,
                    observer=run.observer,
                    rng_new=run.rng_new,
                    args=args,
                    kwargs=kwargs,
                    args_hash=call.args_hash,
                    current_state_hash=call.state_hash,
                    ttl=call.ttl,
                )
            )

        self._purity.check_argument_mutation(func_name, args, kwargs, call.args_hash, run.observer)
        self._purity.report_observed_effects(func_name, run.observer)
        self._files.credit_remembered_reads(func_name, run.tracker, args, kwargs)
        auto_file_deps = snapshot_tracked_deps(run.tracker, func.__module__)
        execution_time = _perf_counter() - call.call_start

        self._purity.warn_shared_result(func, func_name, res, args, kwargs)
        refusal = self._store_refusal(
            func, func_name, res, run.rng_new, spec.cache_if, run.tracker, call.capture_watch, observer=run.observer
        )
        if refusal is not None:
            self._misses.note_not_stored(call.cache_key, refusal)
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
                call.ttl,
                call.state_hash,
                call.args_hash,
                execution_time,
                auto_file_deps=auto_file_deps,
                body_seconds=run.body_seconds,
                saves_seconds=run.saves_seconds,
                rng_replay=self._rng.replay_parts(bool(self._registry.cached[func_name].rng_modules), run.rng_pre),
            )
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

    async def _single_flight(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
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

    def _compute_with_lock(self, spec: CachedFunction, call: Call, compute: Callable[[], Any]) -> Any:
        """Compute with double-checked locking; falls back to unlocked on error.

        Acquiring the lock is best-effort: if *any* backend raises while taking
        it (a Redis ``LockError`` on contention/timeout, a dropped connection,
        an OSError on a file lock), we degrade to an unlocked compute rather than
        crash the user's call. Acquisition, compute, and release are separated so
        a release failure can't re-run the compute, and a compute exception
        propagates normally (it is not mistaken for a lock failure). Under the
        lock the key is looked up again by `_reread`, the same test as the first
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
