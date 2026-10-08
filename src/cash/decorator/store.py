"""Storing a computed result: whether to, whole or in chunks, and the
lineage tag it carries."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import logging
import secrets
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import identity_refs
from .._clock import perf_counter as _perf_counter
from .._memo import RESULT_TYPES, LruMemo
from ..backends.memory_backend import InMemoryBackend
from ..backends.serialization import PickleSerializer
from ..backends.tiered_backend import TieredBackend
from ..effect_observer import EffectObserver
from ..exceptions import CacheBackendError, CashCacheIneffectiveWarning, CashCacheStoreFailedWarning
from ..lineage_tag import set_tags
from ..sizing import estimate_object_size
from ..value_types import IMMUTABLE_PRIMS
from .arg_hashing import LINEAGE_SRC_DECORATOR, LINEAGE_SRC_FROZEN
from .cache_metadata import CacheMetadata
from .cached_function import CachedFunction
from .call_state import NO_WATCH, BodyRun, Call
from .dynamic_sources import DynamicSources, recorded_sources, remember_sources, source_key
from .explain import not_persisted_reason
from .file_deps import snapshot_tracked_deps
from .iterators import chunk_prefix
from .purity_checks import held_sentinels, sentinel_ref

if TYPE_CHECKING:
    from .backend_slot import BackendSlot
    from .explain import MissHistory
    from .file_deps import FileDeps
    from .frozen import FrozenResults
    from .purity_checks import PurityChecks
    from .registry import FunctionRegistry
    from .reporting import Notices
    from .stored_keys import StoredKeyRecord

logger = logging.getLogger(__name__)

STORE_FAILED_FIX = (
    "read the exception: a full disk, a cache_dir you cannot write to, or a "
    "value that cannot be pickled -- return the data, not the handle that "
    "produced it."
)


#: Result types seen to refuse an attribute (dict, list, ndarray, ...): not
#: tried again (`ResultStore.attach_lineage`).
UNTAGGABLE_TYPES: LruMemo[type, bool] = LruMemo(RESULT_TYPES)


def _read_only_array(value: Any) -> bool | None:
    """False/True for a writable/read-only numpy array, None for anything else."""
    value_type = type(value)
    if value_type.__name__ != "ndarray" or not (value_type.__module__ or "").startswith("numpy"):
        return None
    return not value.flags.writeable


def _snapshot(item: Any) -> Any:
    """A copy of a streamed *item* no later edit reaches, or *item* if it cannot be copied.

    An item that cannot be copied cannot be pickled either: its chunk's write
    fails and says so (STORE-CHUNK-FAILED), and the caller keeps its item.
    """
    if type(item) in IMMUTABLE_PRIMS:
        return item
    try:
        return InMemoryBackend._safe_deep_copy(item)
    except Exception:  # noqa: BLE001 - a copy is best effort; the write reports the failure
        return item


def _returned(value: Any) -> dict[str, Any]:
    """The manifest's record of a generator's return value (none for ``None``)."""
    return {} if value is None else {"return_value": value}


def lineage_hash(cache_key: str, auto_file_deps: dict | None) -> str:
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


@dataclass(frozen=True)
class StoreRequest:
    """One result to store: the call it answers (its key, ttl and key
    segments, `Call`), and what the entry's metadata records besides."""

    call: Call
    func_name: str
    #: The wall-clock cost of the call; *body_seconds* replaces it when known.
    execution_time: float = 0.0
    auto_file_deps: dict[str, dict[str, float]] | None = None
    body_seconds: float | None = None
    saves_seconds: float | None = None
    rng_replay: dict[str, Any] | None = None
    #: The fields that make the entry a chunked iterator's manifest
    #: (``iterator_storage``, ``n_chunks``, ``chunk_stream``).
    manifest: dict[str, Any] | None = None
    #: Why one of a manifest's chunks stayed in RAM, which leaves the entry
    #: RAM-only whatever happens to the manifest itself.
    chunks_not_persisted: str | None = None
    #: The non-file ``dynamic_depends_on=`` sources of the cached functions
    #: the call ran (`recorded_sources`).
    dynamic_sources: DynamicSources | None = None


@dataclass
class _ChunkStream:
    """One iterator result's chunks while it streams (`ResultStore.stream_and_store`)."""

    #: The stream's `chunk_prefix`, and the id in it the manifest records.
    prefix: str
    stream: str
    #: The open chunk's items, and their estimated size.
    buffer: list[Any] = field(default_factory=list)
    buffer_bytes: int = 0
    #: Chunks written so far, the ones to drop if the stream is abandoned.
    written: int = 0
    total_items: int = 0
    #: The producer's time: only the spans inside `next()`.
    produced_seconds: float = 0.0
    #: Why a chunk written so far stayed in RAM, if one did.
    not_persisted: str | None = None


class ResultStore:
    """Deciding whether and how to store a result -- whole, or streamed in
    chunks -- and tagging it with its lineage."""

    def __init__(
        self,
        registry: FunctionRegistry,
        backend_slot: BackendSlot,
        frozen: FrozenResults,
        files: FileDeps,
        purity: PurityChecks,
        misses: MissHistory,
        notices: Notices,
        stored_keys: StoredKeyRecord,
    ) -> None:
        self._registry = registry
        self._backend_slot = backend_slot
        self._frozen = frozen
        self._files = files
        self._purity = purity
        self._misses = misses
        self._notices = notices
        self._stored_keys = stored_keys

    def refusal(
        self,
        func: Callable,
        func_name: str,
        res: Any,
        rng_new: bool,
        cache_if: Callable[[Any], bool] | None,
        tracker: Any,
        capture_watch: Any = NO_WATCH,
        observer: EffectObserver | None = None,
    ) -> str | None:
        """Why *res* must not be stored, or ``None`` to store it.

        One decision for the sync, async and streaming paths, and it says WHY,
        because "not stored" is the answer to the next call's "why did that
        miss?".
        """
        # Skip the write exactly once when THIS call revealed that the
        # function draws: its key was built before we knew, so an entry stored
        # now carries no seed epoch and would be rebuilt and matched forever --
        # serving a result computed under a seed the user has since changed.
        # The next call keys it correctly.
        if inspect.isawaitable(res):
            return self._refuse_awaitable(func_name, res)
        refusal = (
            "its first call drew random numbers the key did not yet cover; the next call keys them" if rng_new else None
        )
        if refusal is None and cache_if is not None:
            try:
                refusal = None if cache_if(res) else "cache_if returned False"
            except Exception as e:  # noqa: BLE001 - user predicate
                self._notices.cache_if_raised(func_name, e)
                refusal = "cache_if raised"
        # After the body ran, before deciding to store: a provisional global
        # this call moved must stop being folded.
        if capture_watch is not NO_WATCH:
            self._purity.learn_mutating_captures(func, func_name, capture_watch)
        if refusal is None and self._purity.refuses_identity_coupled(func_name, res):
            refusal = "the result is tied to the identity of an object in memory"
        if refusal is None and self._files.inputs_moved_during_call(func_name, tracker):
            refusal = "a file it read changed while it ran"
        stale_memo = getattr(tracker, "stale_memo_reads", None)
        if refusal is None and stale_memo:
            refusal = (
                f"a memoised helper handed it data read from an earlier version of "
                f"{sorted(stale_memo)[0]}; a fresh process reads the file as it is now"
            )
        unresolved = getattr(tracker, "unresolved_dynamic", None)
        if refusal is None and unresolved:
            refusal = (
                f"a cached function it calls depends, through dynamic_depends_on=, on sources "
                f"nothing could record ({sorted(unresolved)[0]}), so nothing could tell when they change"
            )
        if refusal is None:
            refusal = self._unpicklable_source_refusal(func_name, recorded_sources(tracker))
        if refusal is None and self._files.code_moved_since_keyed(func, func_name):
            refusal = "its code changed on disk after this process keyed it"
        if refusal is None and observer is not None and observer.mock_called:
            # Wherever the mock sat -- below the library call the body makes,
            # or swapped in after the key's bindings were read -- the result
            # may be a test's fake, and the next real run would be served it.
            # Not waivable: no audit makes a fake the answer.
            refusal = "a unittest.mock object was called while it ran, so the result may be a test's fake"
        mutated = observer.mutated_args if observer is not None else None
        if refusal is None and mutated and self._registry.purity_mode(func_name) != "silent":
            # A hit returns the stored value and leaves the caller's object as
            # it was, where this call changed it: downstream of the call, the
            # program then differs between a hit and a miss (`a -= a.mean()`,
            # `rng.shuffle(a)`, `np.clip(..., out=a)`). Not
            # storing makes every call run, which is what the code means.
            # `assume_safe=True` is the audited opt-out.
            names = ", ".join(repr(n) for n in mutated)
            refusal = (
                f"it changed its argument{'s' if len(mutated) > 1 else ''} {names} in place, which a hit would not do"
            )
        return refusal

    def _unpicklable_source_refusal(self, func_name: str, sources: DynamicSources | None) -> str | None:
        """Why the result is not stored when a cached function this call ran
        depends, through ``dynamic_depends_on=``, on a source that cannot be
        pickled with the entry, or None to store it.

        Only this process can ask such a source for its token, so the entry
        stays in RAM (`PersistencePolicy.decide`, ``process_local``). A
        backend with no RAM tier would write it where only this process can
        use it: not stored. Warned once either way, unless the backend keeps
        nothing past the process anyway."""
        if sources is None or sources.picklable:
            return None
        backend = self._backend_slot.backend
        if isinstance(backend, InMemoryBackend):
            return None
        kept = isinstance(backend, TieredBackend) and isinstance(backend.backends[0], InMemoryBackend)
        ids = sources.unpicklable_ids
        shown = ", ".join(ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
        what = (
            "The result is kept in memory for this process only and not written to disk."
            if kept
            else "The result was returned but not cached."
        )
        self._notices.warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "untracked_source",
            f"@cash.cache on {func_name}: a cached function it calls depends on {shown} "
            f"through dynamic_depends_on=, and that source cannot be pickled, so a later "
            f"process could not ask it whether it changed. {what}",
            code="STORE-UNTRACKED-SOURCE",
            fix="make the source picklable (no open connection or lock held as an attribute: "
            "open it in state_token()), or call that function outside this one and pass its "
            "result in as an argument.",
        )
        if kept:
            return None
        return f"a cached function it calls depends on {ids[0]}, which cannot be pickled with the entry"

    def _result_ref(self, func_name: str, result: Any) -> list | None:
        """``["global" | "closure", name]`` when the body returns that sentinel (`sentinel_ref`)."""
        spec = self._registry.cached.get(func_name)
        if spec is None:
            return None
        return sentinel_ref(inspect.unwrap(spec.func), result)

    def _held_refs(self, func_name: str, result: Any) -> list | None:
        """The sentinels the body names that *result* holds inside it (`identity_refs.find`)."""
        spec = self._registry.cached.get(func_name)
        if spec is None:
            return None
        named = held_sentinels(inspect.unwrap(spec.func))
        return identity_refs.find(result, named) if named else None

    def restore_identity(self, func_name: str, metadata: CacheMetadata, value: Any) -> Any:
        """What a hit hands back for *value*: the identity and flags the result had.

        A numpy array stored read-only is read-only again, and a sentinel
        stored as a module global or closure variable is that variable's
        object, when it still holds one of the same type (`CacheMetadata.result_ref`).
        """
        if metadata.read_only and _read_only_array(value) is False:
            try:
                value.flags.writeable = False
            except (AttributeError, ValueError):
                pass
        if metadata.held_refs:
            spec = self._registry.cached.get(func_name)
            if spec is not None:
                value = identity_refs.put_back(value, metadata.held_refs, _resolver(inspect.unwrap(spec.func)))
        ref = metadata.result_ref
        if ref and len(ref) == 2:
            spec = self._registry.cached.get(func_name)
            if spec is not None:
                func = inspect.unwrap(spec.func)
                kind, name = ref
                try:
                    if kind == "closure":
                        cells = dict(zip(func.__code__.co_freevars, func.__closure__ or ()))
                        current = cells[name].cell_contents
                    else:
                        current = func.__globals__[name]
                except (AttributeError, KeyError, ValueError):
                    return value
                if type(current) is type(value):
                    return current
        return value

    def _refuse_awaitable(self, func_name: str, res: Any) -> str:
        """CACHE-RETURNS-AWAITABLE: a sync function handed back a coroutine.

        ``@cash.cache`` over an ordinary sync wrapper (a retry or timing
        decorator) around an ``async def`` gets the coroutine, not its value:
        the wrapper is not a coroutine function, so the sync path ran and
        tried to pickle the coroutine on every call (STORE-FAILED, whose "return
        the data" fix does not apply). The body has not run yet -- the caller
        awaits it -- so nothing can be stored; the call goes through uncached.
        """
        kind = type(res).__name__
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: result not cached. It returned a {kind} "
            f"to await, not a value: the function under @cash.cache is a plain "
            f"(sync) function, typically a wrapper around an async def, so the "
            f"value is only produced after cash has returned.",
            code="CACHE-RETURNS-AWAITABLE",
            fix="put @cash.cache directly on the async def, under the wrapper: "
            "@retry above @cash.cache above async def.",
        )
        return f"it returned a {kind} to await, not a value"

    def attach_lineage(
        self,
        result: Any,
        cache_key: str,
        auto_file_deps: dict | None = None,
        ttl: int | None = None,
        func_name: str | None = None,
    ) -> None:
        """Tag *result* with its lineage hash, beside it (`cash.lineage_tag.set_tags`).

        Works with pandas, polars and modin frames and any other object that
        has a ``__dict__`` and takes a weak reference; the object itself is
        never changed.

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
            # _cash_lineage_hash to int".
            return
        frozen = self._registry.is_frozen(func_name)
        if frozen and type(result) in (list, tuple, dict):
            self._frozen.remember_container(result, func_name, lineage_hash(cache_key, auto_file_deps))
            return
        if frozen and type(result).__name__ == "ndarray" and (type(result).__module__ or "").startswith("numpy"):
            # An array cannot carry a tag, and read-only is a promise numpy
            # enforces: a write raises instead of going stale.
            try:
                self._frozen.remember_array(result, func_name)
            except (AttributeError, TypeError, ValueError):
                pass
            return
        if not frozen and UNTAGGABLE_TYPES.get(type(result)):
            return
        lineage = lineage_hash(cache_key, auto_file_deps)
        # Beside the value, never on it (`cash.lineage_tag.set_tags`): as
        # attributes the tags showed in the user's own data -- `vars()`,
        # a `__dict__`-based `==` or `repr`, the pickle -- and a function
        # returning its argument tagged the caller's object. Say who wrote
        # it: nothing will move this tag when the value is mutated, so
        # `ArgHasher.hash_payload` must not take it for the content -- unless
        # the function was declared frozen=True.
        tags = {
            "_cash_lineage_src": LINEAGE_SRC_FROZEN if frozen else LINEAGE_SRC_DECORATOR,
            "_cash_lineage_hash": lineage,
        }
        if func_name is not None:
            # Named in CACHE-NET-LOSS and KEY-FROZEN-MUTATED.
            tags["_cash_lineage_producer"] = func_name
        if not set_tags(result, **tags):
            if frozen:
                self._frozen.warn_has_no_effect(func_name, result)
            # Once per type, then never tried again: it logged on every call
            # returning a dict or an array, and meant nothing to the user
            # reading CASH_DEBUG.
            UNTAGGABLE_TYPES[type(result)] = True
            logger.debug(
                "results of type %s cannot carry a lineage tag, so a cached function taking one hashes its content",
                type(result).__name__,
            )

    def store(self, request: StoreRequest, result: Any) -> bool:
        """Store *result* for *request*; True if a backend took it."""
        call, func_name = request.call, request.func_name
        cache_key, ttl = call.cache_key, call.ttl
        execution_time, body_seconds = request.execution_time, request.body_seconds
        dynamic = request.dynamic_sources
        try:
            serializer = PickleSerializer()

            # What a later hit saves is the BODY's time. The wall-clock cost
            # also holds cash's own work -- the first call's analysis, the key
            # -- which the next process pays again whether this entry exists
            # or not. Judged on wall-clock, a function that returns at once
            # would be persisted whenever a busy machine made that first-call
            # work cross the 0.1s floor.
            if body_seconds is not None:
                execution_time = body_seconds
            # The ttl the entry is WRITTEN with, a tier's default included: a
            # backend drops an expired entry on read, so the next miss can only
            # say "expired" -- rather than "evicted or cleared" -- if this
            # process and the stored-key record know it.
            ttl_declared = ttl is not None or None
            if ttl is None:
                ttl = self._backend_slot.tier_default_ttl()
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
                saves_seconds=request.saves_seconds,
                serializer_cls=type(serializer),
                ttl=ttl,
                ttl_declared=ttl_declared,
                args_hash=call.args_hash,
                # Every argument's hash, when ``key=`` decided ``args_hash``:
                # a later hit whose arguments differ was matched by it, which
                # explain() says. Parameters left out with ``ignore=`` are
                # never hashed, so there is nothing to record for them.
                call_args_hash=(
                    call.call_args_hash
                    if getattr(getattr(self._registry.cached.get(func_name), "arg_key", None), "key_fn", None)
                    is not None
                    else None
                ),
                state_hash=call.state_hash,
                # Each entry: path -> {'mtime': float, 'size': int}.
                # Validated on subsequent get() via FileDeps.auto_file_deps_fresh.
                auto_file_deps=request.auto_file_deps or None,
                rng_replay=request.rng_replay or None,
                # Decorating a function IS the decision to cache it, however
                # quick it is. The compute floor belongs to the notebook, where
                # cash caches every statement by itself; here it meant a script
                # run twice recomputed everything, which reads as "cash does
                # not cache".
                # Size caps and the tiers' own refusals still apply.
                decorator_entry=True,
                # A SEPARATE promise: the stored value is what the next call
                # hands back, so the RAM tier refuses one it cannot copy rather
                # than sharing it. Not for a `frozen=True` function: declaring
                # a result frozen says it is not modified, and handing the same
                # object back is what that promises for a result no pickle can
                # copy at all. Kept apart from `decorator_entry`, so that
                # `frozen=True` does not look undecorated to the rate ceiling
                # and lose disk persistence.
                copy_required=not self._registry.is_frozen(func_name),
                read_only=_read_only_array(result) or None,
                result_ref=(result_ref := self._result_ref(func_name, result)),
                held_refs=None if result_ref else self._held_refs(func_name, result),
                # The non-file sources its cached callees resolved, with their
                # tokens: a lookup asks them again (`dynamic_sources_fresh`).
                dynamic_sources=(dynamic.records or None) if dynamic is not None else None,
                process_local=True if dynamic is not None and not dynamic.picklable else None,
                **(request.manifest or {}),
            )

            # Kept, not a temporary: TieredBackend writes back where the value
            # landed, and "RAM only" is the answer to the next process's miss.
            meta_dict = meta.to_dict()
            if dynamic is not None:
                self._store_sources(dynamic)
            self._backend_slot.backend.set(cache_key, result, meta_dict, serializer=serializer)
            if dynamic is not None:
                remember_sources(cache_key, dynamic)
            # A tiered backend catches each tier's failure so one bad tier
            # cannot break a call; it reports them here instead, and a result
            # nothing could store is a STORE-FAILED like any other.
            store_errors = meta_dict.get("store_errors")
            if store_errors and not [t for t in (meta_dict.get("storage") or []) if t != "RAM"]:
                raise CacheBackendError("; ".join(str(e) for e in store_errors))
            not_persisted = not_persisted_reason(meta_dict) or request.chunks_not_persisted
            self._misses.remember_outcome(
                cache_key,
                {
                    "stored_at": time.time(),
                    "ttl": ttl,
                    "not_persisted": not_persisted,
                },
            )
            ledger = functools.partial(self._misses.flat_ledger, func_name)
            if not_persisted is None:
                self._stored_keys.note_stored(func_name, cache_key, ttl, ledger)
            else:
                self._stored_keys.note_ram_only(func_name, cache_key, not_persisted, ledger)
            return True
        except Exception as e:  # noqa: BLE001 - a failed store must never fail the call
            # Not only I/O errors: a bare FileBackend or SQLiteBackend pickles
            # on this thread, and pickle raises AttributeError for a local
            # object and whatever a __reduce__ raises. The body has run; its
            # result goes back to the caller whatever happens here.
            self._misses.note_not_stored(cache_key, "the backend refused the write")
            backend_name = type(self._backend_slot.backend).__name__
            self._notices.warn_once(
                CashCacheStoreFailedWarning,
                func_name,
                "",
                f"@cash.cache on {func_name}: backend {backend_name} failed to store "
                f"result ({type(e).__name__}: {e}). Compute succeeded, nothing was "
                f"stored, and the next call recomputes.",
                code="STORE-FAILED",
                fix=STORE_FAILED_FIX,
            )
            return False

    def _warn_cache_if_bypassed(self, spec: CachedFunction) -> None:
        """One-shot: the result outgrew a single chunk, so cache_if cannot run.

        Applying it would mean materializing every chunk back into memory,
        undoing the bound that chunking exists to provide.

        The fix line says **raise** the thresholds, and that direction is
        load-bearing. ``cache_if`` is consulted only in the ``chunk_index == 0``
        branch of ``ResultStore.stream_and_store`` -- the whole result fit one chunk -- and
        this fires at ``chunk_index == 1``, once it did not. Lowering the
        thresholds would produce more chunks and so guarantee the very bypass
        it is warning about.
        """
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            spec.name,
            "",
            f"@cash.cache on {spec.name}: the result exceeded a single chunk "
            f"(chunk_max_items={spec.chunk_max_items}, "
            f"chunk_max_bytes={spec.chunk_max_bytes}), so it was cached without "
            f"cache_if ever being consulted.",
            code="CACHE-IF-BYPASSED",
            fix="raise chunk_max_items / chunk_max_bytes above the size this "
            "result reaches, or return a list instead of an iterator, so "
            "the whole result arrives in one piece for the predicate to "
            "see.",
        )

    def stream_and_store(self, source: Iterator[Any], spec: CachedFunction, call: Call, run: BodyRun) -> Iterator[Any]:
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
        is real and accepted: a caller that takes two items leaves no entry.

        **Time is the producer's, not the wall clock.** Only the spans inside
        `next()` are summed, so a slow consumer cannot inflate the number the
        persistence decision reads.

        **Each stream writes its own chunks.** Two streams of one key overlap
        whenever two callers miss before either finishes (two threads, two
        requests, ``zip(gen(6), gen(6))``). Under shared chunk names the one
        that stops early would delete chunks the finished one stored, and two
        finished ones would store a result mixing chunks of both runs. So chunk
        names carry a stream id the manifest records (`chunk_prefix`): the
        manifest written last names a complete run of its own, and cleaning up
        touches only what this stream wrote.
        """
        tracker, observer = run.tracker, run.observer
        stream = secrets.token_hex(8)
        chunks = _ChunkStream(chunk_prefix(call.cache_key, stream), stream)
        committed = False
        returned: Any = None

        try:
            # Entered ONCE. Per item we only suspend around the `yield`, which
            # is a ContextVar swap; `__enter__` reinstalls patches and rechecks
            # the import hook, which costs microseconds per item.
            with tracker, observer:
                while True:
                    started = _perf_counter()
                    try:
                        item = next(source)
                    except StopIteration as stop:
                        chunks.produced_seconds += _perf_counter() - started
                        # What `yield from` evaluates to: the caller gets it
                        # now, and a replay hands it back from the manifest.
                        returned = stop.value
                        break
                    chunks.produced_seconds += _perf_counter() - started
                    self._buffer_item(spec, call, chunks, item)

                    # The caller's own reads and effects are its own.
                    tracker_token = tracker.suspend()
                    observer_token = observer.suspend()
                    try:
                        yield item
                    finally:
                        observer.resume(observer_token)
                        tracker.resume(tracker_token)

            self._finish_stream(spec, call, run, chunks, returned)
            committed = True
            return returned
        finally:
            if not committed:
                # Abandoned or failed: the chunks written so far are
                # unreferenced (no manifest names them). Best effort -- a
                # killed process can still leave some behind.
                for index in range(chunks.written):
                    try:
                        self._backend_slot.backend.delete(f"{chunks.prefix}:chunk_{index}")
                    except Exception:  # noqa: BLE001 - cleanup must not raise
                        logger.debug("[CORE] could not drop orphan chunk %d", index)

    def _buffer_item(self, spec: CachedFunction, call: Call, chunks: _ChunkStream, item: Any) -> None:
        """Add a streamed *item* to the open chunk, writing the chunk once it is full.

        The item as it is NOW, not a live reference pickled at the chunk's
        end: by then the caller may have edited it (`for row in rows():
        row.append(...)`) or the producer refilled it (a reused buffer), and
        every hit replayed that edit. The caller still gets the object itself.
        """
        snapshot = _snapshot(item)
        chunks.buffer.append(snapshot)
        chunks.buffer_bytes += estimate_object_size(snapshot)
        chunks.total_items += 1
        if len(chunks.buffer) >= spec.chunk_max_items or chunks.buffer_bytes >= spec.chunk_max_bytes:
            self._flush_chunk(spec, call, chunks)

    def _flush_chunk(self, spec: CachedFunction, call: Call, chunks: _ChunkStream) -> None:
        """Write the open chunk as the stream's next one and start a new one."""
        if chunks.written == 1 and spec.cache_if is not None:
            self._warn_cache_if_bypassed(spec)
        chunks.not_persisted = chunks.not_persisted or self._write_one_chunk(
            chunks.prefix,
            chunks.written,
            chunks.buffer,
            spec.name,
            ttl=call.ttl,
            execution_time=chunks.produced_seconds,
        )
        chunks.buffer = []
        chunks.buffer_bytes = 0
        chunks.written += 1

    def _finish_stream(
        self, spec: CachedFunction, call: Call, run: BodyRun, chunks: _ChunkStream, returned: Any
    ) -> None:
        """After the producer ran to exhaustion: the post-call checks, then the
        last chunk and the manifest that names the stream's chunks."""
        func_name = spec.name
        tracker, observer = run.tracker, run.observer
        self._purity.check_argument_mutation(func_name, call.args, call.kwargs, call.call_args_hash, observer)
        self._purity.report_observed_effects(func_name, observer)
        self._files.credit_remembered_reads(func_name, tracker, call.args, call.kwargs)
        auto_file_deps = snapshot_tracked_deps(tracker, spec.func.__module__)

        if chunks.written == 0:
            # Everything fit in one chunk, so cache_if can still see the
            # whole result -- it gates STORAGE, never what the caller
            # already received.
            refusal = self.refusal(
                None, func_name, chunks.buffer, run.rng_new, spec.cache_if, tracker, observer=observer
            )
            if refusal is not None:
                self._misses.note_not_stored(call.cache_key, refusal)
                return
            if chunks.buffer:
                chunks.not_persisted = self._write_one_chunk(
                    chunks.prefix, 0, chunks.buffer, func_name, ttl=call.ttl, execution_time=chunks.produced_seconds
                )
            # An empty iterator still gets a zero-chunk manifest, so a
            # hit returns empty instead of recomputing.
            n_chunks = 1 if chunks.buffer else 0
        else:
            if chunks.buffer:
                self._flush_chunk(spec, call, chunks)
            n_chunks = chunks.written
        self._store_chunked_manifest(
            StoreRequest(
                call,
                func_name,
                execution_time=chunks.produced_seconds,
                auto_file_deps=auto_file_deps,
                chunks_not_persisted=chunks.not_persisted,
                dynamic_sources=recorded_sources(tracker),
            ),
            {"n_chunks": n_chunks, "total_items": chunks.total_items, **_returned(returned)},
            chunks.stream,
        )

    def _store_sources(self, dynamic: DynamicSources) -> None:
        """Store each source too large for a caller's entry once, under its
        `source_key`, unless an earlier caller stored it already. Written
        before the entry: an entry is never there without what it names.
        One that goes later (evicted, or not written) makes its callers miss.
        No ttl of its own: entries with different ttls may name it."""
        backend = self._backend_slot.backend
        for digest, data in dynamic.blobs.items():
            key = source_key(digest)
            if backend.peek_metadata(key) is not None:
                continue
            serializer = PickleSerializer()
            metadata = CacheMetadata(
                key=key,
                timestamp=time.time(),
                serializer_cls=type(serializer),
                # Goes where the entries naming it go (`_write_one_chunk`).
                decorator_entry=True,
                copy_required=False,
            ).to_dict()
            backend.set(key, data, metadata, serializer=serializer)

    def _write_one_chunk(
        self,
        prefix: str,
        chunk_index: int,
        chunk_buffer: list[Any],
        func_name: str,
        ttl: int | None = None,
        execution_time: float = 0.0,
    ) -> str | None:
        """Write a single chunk to the backend; why it stayed in RAM, if it did.

        The chunk's metadata is minimal - the authoritative manifest
        lives at the canonical cache_key. We need *some* metadata for
        the serializer to round-trip correctly; the timestamp and the
        key are enough. We also propagate the manifest's ``ttl`` so
        ``Cash.cleanup()`` (without a ``max_age`` argument) can reclaim
        expired chunks alongside the expired manifest.

        *prefix* is the stream's `chunk_prefix`.
        """
        chunk_key = f"{prefix}:chunk_{chunk_index}"
        serializer = PickleSerializer()
        chunk_metadata = CacheMetadata(
            key=chunk_key,
            timestamp=time.time(),
            serializer_cls=type(serializer),
            execution_time=execution_time,
            ttl=ttl,
            # A chunk is not a result whose worth is decided on its own: it is
            # part of a decorated result, and goes where that result goes.
            # Judged by the compute floor instead, on the time produced so
            # far, the early chunks of a quick generator stayed in RAM while
            # its manifest reached disk, and every later process found the
            # entry incomplete and recomputed it.
            decorator_entry=True,
            # The replay hands out the chunk's items: a copy, as for any
            # decorated result (`ResultStore.store`).
            copy_required=not self._registry.is_frozen(func_name),
        ).to_dict()
        try:
            self._backend_slot.backend.set(chunk_key, chunk_buffer, chunk_metadata, serializer=serializer)
            # A tiered backend reports a tier's failure on the metadata
            # instead of raising (`ResultStore.store`).
            store_errors = chunk_metadata.get("store_errors")
            if store_errors and not [t for t in (chunk_metadata.get("storage") or []) if t != "RAM"]:
                raise CacheBackendError("; ".join(str(e) for e in store_errors))
        except Exception as e:  # noqa: BLE001 - as in `ResultStore.store`: the stream goes on
            backend_name = type(self._backend_slot.backend).__name__
            self._notices.warn_once(
                CashCacheStoreFailedWarning,
                chunk_key,
                "",
                # NOT "you will get a truncated iterator". ``CallRunner._chunks_are_intact``
                # probes every chunk and turns a manifest with a hole into a
                # MISS, on both read paths, so the cost is a permanent recompute
                # rather than a short answer.
                f"@cash.cache: backend {backend_name} failed to store "
                f"chunk {chunk_index} of {prefix} ({type(e).__name__}: {e}), "
                f"so the entry can never be read back and every later call "
                f"recomputes it.",
                code="STORE-CHUNK-FAILED",
                fix="clear the entry with f.cache_clear() before anything reads "
                "it, then fix the write the exception names.",
            )
            return None
        return not_persisted_reason(chunk_metadata)

    def _store_chunked_manifest(self, request: StoreRequest, manifest_data: dict[str, Any], stream: str) -> None:
        """Write the manifest entry for a chunked iterator at the call's key.

        The value stored at the key is the manifest dict (``n_chunks``,
        ``total_items``), stored as any decorated result is
        (`ResultStore.store`): the same persistence rules, and the same record
        of where it landed. Its metadata flags the entry as chunked so the
        hit path knows to use ``ChunkedCachedIterator``, and names the
        *stream* whose chunks it covers. The chunks of the manifest it
        replaces are deleted once it is written: nothing names them any more.
        """
        cache_key = request.call.cache_key
        replaced = self._replaced_stream(cache_key, stream)
        stored = self.store(
            dataclasses.replace(
                request,
                # The producer's time is all body: only the spans inside `next()`.
                body_seconds=request.execution_time,
                manifest={
                    "iterator_storage": "chunked",
                    "n_chunks": manifest_data["n_chunks"],
                    "chunk_stream": stream,
                },
            ),
            manifest_data,
        )
        if stored and replaced is not None:
            self._drop_chunks(*replaced)

    def _replaced_stream(self, cache_key: str, stream: str) -> tuple[str, int] | None:
        """The chunk prefix and count of the manifest at *cache_key* that a
        manifest for *stream* is about to replace, if there is one."""
        try:
            previous = self._backend_slot.backend.get_metadata(cache_key)
        except Exception:  # noqa: BLE001 - a lookup for cleanup must not fail a store
            return None
        if not previous or previous.get("iterator_storage") != "chunked":
            return None
        old = previous.get("chunk_stream")
        if not old or old == stream:
            return None
        return chunk_prefix(cache_key, old), int(previous.get("n_chunks") or 0)

    def _drop_chunks(self, prefix: str, n_chunks: int) -> None:
        """Delete the chunks a replaced manifest named. Best effort: a chunk
        left behind is unreferenced, and a reader still replaying it
        recomputes the rest (`ChunkedCachedIterator`)."""
        for index in range(n_chunks):
            try:
                self._backend_slot.backend.delete(f"{prefix}:chunk_{index}")
            except Exception:  # noqa: BLE001 - cleanup must not raise
                logger.debug("[CORE] could not drop replaced chunk %d of %s", index, prefix)


def _resolver(func: Any) -> Any:
    """``kind, name -> object``: what *func*'s global or closure variable holds now."""

    def resolve(kind: str, name: str) -> Any:
        if kind == "closure":
            cells = dict(zip(func.__code__.co_freevars, func.__closure__ or ()))
            try:
                return cells[name].cell_contents
            except ValueError as exc:  # an empty cell
                raise LookupError(name) from exc
        return func.__globals__[name]

    return resolve
