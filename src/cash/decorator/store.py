"""Storing a computed result: whether to, whole or in chunks, and the
lineage tag it carries."""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._clock import perf_counter as _perf_counter
from .._memo import RESULT_TYPES, LruMemo
from ..backends.serialization import get_serializer
from ..effect_observer import EffectObserver
from ..exceptions import CacheBackendError, CashCacheIneffectiveWarning, CashCacheStoreFailedWarning
from ..object_hashing import estimate_object_size
from ..value_types import IMMUTABLE_PRIMS
from .arg_hashing import LINEAGE_SRC_DECORATOR, LINEAGE_SRC_FROZEN
from .cache_metadata import CacheMetadata
from .cached_function import CachedFunction
from .call_state import NO_WATCH
from .explain import not_persisted_reason
from .file_deps import snapshot_tracked_deps

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

    def attach_lineage(
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
        try:
            # Say who wrote it: nothing will move this tag when the value is
            # mutated, so `ArgHasher.hash_payload` must not take it for the content
            # -- unless the function was declared frozen=True.
            try:
                result._cash_lineage_src = LINEAGE_SRC_FROZEN if frozen else LINEAGE_SRC_DECORATOR
                if func_name is not None:
                    # Named in CACHE-NET-LOSS and KEY-FROZEN-MUTATED.
                    result._cash_lineage_producer = func_name
            except (AttributeError, TypeError):
                if frozen:
                    self._frozen.warn_has_no_effect(func_name, result)
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
                # user reading CASH_DEBUG.
                UNTAGGABLE_TYPES[type(result)] = True
                logger.debug(
                    "results of type %s cannot carry a lineage tag, so a cached function taking one hashes its content",
                    type_name,
                )

        except (AttributeError, TypeError):
            logger.debug("Failed to attach lineage hash to %s result", type(result).__name__)

    def store(
        self,
        cache_key: str,
        func_name: str,
        result: Any,
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
            # the 0.1s floor (Windows CI).
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
                saves_seconds=saves_seconds,
                serializer_cls=type(serializer),
                ttl=ttl,
                ttl_declared=ttl_declared,
                args_hash=args_hash,
                state_hash=state_hash,
                # Each entry: path -> {'mtime': float, 'size': int}.
                # Validated on subsequent get() via FileDeps.auto_file_deps_fresh.
                auto_file_deps=auto_file_deps or None,
                rng_replay=rng_replay or None,
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
            )

            # Kept, not a temporary: TieredBackend writes back where the value
            # landed, and "RAM only" is the answer to the next process's miss.
            meta_dict = meta.to_dict()
            self._backend_slot.backend.set(cache_key, result, meta_dict, serializer=serializer)
            # A tiered backend catches each tier's failure so one bad tier
            # cannot break a call; it reports them here instead, and a result
            # nothing could store is a STORE-FAILED like any other.
            store_errors = meta_dict.get("store_errors")
            if store_errors and not [t for t in (meta_dict.get("storage") or []) if t != "RAM"]:
                raise CacheBackendError("; ".join(str(e) for e in store_errors))
            not_persisted = not_persisted_reason(meta_dict)
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
        except (OSError, TypeError, pickle.PicklingError, RuntimeError, CacheBackendError) as e:
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

    def stream_and_store(
        self,
        source,
        *,
        cache_key,
        spec,
        tracker,
        observer,
        rng_new,
        args,
        kwargs,
        args_hash,
        current_state_hash,
        ttl,
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
        is real and accepted: a caller that takes two items leaves no entry.

        **Time is the producer's, not the wall clock.** Only the spans inside
        `next()` are summed, so a slow consumer cannot inflate the number the
        persistence decision reads.
        """
        func_name, cache_if = spec.name, spec.cache_if
        chunk_max_items, chunk_max_bytes = spec.chunk_max_items, spec.chunk_max_bytes

        buffer: list[Any] = []
        buffer_bytes = 0
        chunk_index = 0
        total_items = 0
        produced_seconds = 0.0
        committed = False

        try:
            # Entered ONCE. Per item we only suspend around the `yield`, which
            # is a ContextVar swap; `__enter__` reinstalls patches and rechecks
            # the import hook, which costs microseconds per item.
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
                            self._warn_cache_if_bypassed(spec)
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

            self._purity.check_argument_mutation(func_name, args, kwargs, args_hash, observer)
            self._purity.report_observed_effects(func_name, observer)
            self._files.credit_remembered_reads(func_name, tracker, args, kwargs)
            auto_file_deps = snapshot_tracked_deps(tracker, spec.func.__module__)

            if chunk_index == 0:
                # Everything fit in one chunk, so cache_if can still see the
                # whole result -- it gates STORAGE, never what the caller
                # already received.
                refusal = self.refusal(None, func_name, buffer, rng_new, cache_if, tracker, observer=observer)
                if refusal is not None:
                    self._misses.note_not_stored(cache_key, refusal)
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
                        self._warn_cache_if_bypassed(spec)
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
                        self._backend_slot.backend.delete(f"{cache_key}:chunk_{index}")
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
            self._backend_slot.backend.set(chunk_key, chunk_buffer, chunk_metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self._backend_slot.backend).__name__
            self._notices.warn_once(
                CashCacheStoreFailedWarning,
                f"{cache_key}:chunk_{chunk_index}",
                "",
                # NOT "you will get a truncated iterator". ``CallRunner._chunks_are_intact``
                # probes every chunk and turns a manifest with a hole into a
                # MISS, on both read paths, so the cost is a permanent recompute
                # rather than a short answer.
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
            self._backend_slot.backend.set(cache_key, manifest_data, metadata, serializer=serializer)
        except (OSError, TypeError, pickle.PicklingError, RuntimeError) as e:
            backend_name = type(self._backend_slot.backend).__name__
            self._notices.warn_once(
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
