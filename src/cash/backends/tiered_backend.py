"""Tiered (multi-level) cache backend for Cash."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from cash import cost_model

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning
from ._base import CacheBackend, MetadataDict
from .adaptive_caps import human_bytes
from .serialization import PickleSerializer, Serializer
from .value_policy import WORTH_CEILING_BYTES_PER_SECOND, worth_its_bytes

_UNSEEN = object()

#: Promotion thresholds a `TieredBackend` uses unless it is given others.
DEFAULT_MIN_PERSIST_COMPUTE_S = 1.0
DEFAULT_MIN_PERSIST_SAVINGS_PCT = 0.20

logger = logging.getLogger(__name__)

__all__ = ["TieredBackend"]


class _TierWrites(NamedTuple):
    """What one pass over the persistent tiers did (`_write_persistent_tiers`)."""

    stored: list[str]  #: the tiers that took the entry
    size_refused: bool  #: a tier skipped the entry as too big for its cap
    refused_size: int  #: the size those caps were compared with
    refusing_caps: list[int]  #: the caps that refused it
    errors: list[str]  #: the tiers whose write raised, and what it raised


def _cap_list(caps: list[int] | None) -> str:
    """ " (its cap is 512.0 MiB)" / " (their caps are ...)" / "" when unknown."""
    if not caps:
        return ""

    rendered = ", ".join(human_bytes(c) for c in caps)
    return f" (cap: {rendered})" if len(caps) == 1 else f" (caps: {rendered})"


class TieredBackend(CacheBackend):
    """
    Backend that manages multiple cache tiers (e.g., Memory -> File -> S3).
    Implements smart promotion and read-repair.
    """

    # Map a backend implementation class to the cost-model backend kind used
    # to predict restore time. Anything unmapped is treated as disk (the
    # cost model itself also falls back to "disk" for unknown kinds).
    _BACKEND_KIND_BY_CLASS = {
        "InMemoryBackend": "ram",
        "FileBackend": "disk",
        "SQLiteBackend": "disk",
        "RedisBackend": "redis",
        "S3Backend": "s3",
    }

    def __init__(
        self,
        backends: list[CacheBackend],
        promotion_policy: Callable[[float, int], bool] | None = None,
        *,
        min_persist_compute_s: float = DEFAULT_MIN_PERSIST_COMPUTE_S,
        min_persist_savings_pct: float = DEFAULT_MIN_PERSIST_SAVINGS_PCT,
    ) -> None:
        """
        Args:
            backends: List of cache backends, ordered by speed (fastest first).
            promotion_policy: Callable taking (execution_time, size_bytes) and returning True if should promote.
            min_persist_compute_s: Compute floor for the serialization-aware
                decision — nothing below this is promoted past tier 0. The
                default (1.0 s) matches the fallback ``_default_promotion_policy``;
                the factory lowers it to 0.1 s for the smart-persistence stack.
            min_persist_savings_pct: Required fraction of compute time that a
                cache hit must save to be worth promoting. Mirrors
                ``CashConfig.min_cache_savings_pct`` (Gate A's threshold).
        """
        self.backends = backends
        # See `_drop_ram_if_cleared`.
        self._generation: Any = _UNSEEN
        self._generation_checked_at = 0.0
        self._stamp_writes_seen = 0
        self.promotion_policy = promotion_policy or self._default_promotion_policy
        self._min_persist_compute_s = min_persist_compute_s
        self._min_persist_savings_pct = min_persist_savings_pct
        # Once-per-session dedup for the oversize-refusal warning.
        self._warned_oversize = False
        #: Same, for the bytes-per-compute-second ceiling (`value_policy`).
        # Statements already told CACHE-NOT-WORTH-BYTES this session.
        self._warned_not_worth: set[str] = set()
        #: Refusals held for one warning per cell (`begin_cell_warnings`).
        self._not_worth_batch: list[tuple[str, int, float]] | None = None

    def get_metadata(self, key: str) -> dict | None:
        """Get only metadata for a cache key from the first backend that has it.

        Checks backends in order (fast → slow), returning the first hit.
        Supports metadata-only entries written by `set_metadata_only`.
        """
        for backend in self.backends:
            meta = backend.get_metadata(key)
            if meta is not None:
                return meta
        return None

    def set_metadata_only(self, key: str, metadata: dict) -> None:
        """Persist metadata without data payload to all backends that support it."""
        for backend in self.backends:
            if hasattr(backend, "set_metadata_only"):
                backend.set_metadata_only(key, metadata)

    def delete(self, key: str) -> None:
        for backend in self.backends:
            backend.delete(key)

    def clear(self) -> None:
        for backend in self.backends:
            backend.clear()

    def list_entries(self) -> list[dict[str, Any]]:
        seen_keys = set()
        entries = []
        for backend in self.backends:
            for entry in backend.list_entries():
                key = entry.get("key")
                if key not in seen_keys:
                    seen_keys.add(key)
                    entries.append(entry)
        return entries

    def entry_count(self) -> int:
        """The largest tier's count.

        The tiers overlap, and only `list_entries` can tell by how much: it
        reads every entry's key. The notebook writes a metadata record to the
        persistent tier for every statement it stores, RAM-only values
        included, so there the largest tier holds them all; a RAM-only entry
        from a decorated function is the one thing this can miss.
        """
        return max((b.entry_count() for b in self.backends), default=0)

    def tier_labels(self) -> list[str]:
        """Flatten child tier labels in configured order.

        Nested composite backends are expanded transitively, so a
        ``TieredBackend([TieredBackend([RAM, DISK]), S3])`` reports
        ``['RAM', 'DISK', 'S3']`` — one dot per leaf storage tier.
        """
        labels: list[str] = []
        for b in self.backends:
            labels.extend(b.tier_labels())
        return labels

    def shutdown(self) -> None:
        """Propagate shutdown to every child backend.

        This is what lets ``atexit`` drain pending async writes in
        Python scripts: ``Cash.shutdown()`` → ``TieredBackend.shutdown()``
        → each tier's own ``shutdown()`` → each tier's PendingWrites
        executor finishes its queue before the process exits. Errors
        from one tier do not block shutdown of the others — we still
        owe every backend its cleanup call.
        """
        for backend in self.backends:
            try:
                backend.shutdown()
            except Exception as e:  # noqa: BLE001 — best-effort cleanup
                logger.warning(
                    "Shutdown failed for backend %s: %s",
                    type(backend).__name__,
                    e,
                )

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        total = 0
        seen_keys = set()
        for backend in self.backends:
            for entry in list(backend.list_entries()):
                key = entry.get("key")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if is_expired(entry):
                    for b in self.backends:
                        b.delete(key)
                    total += 1
        return total

    def _promotion_backend_kind(self) -> str:
        """Cost-model backend kind of the first tier past RAM (the primary
        persistence target). Used to predict restore cost at ``set`` time."""
        if len(self.backends) > 1:
            name = type(self.backends[1]).__name__
            return self._BACKEND_KIND_BY_CLASS.get(name, "disk")
        return "disk"

    def _cost_model_promote(
        self,
        type_name: str,
        size_bytes: int,
        execution_time: float,
        backend_kind: str,
        *,
        floor: bool = True,
    ) -> bool:
        """Serialization-aware promotion decision (the same rule Gate A uses):
        promote only when recomputing costs more than the predicted restore.

        ``promote if execution_time - est_restore_time > min_savings * execution_time``

        The prediction comes from the fitted ``cost_model`` (serialize+write /
        read+deserialize end-to-end), so — unlike the old raw-bandwidth model —
        bigger objects are correctly *more* likely to persist when their
        recompute cost is high.
        """
        if floor and execution_time < self._min_persist_compute_s:
            return False

        est_restore = cost_model.estimated_restore_time(type_name, size_bytes, backend_kind)
        return execution_time - est_restore > self._min_persist_savings_pct * execution_time

    def _default_promotion_policy(self, execution_time: float, size_bytes: int) -> bool:
        """Fallback policy used when no ``promotion_policy`` is supplied and the
        entry's metadata carries no cost-model family (so ``set`` can't predict
        with the real type).

        Serialization-aware like the smart policy, but assumes the slowest
        (``_GENERIC``) family since the caller gave only ``size_bytes``. Keeps
        the 1.0 s compute floor as a designed floor for the fallback path.
        """
        return self._cost_model_promote("", size_bytes, execution_time, self._promotion_backend_kind())

    @staticmethod
    def _serialized_size(value: Any, serializer: Serializer | None) -> int | None:
        """Bytes this value takes once serialized, or None if it cannot be.

        The number a disk cap is actually about, and the one ``cash inspect``
        reports -- unlike the in-memory footprint the RAM tier measures, which
        is what the size gate had been comparing.

        Called only on the refusal path (see the caller), so the cost lands on
        values that were about to be thrown away. Returns None when the value
        cannot be serialized at all: the write would fail anyway, and the
        caller then falls back to the memory estimate rather than guessing.
        """
        try:
            ser = serializer if serializer is not None else PickleSerializer()
            return len(ser.serialize(value))
        except Exception:  # noqa: BLE001 - a sizing probe must never raise
            logger.debug("Could not measure the serialized size", exc_info=True)
            return None

    def _warn_oversize_not_persisted(
        self,
        key: str,
        size_bytes: int,
        caps: list[int] | None = None,
    ) -> None:
        """Warn once/session that a worth-persisting value fit no disk tier.

        The object is larger than every persistent tier's whole cap, so there
        is nowhere durable to put it. Cash offers it to the RAM tier rather
        than write-and-evict forever, and tells the user how to actually cache
        it. Deduped to once per session.

        *caps* is what the size was compared against, so the message can name
        the number the user set instead of alluding to it. A tester capped a
        cache at 500 MB, watched a 263 MB working set never get stored, and had
        no way to work out why from either the warning or ``cash inspect`` --
        the gate was comparing the value's in-memory footprint against a disk
        cap, while the SIZE column showed serialized bytes. Both numbers now
        appear here, and the one compared is the serialized one.

        "Offers", not "keeps": the RAM tier applies its own byte cap, which is
        machine-scaled and independent of ``max_cache_size``. That cap is
        normally SMALLER than the disk threshold that refused this value (4.0
        GiB against 18.7 GiB on one measured machine), so the usual outcome is
        eviction inside the same ``set()`` and no caching at all -- not
        RAM-only caching. Measured on a 4 MiB value that warns in both arms:
        RAM cap 100 MiB -> 2 calls, 1 execution; RAM cap 1 MiB -> 2 calls, 2
        executions."""
        if self._warned_oversize:
            return
        self._warned_oversize = True

        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-VALUE-TOO-BIG",
            # "offered to the RAM tier", not "kept in RAM": the RAM tier applies
            # its own byte cap (InMemoryBackend._evict_to_byte_cap), which is
            # machine-scaled and independent of max_cache_size, and is normally
            # SMALLER than the disk threshold that refused this value -- so the
            # usual outcome is eviction inside the same set(), i.e. no caching
            # at all rather than RAM-only caching. Measured on a 4 MiB value:
            # RAM cap 100 MiB -> 2 calls, 1 execution; RAM cap 1 MiB -> 2 calls,
            # 2 executions. Keep this and docs/warnings.md#cache-value-too-big
            # saying the same thing.
            f"cached value {key!r} is {human_bytes(size_bytes)} serialized, "
            f"which is more than every persistent cache tier's whole cap"
            f"{_cap_list(caps)}, so only the RAM tier was offered it -- it will "
            f"not survive a kernel restart, and if it is over the RAM tier's "
            f"own cap too it is evicted at once and nothing is cached.",
            f"raise max_cache_size above {human_bytes(size_bytes)} (a "
            f"comfortable multiple of it, so the cache can hold more than this "
            f"one entry), or cache something smaller -- the aggregate, the "
            f"sample, or the columns you actually use. The size named here is "
            f"the serialized one, the same number `cash inspect` reports.",
        )

    def _drop_persisted_call_refs(self, refs) -> None:
        """Take the persisted call results that a refused statement held.

        A cached call's result is stored once, in a ``call:`` entry, and the
        statement that produced it holds a ``CallRef``. Those entries are never
        judged on their own -- refusing one leaves the statement pointing at
        something that is not there -- so the decision belongs to the
        statement, and when the statement is refused for its bytes the call
        results go with it. Without this the refusal reclaims nothing: r26s5's
        seven 1.3 GB frames are ``call:`` entries, and its statements' own
        entries are a few KB each.

        Only the PERSISTENT tiers. The RAM copy is what makes the rest of this
        session fast and is not what the cache is spending disk on.

        Best effort, and a wrong guess is cheap: if another statement also
        refers to one of these, that statement becomes a miss and recomputes.
        It never restores something else.
        """
        for ref in refs or ():
            for backend in self.backends[1:]:
                try:
                    backend.delete(ref)
                except Exception:  # noqa: BLE001 - reclaiming disk never fails a write
                    logger.debug("Could not drop call ref %r", ref, exc_info=True)

    def _warn_not_worth_its_bytes(
        self,
        key: str,
        size_bytes: int,
        compute_seconds: float,
        code: str | None = None,
    ) -> None:
        """Warn once/session that a value cost more disk than it saves compute.

        Round 26's loudest unanimous finding was not that the cache was large
        but that nothing said so while they worked: "I checked free disk out of
        habit", "my project folder felt large", "nothing surfaces it while you
        work. Not the badge, not a warning, not `%cash_stats`". Two of five
        would have set `max_cache_size` on day one had anything told them.

        So the refusal says what it refused and what it would have cost, in the
        units the decision was made in. Deduped to once per session, like the
        oversize warning -- a sweep hits this on every iteration, and 72 copies
        of one message is the same silence by a different route.
        """
        # Once per STATEMENT, not per session: round 28's testers each saw one
        # of these per kernel and every later refusal was silent -- including
        # the ones on the steps they restarted into. Named by its code, which
        # is what a reader can find in their notebook; a key is not.
        ident = (code or "").strip() or str(key)
        if ident in self._warned_not_worth:
            return
        self._warned_not_worth.add(ident)
        lines = [ln for ln in ident.splitlines() if ln.strip()]
        # A loop body's stored code starts with its context marker comment.
        first = next((ln for ln in lines if not ln.lstrip().startswith("#")), lines[0] if lines else str(key))
        named = f"`{first[:80]}`" if code else repr(key)
        if self._not_worth_batch is not None:
            self._not_worth_batch.append((named, size_bytes, compute_seconds))
            return
        self._say_not_worth([(named, size_bytes, compute_seconds)])

    def begin_cell_warnings(self) -> None:
        """Hold CACHE-NOT-WORTH-BYTES refusals until `end_cell_warnings`, to
        say them once for the cell: a sweep cell said it 12 times, five lines
        each (round 29, r29s5)."""
        self._not_worth_batch = []

    def end_cell_warnings(self) -> None:
        batch, self._not_worth_batch = self._not_worth_batch, None
        if batch:
            self._say_not_worth(batch)

    #: Statements a combined refusal names; the rest are counted.
    _NOT_WORTH_NAMED = 5

    def _say_not_worth(self, refused: list[tuple[str, int, float]]) -> None:
        ceiling = human_bytes(WORTH_CEILING_BYTES_PER_SECOND)
        if len(refused) == 1:
            named, size_bytes, compute_seconds = refused[0]
            rate = size_bytes / max(compute_seconds, 1e-9) / (1024**2)
            what = (
                f"the value of {named} is {human_bytes(size_bytes)} serialized but "
                f"only takes {compute_seconds:.2f}s to recompute -- "
                f"{rate:,.0f} MiB of cache per second saved, against the "
                f"{ceiling} per second cash is willing to spend. It was not "
                f"persisted, so it is recomputed rather than restored."
            )
        else:
            shown = ", ".join(
                f"{named} ({human_bytes(size)} for {secs:.2f}s)"
                for named, size, secs in refused[: self._NOT_WORTH_NAMED]
            )
            more = len(refused) - self._NOT_WORTH_NAMED
            what = (
                f"{len(refused)} values in this cell take more cache per second "
                f"saved than the {ceiling} cash is willing to spend: {shown}"
                + (f" and {more} more" if more > 0 else "")
                + ". They were not persisted, so they are recomputed rather "
                "than restored."
            )
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-NOT-WORTH-BYTES",
            what,
            "nothing, if the recompute is cheap enough that you had not "
            "noticed it -- that is the trade being made. To cache it anyway, "
            "say so explicitly: `@cash:persist` on the statement, or "
            "`@cash.cache` on the function, both of which cash honours without "
            "re-taking the decision. Caching something smaller -- the "
            "aggregate, the sample, the columns you use -- is usually the "
            "better answer for a value this large. `cash inspect` in a "
            "terminal lists what the cache does hold, each entry's size "
            "next to the time it saves.",
            location=("<cash>", 1),
        )

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The fastest tier's metadata for *key*, without counting an access
        or promoting anything. See `BaseBackend.peek_metadata`."""
        for backend in self.backends:
            metadata = backend.peek_metadata(key)
            if metadata is not None:
                metadata = dict(metadata)
                metadata["source"] = getattr(type(backend), "source_label", None) or type(backend).__name__
                return metadata
        return None

    #: How often a process checks whether its disk cache was cleared under it.
    _GENERATION_CHECK_EVERY = 1.0

    def _drop_ram_if_cleared(self) -> None:
        """Forget RAM-held results when the disk cache was cleared under us.

        `cash clear` on a live service cleared the disk and nothing else: a
        worker kept serving the pre-clear answer from its RAM tier (round 18,
        5 stale answers in a row). The disk tier's generation token moves on a
        clear; this compares it at most once a second -- one stat, not one per
        hit -- and empties the faster tiers when it moved.
        """
        now = time.monotonic()
        if now - self._generation_checked_at < self._GENERATION_CHECK_EVERY:
            return
        self._generation_checked_at = now
        disk = next((b for b in self.backends if hasattr(b, "generation_token")), None)
        if disk is None:
            return
        try:
            token = disk.generation_token()
        except Exception:  # noqa: BLE001 - a check must never break a read
            return
        # From no stamp to one is a directory being created, not cleared --
        # unless THIS process wrote that stamp since the last look. A process
        # that started cold saw no stamp, created one on its first write, and
        # took the clear that followed -- the stamp gone again with `--all`, or
        # rewritten by `--function` -- for "still new": 5 of 5 kept serving
        # the pre-clear answer from RAM (round 19).
        known = self._generation
        writes = getattr(disk, "stamp_writes", 0)
        # This process writing the stamp AGAIN is itself the evidence: it
        # stamps a directory only when it finds none, so a second stamp means
        # the first was taken away in between -- by `cash clear --all`, while a
        # long call ran that began inside the one-second window after the
        # first write and so was never checked (round 20: 7 of 10).
        restamped = writes > self._stamp_writes_seen and (
            self._stamp_writes_seen >= 1 or writes - self._stamp_writes_seen >= 2
        )
        if known in (_UNSEEN, None) and writes != self._stamp_writes_seen:
            known = getattr(disk, "written_stamp", None)
        self._stamp_writes_seen = writes
        if restamped or (known not in (_UNSEEN, None) and token != known):
            for faster in self.backends[: self.backends.index(disk)]:
                try:
                    faster.clear()
                except Exception:  # noqa: BLE001
                    logger.debug("could not drop %s after a clear", type(faster).__name__)
        self._generation = token

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        self._drop_ram_if_cleared()
        for i, backend in enumerate(self.backends):
            metadata, value = backend.get(key)
            # Key-presence test: metadata is None when the child backend
            # reports "key absent" (per its API contract). A non-None
            # metadata dict with a None value means the user genuinely
            # cached None — still a hit.
            if metadata is not None:
                # Read-Repair / Promotion to faster tiers
                # If found in Tier 2 (File), promote to Tier 1 (Memory)
                for j in range(i):
                    # Always offered: a faster tier should hold what is being
                    # read. One it can never hold -- over the RAM tier's
                    # eviction target -- that tier refuses itself
                    # (`InMemoryBackend.set`), which is what keeps a restore of
                    # a big entry from emptying it.
                    try:
                        self.backends[j].set(key, value, metadata)
                    except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                        logger.warning(
                            "Failed to promote key '%s' to tier %d (%s): %s",
                            key,
                            j,
                            type(self.backends[j]).__name__,
                            e,
                        )

                # Inject source information
                metadata["source"] = getattr(type(backend), "source_label", None) or type(backend).__name__

                return metadata, value
        return None, None

    def _write_persistent_tiers(
        self,
        key: str,
        value: Any,
        metadata: MetadataDict,
        serializer: Serializer | None,
        cap_size: int,
    ) -> _TierWrites:
        """Write to every tier past RAM that takes an entry this size."""
        stored_destinations: list[str] = []
        errors: list[str] = []
        size_refused = False  # a tier skipped this object because it's too big
        refusing_caps: list[int] = []  # the caps it was measured against
        refused_size = cap_size  # the size that was actually compared
        for i in range(1, len(self.backends)):
            backend = self.backends[i]
            cap = backend._promotion_size_cap()
            # Guard against non-numeric caps (e.g. a MagicMock tier in
            # tests) — treat anything that isn't a real number as no cap.
            if isinstance(cap, bool) or not isinstance(cap, (int, float)):
                cap = None
            if cap is not None and cap_size and cap_size > cap:
                # About to refuse. `cap_size` is the value's IN-MEMORY
                # footprint (the RAM tier measures it on the way past), and
                # what this tier stores is the SERIALIZED form -- for a
                # frame of strings, two or more times smaller. Refusing on
                # the memory number cost a tester their whole cache: a
                # 160 MB entry, under their 500 MB cap by any measure they
                # could see, was never stored because it took more than
                # that in RAM, and `cash inspect` showed them the 160.
                #
                # So measure properly before refusing. Serializing is
                # expensive, which is why it happens HERE and not on every
                # write: this branch is reached only when the value was
                # about to be dropped, and the alternative to the cost is a
                # wrong answer to "will this fit".
                true_size = self._serialized_size(value, serializer)
                if true_size is not None and true_size <= cap:
                    cap_size = true_size
                else:
                    logger.debug(
                        "[TIERED] Skipping %s for key %r: size %d > cap %d",
                        type(backend).__name__,
                        key,
                        true_size or cap_size,
                        cap,
                    )
                    size_refused = True
                    refused_size = true_size or cap_size
                    refusing_caps.append(int(cap))
                    continue
            try:
                backend.set(key, value, metadata, serializer)
                _label = getattr(type(backend), "source_label", None) or type(backend).__name__
                stored_destinations.append(_label)
            except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                logger.warning("[TIERED] Failed to write to backend %s: %s", type(backend).__name__, e)
                # And on the entry's metadata, which is how the caller hears
                # about it: a failure here used to be this log line alone, so
                # an unpicklable result reported STORE-FAILED on a FileBackend
                # and nothing at all on the default tiered one -- no warning,
                # and an empty cache_info()['warnings'] (found attacking the
                # decorator before round 26).
                errors.append(f"{type(backend).__name__}: {type(e).__name__}: {e}")
        return _TierWrites(stored_destinations, size_refused, refused_size, refusing_caps, errors)

    def persist_from_memory(self, key: str, rebuild_seconds: float) -> bool:
        """Write an entry only the RAM tier holds to the persistent tiers, when
        restoring it would beat rebuilding it.

        ``set`` decides by the statement's own compute time, and a statement is
        often cheap only because its inputs are there: ``latest =
        sales['week'].max()`` takes 0.06 s, but after a restart ``sales`` is
        gone too, and so is everything back to the folder it was read from --
        35 s (round 23, r23s2). *rebuild_seconds* is that whole cost, and the
        same cost-model rule decides with it. Only a notebook value (it carries
        a cost-model family) is considered. Returns True when it was written.
        """
        if len(self.backends) < 2:
            return False
        peek = getattr(self.backends[0], "peek_entry", None)
        entry = peek(key) if peek is not None else None
        if entry is None:
            return False
        stored_metadata, value = entry
        if any(d != "RAM" for d in stored_metadata.get("storage") or ()):
            return False  # on disk already
        if stored_metadata.get("metadata_only") or stored_metadata.get("cost_model_family") is None:
            return False
        size = stored_metadata.get("cost_model_size_bytes", stored_metadata.get("size", 0))
        if not self._cost_model_promote(
            stored_metadata.get("cost_model_type_name", ""), size, rebuild_seconds, self._promotion_backend_kind()
        ):
            return False
        # The bytes-per-compute-second ceiling applies here too, and this is
        # where it matters most: a notebook statement sets `defer_persist`, so
        # it never reaches the promotion block in `set` -- this end-of-cell
        # pass is how notebook values get to disk, and notebook values are what
        # filled round 26's caches. Weighed like `set` does it, with the call
        # results the entry refers to, and against *rebuild_seconds*: what a
        # restore actually saves here is the whole upstream chain, not the one
        # statement's own time.
        if not stored_metadata.get("force_persist"):
            weight = (stored_metadata.get("size") or size) + int(stored_metadata.get("call_ref_bytes") or 0)
            if not worth_its_bytes(weight, rebuild_seconds):
                self._warn_not_worth_its_bytes(key, weight, rebuild_seconds, code=stored_metadata.get("code"))
                self._drop_persisted_call_refs(stored_metadata.get("call_refs"))
                stored_metadata["persist_skipped"] = "bytes"
                return False
        metadata = {
            k: v
            for k, v in stored_metadata.items()
            if k not in ("persist_skipped", "source", "storage", "defer_persist")
        }
        metadata["rebuild_time"] = rebuild_seconds
        writes = self._write_persistent_tiers(key, value, metadata, None, stored_metadata.get("size") or size)
        if not writes.stored:
            if writes.size_refused:
                self._warn_oversize_not_persisted(key, writes.refused_size, writes.refusing_caps)
            return False
        stored_metadata["storage"] = ["RAM", *writes.stored]
        stored_metadata.pop("persist_skipped", None)
        return True

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> None:
        if not self.backends:
            return

        # Keep a reference to the original dict so we can propagate storage info back
        original_metadata = metadata
        #: What a tier refused to write this call, for the caller to report.
        store_errors: list[str] = []
        metadata = dict(metadata) if metadata is not None else {}
        stored_destinations = []
        # A tier's `default_ttl` belongs to the entry, not to that tier: stamped
        # once, here, every tier's copy expires together. Left to the file tier
        # alone, a process kept serving the result from RAM long after the
        # disk copy had expired.
        if metadata.get("ttl") is None:
            for tier in self.backends:
                default = getattr(tier, "_default_ttl", None)
                if default is not None:
                    metadata["ttl"] = default
                    break

        # Always write to Tier 0 (Memory). It may refuse a value its cap could
        # never hold (`InMemoryBackend.set` returns False); then it is not a
        # destination, and the badge and the miss explanation must not say so.
        try:
            if self.backends[0].set(key, value, metadata, serializer) is not False:
                stored_destinations.append("RAM")
        except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
            logger.warning(
                "Failed to write key '%s' to tier 0 (%s): %s",
                key,
                type(self.backends[0]).__name__,
                e,
            )
            # And on the entry's metadata, so the caller hears it: the RAM tier
            # refuses a value it cannot copy, and nothing else would say why the
            # call recomputes every time.
            store_errors.append(f"{type(self.backends[0]).__name__}: {type(e).__name__}: {e}")

        # Check promotion for subsequent tiers. The promotion decision is
        # made per-tier so a single set() can land in some tiers and skip
        # others — e.g. a 20 MB DataFrame goes to RAM + DISK but skips
        # Redis (10 MB cap).
        if len(self.backends) > 1:
            exec_time = metadata.get("execution_time", 0)
            size = metadata.get("size", 0)

            # Check if force_persist is set via @cash:persist annotation
            force_persist = metadata.get("force_persist", False)

            # Promotion decision — same rule as the statement processor's Gate A
            # (predicted restore vs compute), applied here so the two gates can
            # never contradict each other. When the entry carries a cost-model
            # family (notebook-cached values), predict restore time with the
            # real type; otherwise fall through to the 2-arg promotion_policy
            # (injected test lambdas, the decorator path, legacy metadata).
            family = metadata.get("cost_model_family")
            deferred = bool(metadata.pop("defer_persist", False)) and not force_persist
            if original_metadata is not None:
                original_metadata.pop("defer_persist", None)
            decorated = bool(metadata.get("decorator_entry"))
            if force_persist:
                past_compute_floor = True
            elif deferred:
                # A version the same cell replaces: the end-of-cell pass
                # persists the final one (``persist_from_memory``).
                past_compute_floor = False
            elif decorated:
                # A decorated result is persisted, full stop. No compute floor
                # and no cost model: `@cash.cache` is the caller having already
                # decided, and cash's job is to honour that rather than re-take
                # the decision per call.
                #
                # Both gates were wrong here in their own way. The floor (0.1 s)
                # meant a script run twice recomputed everything, which is how
                # two of six agents attacking the decorator before round 26
                # reported "no bugs found" -- nothing had ever reached disk. The
                # cost model then inherited the whole decision, and it rests on
                # a fitted intercept measured at 10.4 ms against a real small
                # read of ~1.3 ms, so nothing under about 13 ms of body was
                # stored however often it was called.
                #
                # What still applies is the per-tier size caps below: a value
                # too large for any disk tier has nowhere to go, and says so.
                past_compute_floor = True
            elif family is not None:
                past_compute_floor = self._cost_model_promote(
                    metadata.get("cost_model_type_name", ""),
                    metadata.get("cost_model_size_bytes", size),
                    exec_time,
                    self._promotion_backend_kind(),
                )
            else:
                past_compute_floor = self.promotion_policy(exec_time, size)

            # Size used for per-tier caps — prefer the cost-model estimate
            # (the notebook path sets no plain 'size' key).
            cap_size = size or metadata.get("cost_model_size_bytes", 0)

            # ...and worth the bytes it would occupy. Every gate above asks
            # whether restoring beats recomputing; none of them asks what the
            # answer COSTS. Round 26's five caches held 58 GiB for 61-360 MB of
            # input data, and the three mechanisms behind that (see
            # `value_policy`) are each a population version pruning cannot
            # ration: r26s5's 1.3 GB frames at 5.0 s of compute, r26s4's 48 MiB
            # loop iterations at 0.00 s, r26s3's spare copies. One rate, applied
            # here, covers all three.
            #
            # `force_persist` and `decorator_entry` are exempt for the same
            # reason they are exempt from the compute floor: the caller has
            # already decided, and re-taking that decision per call is what
            # `@cash.cache` exists to stop.
            # A statement entry is weighed with the call results it REFERS to,
            # not just its own bytes. A cached call's result is stored once, in
            # a `call:` entry, and the statement that produced it holds a
            # `CallRef` -- so the statement's own entry is a few KB while the
            # thing it restores is hundreds of MB. r26s5's seven 1.3 GB frames
            # are `call:` entries, and the 72 statements referencing them
            # declare 14,293 MiB of `call_ref_bytes` between them.
            #
            # A `call:` entry is therefore never judged on its own: refusing it
            # leaves the statement that points at it restoring a reference to
            # something that is not there, so the statement "hits" and then
            # rebuilds anyway -- a cache entry that costs disk and saves
            # nothing. `test_superseded_versions_are_pruned` caught exactly
            # that. The decision belongs to the statement, which is the thing
            # whose compute is actually being saved.
            weight = cap_size + int(metadata.get("call_ref_bytes") or 0)
            is_call_entry = str(key).startswith("call:")
            # ...except one not digested (`call_refs.ESTIMATED_FIELD`): only
            # the statement it is the plain result of refers to it, so no
            # other statement's refusal would drop it. Weighed by its
            # estimated PICKLED size, which is what disk holds: r28s5's result
            # is 402 MiB pickled and 1.7 GiB in memory, as 3.7 million strings.
            if is_call_entry and metadata.get("value_bytes_estimated"):
                weight = int(metadata.get("value_bytes") or cap_size)
                is_call_entry = False
            bytes_refused = False
            if past_compute_floor and not (force_persist or decorated) and not is_call_entry:
                if not worth_its_bytes(weight, exec_time):
                    past_compute_floor = False
                    bytes_refused = True
                    # A `call:` entry is said by the statement holding its
                    # result, which names the code the user wrote; said here
                    # too, every refusal was printed twice, the second naming
                    # an internal key (round 29, r29s1 and r29s3).
                    if not str(key).startswith("call:"):
                        self._warn_not_worth_its_bytes(key, weight, exec_time, code=metadata.get("code"))
                    self._drop_persisted_call_refs(metadata.get("call_refs"))

            writes = (
                self._write_persistent_tiers(key, value, metadata, serializer, cap_size)
                if past_compute_floor
                else _TierWrites([], False, cap_size, [], [])
            )
            stored_destinations.extend(writes.stored)
            store_errors.extend(writes.errors)
            size_refused = writes.size_refused

            # The value was worth persisting (cleared the compute floor) but
            # every persistent tier refused it as too big for its cap — it will
            # live in RAM only and vanish on the next kernel restart. A clean
            # no-op beats a treadmill, but the user should know why nothing
            # durable was written and how to fix it.
            if size_refused and not any(d != "RAM" for d in stored_destinations):
                self._warn_oversize_not_persisted(key, writes.refused_size, writes.refusing_caps)

        # Update metadata with storage info so UI can see it immediately
        if metadata is not None:
            metadata["storage"] = stored_destinations

        # Propagate storage info back to the caller's original metadata dict
        if original_metadata is not None:
            original_metadata["storage"] = stored_destinations
            if store_errors:
                original_metadata["store_errors"] = store_errors
            # And why it went no further, so "why did the next process miss?"
            # has an answer: the compute floor / cost model, or a size cap.
            if len(self.backends) > 1 and not any(d != "RAM" for d in stored_destinations):
                if bytes_refused:
                    original_metadata["persist_skipped"] = "bytes"
                elif size_refused:
                    original_metadata["persist_skipped"] = "size"
                elif deferred:
                    original_metadata["persist_skipped"] = "replaced_in_cell"
                elif not past_compute_floor:
                    original_metadata["persist_skipped"] = "compute"

        # Log visibility
        if stored_destinations:
            logger.debug("[STORAGE] Stored in: %s", ", ".join(stored_destinations))
