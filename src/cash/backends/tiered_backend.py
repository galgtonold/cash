"""Tiered (multi-level) cache backend for Cash."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from ._base import CacheBackend, MetadataDict
from .clear_watch import ClearWatcher
from .persistence_policy import PersistencePolicy
from .serialization import PickleSerializer, Serializer
from .store_notices import StoreNotices

logger = logging.getLogger(__name__)

__all__ = ["TieredBackend"]


class _TierWrites(NamedTuple):
    """What one pass over the persistent tiers did (`_write_persistent_tiers`)."""

    stored: list[str]  #: the tiers that took the entry
    size_refused: bool  #: a tier skipped the entry as too big for its cap
    refused_size: int  #: the size those caps were compared with
    refusing_caps: list[int]  #: the caps that refused it
    errors: list[str]  #: the tiers whose write raised, and what it raised


class TieredBackend(CacheBackend):
    """Tiers ordered fastest first (e.g. RAM -> disk -> S3).

    Every value goes to the first tier; `policy` decides which also go past
    it. A read from a slower tier is copied into the faster ones.
    """

    def __init__(
        self,
        backends: list[CacheBackend],
        promotion_policy: Callable[[float, int], bool] | None = None,
        *,
        policy: PersistencePolicy | None = None,
    ) -> None:
        """
        Args:
            backends: The tiers, fastest first.
            promotion_policy: ``(execution_time, size_bytes) -> bool``, to
                decide instead of the cost model for an entry that carries no
                cost-model family. Explicit requests to keep an entry and the
                bytes-per-second ceiling still apply.
            policy: The persistence rule; `PersistencePolicy` defaults if omitted.
        """
        self.backends = backends
        self.promotion_policy = promotion_policy
        self.policy = policy if policy is not None else PersistencePolicy()
        self.notices = StoreNotices()
        self._clear_watch = ClearWatcher()

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
        """Persist metadata without data payload to every tier that keeps it."""
        for backend in self.backends:
            backend.set_metadata_only(key, metadata)

    @property
    def default_ttl(self) -> float | None:
        """The first tier's ``default_ttl`` that is set: it belongs to the
        entry, so every tier's copy expires together (see ``set``)."""
        return next((t for t in (b.default_ttl for b in self.backends) if t is not None), None)

    @property
    def local_dir(self) -> str | None:
        """The directory of the first tier that keeps entries on local disk."""
        return next((d for d in (b.local_dir for b in self.backends) if d is not None), None)

    def generation_token(self) -> tuple | None:
        disk = self._disk_tier()
        return disk.generation_token() if disk is not None else None

    def _disk_tier(self) -> CacheBackend | None:
        """The tier whose directory can be cleared under this process."""
        return next((b for b in self.backends if b.local_dir is not None), None)

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
        """The cost-model kind of the first tier past RAM, which a persisted
        value is restored from."""
        return self.backends[1].cost_kind if len(self.backends) > 1 else "disk"

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

    def _drop_persisted_call_refs(self, refs) -> None:
        """Take the persisted call results that a refused statement held.

        A cached call's result is stored once, in a ``call:`` entry, and the
        statement that produced it holds a ``CallRef``. Those entries are never
        judged on their own -- refusing one leaves the statement pointing at
        something that is not there -- so the decision belongs to the
        statement, and when the statement is refused for its bytes the call
        results go with it. Without this the refusal reclaims nothing: the
        large values are in the call entries, and the statements' own entries
        are a few KB each.

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

    def hold_notices(self) -> None:
        self.notices.hold()

    def release_notices(self) -> None:
        self.notices.release()

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """The fastest tier's metadata for *key*, without counting an access
        or promoting anything. See `BaseBackend.peek_metadata`."""
        for backend in self.backends:
            metadata = backend.peek_metadata(key)
            if metadata is not None:
                metadata = dict(metadata)
                metadata["source"] = backend.source_label
                return metadata
        return None

    def _drop_ram_if_cleared(self) -> None:
        """Empty the tiers in front of the disk tier when it was cleared from
        outside this process (`ClearWatcher`)."""
        disk = self._disk_tier()
        if disk is None or not self._clear_watch.cleared(disk):
            return
        for faster in self.backends[: self.backends.index(disk)]:
            try:
                faster.clear()
            except Exception:  # noqa: BLE001 - a read must not fail over a stale RAM tier
                logger.debug("could not drop %s after a clear", type(faster).__name__, exc_info=True)

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
                metadata["source"] = backend.source_label

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
            cap = backend.promotion_size_cap()
            # Guard against non-numeric caps (e.g. a MagicMock tier in
            # tests) — treat anything that isn't a real number as no cap.
            if isinstance(cap, bool) or not isinstance(cap, (int, float)):
                cap = None
            if cap is not None and cap_size and cap_size > cap:
                # About to refuse. `cap_size` is the value's IN-MEMORY
                # footprint (the RAM tier measures it on the way past), and
                # what this tier stores is the SERIALIZED form -- for a
                # frame of strings, two or more times smaller, and the
                # serialized size is the one `cash inspect` shows. So
                # measure properly before refusing. Serializing is
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
                stored_destinations.append(backend.source_label)
            except Exception as e:  # noqa: BLE001 (intentional: backend errors must not propagate)
                logger.warning("[TIERED] Failed to write to backend %s: %s", type(backend).__name__, e)
                # And on the entry's metadata, which is how the caller hears
                # about it (STORE-FAILED, cache_info()['warnings']), as it
                # would from a bare FileBackend.
                errors.append(f"{type(backend).__name__}: {type(e).__name__}: {e}")
        return _TierWrites(stored_destinations, size_refused, refused_size, refusing_caps, errors)

    def persist_from_memory(self, key: str, rebuild_seconds: float) -> bool:
        """Write an entry only the RAM tier holds to the persistent tiers, when
        restoring it would beat rebuilding it.

        ``set`` decides by the statement's own compute time, and a statement is
        often cheap only because its inputs are there: ``latest =
        sales['week'].max()`` takes 0.06 s, but after a restart ``sales`` is
        gone too, and so is everything back to the folder it was read from.
        *rebuild_seconds* is that whole cost, and the
        same cost-model rule decides with it. Only a notebook value (it carries
        a cost-model family) is considered. Returns True when it was written.
        """
        if len(self.backends) < 2:
            return False
        entry = self.backends[0].peek_entry(key)
        if entry is None:
            return False
        stored_metadata, value = entry
        if any(d != "RAM" for d in stored_metadata.get("storage") or ()):
            return False  # on disk already
        decision = self.policy.decide_rebuild(
            stored_metadata, rebuild_seconds, backend_kind=self._promotion_backend_kind()
        )
        if decision.skipped == "bytes":
            self.notices.not_worth_bytes(key, decision.weight, rebuild_seconds, code=stored_metadata.get("code"))
            self._drop_persisted_call_refs(stored_metadata.get("call_refs"))
            stored_metadata["persist_skipped"] = "bytes"
        if not decision.persist:
            return False
        size = stored_metadata.get("cost_model_size_bytes", stored_metadata.get("size", 0))
        metadata = {
            k: v
            for k, v in stored_metadata.items()
            if k not in ("persist_skipped", "source", "storage", "defer_persist")
        }
        metadata["rebuild_time"] = rebuild_seconds
        writes = self._write_persistent_tiers(key, value, metadata, None, stored_metadata.get("size") or size)
        if not writes.stored:
            if writes.size_refused:
                self.notices.too_big(key, writes.refused_size, writes.refusing_caps)
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
        if metadata.get("ttl") is None and self.default_ttl is not None:
            metadata["ttl"] = self.default_ttl

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

        decision = None
        if len(self.backends) > 1:
            deferred = bool(metadata.pop("defer_persist", False))
            if original_metadata is not None:
                original_metadata.pop("defer_persist", None)
            decision = self.policy.decide(
                key,
                metadata,
                backend_kind=self._promotion_backend_kind(),
                deferred=deferred,
                override=self.promotion_policy,
            )
            size = metadata.get("size", 0) or 0
            cap_size = size or metadata.get("cost_model_size_bytes", 0)
            if decision.skipped == "bytes":
                exec_time = metadata.get("execution_time", 0) or 0
                if decision.report:
                    self.notices.not_worth_bytes(key, decision.weight, exec_time, code=metadata.get("code"))
                self._drop_persisted_call_refs(metadata.get("call_refs"))
            writes = (
                self._write_persistent_tiers(key, value, metadata, serializer, cap_size)
                if decision.persist
                else _TierWrites([], False, cap_size, [], [])
            )
            stored_destinations.extend(writes.stored)
            store_errors.extend(writes.errors)
            size_refused = writes.size_refused
            # Worth persisting, but too big for every persistent tier's cap: it
            # lives in RAM only, and the user should know why and what to do.
            if size_refused and not any(d != "RAM" for d in stored_destinations):
                self.notices.too_big(key, writes.refused_size, writes.refusing_caps)

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
            if decision is not None and not any(d != "RAM" for d in stored_destinations):
                if decision.skipped == "bytes":
                    original_metadata["persist_skipped"] = "bytes"
                elif size_refused:
                    original_metadata["persist_skipped"] = "size"
                elif decision.skipped is not None:
                    original_metadata["persist_skipped"] = decision.skipped

        # Log visibility
        if stored_destinations:
            logger.debug("[STORAGE] Stored in: %s", ", ".join(stored_destinations))
