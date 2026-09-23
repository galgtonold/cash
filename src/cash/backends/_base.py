"""Abstract base class and shared types for cache backends."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CacheMetadata", "MetadataDict", "CacheBackend", "ttl_expired"]

#: Cost assumed for an entry whose execution time is unknown -- written without
#: one (raw backend use), or ranked with nothing recorded about it. Small, so an
#: entry of known cost outranks it.
UNKNOWN_COST_S = 0.001

#: Value-per-byte steps per doubling (see `gdsf_value`): 1/16 octave, ~4.4%.
_GDSF_STEPS_PER_OCTAVE = 16


def gdsf_value(metadata: MetadataDict | dict, size: int) -> float:
    """The value term of a GreedyDual-Size-Frequency priority, both tiers' ranking.

    ``hits * execution_time / size``, rounded down to a 1/16-octave step.
    Unrounded, a one-byte size difference decided between entries of equal
    cost -- and the newest entry is often the one a byte bigger (a longer
    key), so it went first: recency inverted among equals, and the
    evict-after-write warning fired on a healthy cache. Real values span
    seven orders of magnitude, so a 4% step costs no ranking that matters,
    and ties fall through to recency as they should.
    """
    cost = metadata.get("execution_time") or 0.0
    if cost <= 0:
        cost = UNKNOWN_COST_S
    hits = metadata.get("access_count", 0) + 1
    value = hits * cost / max(1, size)
    return 2.0 ** (math.floor(math.log2(value) * _GDSF_STEPS_PER_OCTAVE) / _GDSF_STEPS_PER_OCTAVE)


# Serializes the one-time creation of a backend's per-key lock registry
# (``CacheBackend._inprocess_key_lock``); held only for that, never a compute.
_KEY_LOCK_BOOTSTRAP = threading.Lock()

# The metadata channel backends actually see: an opaque dict they round-trip
# without inspecting (the channel is polymorphic — both CacheMetadata and the
# notebook layer's StatementCacheMetadata flow through it as plain dicts). The
# typed CacheMetadata view lives only at the cash-layer edges.
MetadataDict = dict[str, Any]


def ttl_expired(timestamp: float | None, ttl: float | None, now: float | None = None) -> bool:
    """Has an entry written at *timestamp* outlived *ttl* seconds?

    The one TTL rule every cache path applies. ``None`` never expires.
    ``ttl <= 0`` means "never fresh" and is decided without consulting the
    clock: a same-tick re-read can measure an age of ``0.0`` on a coarse
    timer, and ``0.0 > 0`` would hand back the very entry ``ttl=0`` exists to
    reject. Otherwise an entry is expired once its age exceeds the ttl. An
    entry with no timestamp reads as written at the epoch, so it expires under
    any ttl rather than being served forever.
    """
    if ttl is None:
        return False
    if ttl <= 0:
        return True
    return (time.time() if now is None else now) - (timestamp or 0) > ttl


@dataclass(frozen=True)
class CacheMetadata:
    """Typed, in-memory view of a decorator cache entry's metadata.

    This is the *edge* representation: producers (the ``@cash.cache``
    decorator) build one and call :meth:`to_dict` before handing it to a
    backend; consumers call :meth:`from_dict` on what a backend returns.
    Backends themselves never see this class — they round-trip a plain
    ``dict`` opaquely (the metadata channel is polymorphic; the notebook
    layer pushes a differently-shaped ``StatementCacheMetadata`` through
    the same channel).

    Every field is optional and defaults to ``None`` — "absent" and
    "unset" are deliberately collapsed (a frozen dataclass field always
    has a value). :meth:`to_dict` omits ``None`` fields so the on-the-wire
    dict matches the historical "only-set-keys" shape, and :meth:`from_dict`
    ignores unknown keys and defaults missing ones, so it tolerates caches
    written by older or newer versions.
    """

    key: str | None = None
    created_at: float | None = None
    last_access: float | None = None
    access_count: int | None = None
    size: int | None = None
    storage: list[str] | None = None
    ttl: int | None = None
    # The ttl came from the decorator (``ttl=``), not from a tier's
    # ``default_ttl``: a lowered tier default shortens only the latter, and
    # ``cash inspect`` / ``cash clear --expired`` need to tell them apart.
    ttl_declared: bool | None = None
    execution_time: float | None = None
    # The function's OWN time, excluding everything cash did around it.
    # ``execution_time`` is measured from the top of the wrapper and so
    # includes cache-key hashing and lookup -- fine for reporting what a call
    # cost, useless for asking whether caching PAID, because the overhead
    # being judged is inside the number it would be judged against.
    body_seconds: float | None = None
    # The wall time a hit stands in for: the body's time divided by the
    # threads that were running cached calls alongside it. Sixteen 0.5 s calls
    # on eight threads took 1 s to run, not 8 s (round 20).
    saves_seconds: float | None = None
    outputs: list[str] | None = None
    lineage_hash: str | None = None
    source: str | None = None  # Backend source identifier (e.g. 'RAM', 'disk')
    # Decorator-stamped identity / lineage fields.
    func_name: str | None = None
    args_hash: str | None = None
    state_hash: str | None = None
    timestamp: float | None = None
    auto_file_deps: dict[str, dict[str, float]] | None = None
    iterator_storage: str | None = None
    n_chunks: int | None = None
    # Deserialization instruction; round-tripped so get() can rebuild the value.
    serializer_cls: type | None = None
    # Notebook-annotation flags consumed by TieredBackend / lineage.
    #: Written by ``@cash.cache``: this entry was asked for by a decorator, so
    #: the compute floor, the cost model and the rate ceiling do not gate it
    #: (``TieredBackend.set``). Says nothing about how the value is stored.
    decorator_entry: bool | None = None

    #: The stored value IS what the next call hands back, so the RAM tier must
    #: really copy it and refuses one it cannot (``InMemoryBackend.set``). Set
    #: by ``@cash.cache`` EXCEPT for a ``frozen=True`` function, which has
    #: already promised the result is not modified -- handing the same object
    #: back is what that promises. Split out of ``decorator_entry``: the two
    #: rode on one flag, so ``frozen=True`` silently lost disk persistence
    #: when the rate ceiling started reading it as "not a decorated entry".
    copy_required: bool | None = None

    #: Where the global RNG stood before and after the call that computed this
    #: entry, so a hit can leave it where the body did (``Cash._rng_replay_parts``).
    rng_replay: dict[str, Any] | None = None

    force_persist: bool | None = None
    metadata_only: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of set fields, omitting ``None`` (the wire format)."""
        return {f.name: value for f in fields(self) if (value := getattr(self, f.name)) is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheMetadata:
        """Build from a backend dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    **Error contract**

    * `get()` returns `(None, None)` for missing or corrupt entries.
      Raises `CacheBackendError` on infrastructure
      failures (disk I/O, network, permission errors).
    * `set()` raises `CacheBackendError` on write
      failures.  Implementations should clean up partial writes before raising.
    * `delete()` and `clear()` raise
      `CacheBackendError` on infrastructure failures.

    **Tier-promotion hints**

    * `max_size_bytes` — upper bound on object size this backend is happy
      to host as a tier inside a `TieredBackend`. `None` (default)
      means unbounded. The cap is a *promotion hint*, not a hard gate: a
      bare-backend `set()` writes whatever you give it. Only the tiered
      pipeline skips a tier whose cap the object exceeds.
    """

    # Tier-promotion hint. See class docstring.
    max_size_bytes: int | None = None

    #: What `cost_model` predicts this backend's restore times as: ``"ram"``,
    #: ``"disk"``, ``"redis"`` or ``"s3"``.
    cost_kind: str = "disk"

    #: The ttl an entry written without one gets; None never expires. Set by
    #: backends that take a ``default_ttl``; read through :attr:`default_ttl`.
    _default_ttl: float | None = None

    #: How many times this process wrote the generation stamp, and the token
    #: its last write left. Kept by a backend whose :meth:`generation_token`
    #: can move (the file tier); see ``clear_watch.ClearWatcher``.
    stamp_writes: int = 0
    written_stamp: tuple | None = None

    @property
    def source_label(self) -> str:
        """What this tier is called where an entry says where it came from
        (``RAM``, ``SQLITE``). Backends name themselves with a class
        attribute; the default is the class name."""
        return type(self).__name__

    @property
    def default_ttl(self) -> float | None:
        """The ttl an entry written without one is stamped with, or None."""
        return self._default_ttl

    @property
    def local_dir(self) -> str | None:
        """The local directory this backend keeps its entries in, or None for
        one that keeps them in memory or elsewhere (a server, a bucket)."""
        return None

    def generation_token(self) -> tuple | None:
        """A token that moves when this backend's store is cleared from
        outside the process, or None when it cannot tell."""
        return None

    def set_metadata_only(self, key: str, metadata: dict) -> None:  # noqa: B027 - intentional no-op default
        """Keep *metadata* for *key* without a value, where the backend can.

        Lets a notebook show what a statement cost after a restart even when
        its value was not worth storing. The default keeps nothing.
        """

    def peek_entry(self, key: str) -> tuple[MetadataDict, Any] | None:
        """``(metadata, value)`` for *key* without counting a use, or None.

        Only a backend holding live values can answer cheaply (the RAM tier);
        the default does not answer.
        """
        return None

    def persist_from_memory(self, key: str, rebuild_seconds: float) -> bool:
        """Write a value held only in a faster tier to a slower one when
        rebuilding it would cost *rebuild_seconds*; True when it was written.
        Only a tiered backend has tiers to move a value between."""
        return False

    def hold_notices(self) -> None:  # noqa: B027 - intentional no-op default
        """Hold notices about values not persisted until `release_notices`,
        to say them once for a batch of stores. The default holds nothing."""

    def release_notices(self) -> None:  # noqa: B027 - intentional no-op default
        """Say the notices held since `hold_notices`."""

    @abstractmethod
    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        """Retrieve (metadata, value) from the cache.

        Returns ``(None, None)`` when *key* is not found.  Raises
        `CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Any | None = None) -> None:
        """Set a value in the cache with optional metadata.

        Args:
            key: Cache key
            value: The value to store (raw object)
            metadata: Optional metadata dictionary
            serializer: Optional serializer instance to use for serialization (if backend requires it)

        Raises:
            CacheBackendError: On write failures.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the cache.

        Raises `CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache.

        Raises `CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def list_entries(self) -> list[dict[str, Any]]:
        """List all cache entries with their metadata."""
        ...

    def entry_count(self) -> int:
        """How many entries this backend holds, as cheaply as it can tell.

        The default counts `list_entries`, which reads every entry's
        metadata. Backends that can count without that override it: on a
        file cache of a few thousand entries the read is seconds to tens of
        seconds on Windows, and ``%cash_on`` wants only the number.
        """
        return len(self.list_entries())

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        """Iterate over all items and delete those where ``is_expired(metadata)`` is True.

        Returns the number of deleted items.  The default implementation
        scans `list_entries` and calls `delete` for each expired
        entry.  Subclasses may override for more efficient backend-native
        expiration (e.g. Redis TTL).
        """
        count = 0
        for entry in self.list_entries():
            if is_expired(entry):
                self.delete(entry["key"])
                count += 1
        return count

    def peek_metadata(self, key: str) -> MetadataDict | None:
        """Metadata for *key* WITHOUT counting it as a use, or ``None``.

        For looking, not reading: ``explain()`` asks whether the next call
        would hit, and going through ``get()`` recorded that as an access --
        USES and LAST USED moved in ``cash inspect``, and the file backend
        rewrote the entry to persist the stamp (round 18). The default is
        ``get_metadata``; a backend whose ``get_metadata`` records an access
        overrides this.
        """
        return self.get_metadata(key)

    def get_metadata(self, key: str) -> MetadataDict | None:
        """Get only metadata for a cache key without deserializing the value.

        Returns the metadata dict if the key exists, or ``None`` otherwise.
        The default implementation performs a full ``get()`` and discards the
        value.  Subclasses (e.g. `FileBackend`) may override for a
        more efficient metadata-only read path.
        """
        metadata, _ = self.get(key)
        return metadata

    @staticmethod
    def _init_metadata(metadata: dict[str, Any] | None, key: str) -> dict[str, Any]:
        """Ensure standard metadata fields are set (key, created_at, last_access, access_count).

        Backends call this at the start of ``set()`` to stamp the common
        header fields, then add backend-specific fields (size, storage, etc.).
        """
        if metadata is None:
            metadata = {}
        metadata["key"] = key
        now = time.time()
        metadata.setdefault("created_at", now)
        metadata.setdefault("last_access", now)
        metadata.setdefault("access_count", 0)
        return metadata

    def promotion_size_cap(self) -> int | None:
        """Largest single object this backend accepts via *tiered* promotion.

        The default is the static class-level :attr:`max_size_bytes` hint
        (Redis 10 MB, SQLite 100 MB, unbounded elsewhere). Backends whose
        cap is *dynamic* — notably the file tier, whose LRU cap is scaled to
        free disk — override this to derive a per-object refusal
        threshold from their own instance cap, so an object too big to hold
        without thrashing is skipped rather than written-then-evicted.

        Consulted only by ``TieredBackend.set()``; a bare-backend ``set()``
        still writes whatever it is given.
        """
        return type(self).max_size_bytes

    def tier_labels(self) -> list[str]:
        """Ordered labels of the storage tiers this backend exposes.

        Used by the notebook badge renderer to lay out one indicator dot
        per tier. The default returns ``[source_label]`` (or the class
        name as a fallback) — a single-tier label. ``TieredBackend``
        overrides it to return its children's labels in configured order.
        """
        return [self.source_label]

    def shutdown(self) -> None:  # noqa: B027 - intentional no-op default; subclasses override as needed
        """Perform any necessary cleanup before exit (e.g. waiting for async writes)."""

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        """Return a context manager that single-flights computes for *key*.

        Consumed by ``Cash._compute_with_lock`` when ``use_locking=True``:
        concurrent same-key callers serialize on this lock so the expensive
        miss→compute→store body runs once and the rest observe the stored
        result (double-checked inside the lock).

        The default is an **in-process** per-key ``threading.RLock``.
        Cross-*process* single-flight needs the Redis backend, which overrides
        this with a distributed lock.

        ``RLock`` (not ``Lock``) so a compute that re-enters the same key on
        the same thread — memoized recursion, a fn that calls itself — re-
        acquires instead of deadlocking. Other threads still block until the
        leader fully releases.
        """
        return self._inprocess_key_lock(key)

    def _inprocess_key_lock(self, key: str) -> threading.RLock:
        """The process-local ``RLock`` for *key*, created on first use.

        The registry holds the locks weakly: a lock lives while some caller
        holds or waits on it, so one lock per key ever used does not pile up
        over a long session. Created lazily because many subclasses do not
        call ``super().__init__``.
        """
        registry = self.__dict__.get("_inprocess_key_locks")
        if registry is None:
            with _KEY_LOCK_BOOTSTRAP:
                registry = self.__dict__.get("_inprocess_key_locks")
                if registry is None:
                    self.__dict__["_inprocess_key_locks_meta"] = threading.Lock()
                    registry = weakref.WeakValueDictionary()
                    self.__dict__["_inprocess_key_locks"] = registry
        with self.__dict__["_inprocess_key_locks_meta"]:
            lk = registry.get(key)
            if lk is None:
                lk = registry[key] = threading.RLock()
        return lk
