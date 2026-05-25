"""Abstract base class and shared types for cache backends."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["CacheMetadata", "CacheBackend", "PendingWrites"]


class PendingWrites:
    """Per-backend background-write scheduler.

    Used by backends whose writes are slow enough that blocking the
    user's calling thread on every ``set()`` would noticeably hurt
    cell-execution latency (File, SQLite, Redis, S3 — anything that
    touches a disk or a network).

    Semantics
    ---------
    * One single-worker executor per backend, so writes to *that*
      backend are naturally serialized (no concurrent fsync, no
      reordered writes for the same key).
    * Serialization of the user's value MUST happen on the calling
      thread, not inside the worker, so that a post-``set()`` mutation
      can't corrupt the cached copy. The helper just dispatches the
      already-prepared payload; it has no opinion about *what* the
      bytes are.
    * ``submit(key, fn, *args)`` records the future under ``key`` so
      subsequent ``get(key)`` / ``delete(key)`` calls on the owning
      backend can wait for it. If a previous write for the same key is
      still pending, the new submission waits for the old one first —
      this keeps "set k=a; set k=b" ordering deterministic.
    * ``wait(key)`` blocks until any pending write for that key resolves
      and re-raises its exception, so failures surface on the next
      observation of that key.
    * ``shutdown(wait=True)`` blocks until every in-flight write
      completes — called by ``CacheBackend.shutdown()`` (which itself
      runs from Cash's ``atexit`` handler).
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        # Per-thread state. ``current_key`` is set on the worker while it
        # processes a task, so re-entrant ``drain(key)`` / ``wait(key)``
        # calls from inside that same task (e.g. via _check_and_evict)
        # can detect "I'd be waiting on myself" and skip the block
        # instead of deadlocking.
        self._tls = threading.local()

    def _run_task(self, key: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        """Worker-side wrapper: marks ``current_key`` so re-entrant
        ``drain``/``wait`` for the same key see the marker and bail out."""
        self._tls.current_key = key
        try:
            return fn(*args, **kwargs)
        finally:
            self._tls.current_key = None

    def submit(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Submit ``fn(*args, **kwargs)`` to run in the background, tagged with *key*.

        If a previous submission for the same key is still in flight,
        wait for it before submitting the new one. This preserves write
        ordering for repeated sets of the same key.
        """
        with self._lock:
            prev = self._pending.get(key)
        # Wait OUTSIDE the lock to avoid blocking other keys' submissions
        # while a slow write completes.
        if prev is not None and not prev.done():
            try:
                prev.result()
            except Exception:  # noqa: BLE001 — surfaces later via wait(key)
                pass
        with self._lock:
            if self._shutdown:
                raise RuntimeError("PendingWrites: executor has been shut down")
            future = self._executor.submit(self._run_task, key, fn, args, kwargs)
            self._pending[key] = future
        return future

    def wait(self, key: str) -> None:
        """Block until the pending write for *key* (if any) finishes.

        Re-raises any exception the worker raised, so background write
        failures surface at the next ``get()``/``delete()``.

        Re-entrancy guard: if the calling thread is the worker thread
        currently processing the same key, we'd be waiting on our own
        future. Skip the wait — we're guaranteed up-to-date by virtue
        of being mid-write.
        """
        if getattr(self._tls, "current_key", None) == key:
            return
        with self._lock:
            future = self._pending.get(key)
        if future is not None:
            future.result()  # re-raises on failure

    def drain(self, key: str) -> None:
        """Wait for the pending write for *key* and forget it.

        Used by ``delete(key)`` so the delete-then-set race produces the
        expected absence rather than a ghost entry from a slow write.
        Swallows any exception the worker raised — we're throwing the
        result away anyway.

        Re-entrancy guard: when the worker calls drain on its own key
        (e.g. through ``_check_and_evict`` deciding to evict the entry
        we're currently writing), waiting on the future would deadlock.
        The caller is the one producing the value, so we let them
        proceed; the write will complete after this method returns.
        """
        if getattr(self._tls, "current_key", None) == key:
            with self._lock:
                self._pending.pop(key, None)
            return
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None:
            try:
                future.result()
            except Exception:  # noqa: BLE001 — we're about to delete, exception is moot
                pass

    def pending_count(self) -> int:
        """How many writes are still in flight. Useful for status panels."""
        with self._lock:
            return sum(1 for f in self._pending.values() if not f.done())

    def wait_all(self) -> None:
        """Block until every pending write completes.

        Used by bulk reads (``list_entries``, ``cleanup_expired``) and
        destructive ops (``clear``) so they observe consistent state
        rather than racing in-flight writes. Re-entrancy guard same as
        ``wait``/``drain``: skip if the calling thread is the worker.
        """
        if getattr(self._tls, "current_key", None) is not None:
            return
        with self._lock:
            futures = list(self._pending.values())
        for f in futures:
            try:
                f.result()
            except Exception:  # noqa: BLE001 — surfaced separately via wait(key)
                pass

    def shutdown(self, wait: bool = True) -> None:
        """Block (when ``wait=True``) until every in-flight write finishes."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait)


class CacheMetadata(TypedDict, total=False):
    """Typed dictionary describing cache entry metadata.

    Fields are populated by backends during ``set()`` and returned by
    ``get()``.  Not all fields are present on every entry — ``total=False``
    marks all keys as optional so callers must use ``.get()`` for
    non-guaranteed fields.
    """

    key: str
    created_at: float
    last_access: float
    access_count: int
    size: int
    storage: list[str]
    ttl: int
    execution_time: float
    outputs: list[str]
    lineage_hash: str
    source: str  # Backend source identifier (e.g. 'RAM', 'disk')


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

    @abstractmethod
    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        """Retrieve (metadata, value) from the cache.

        Returns ``(None, None)`` when *key* is not found.  Raises
        `CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Any | None = None) -> None:
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

    def get_metadata(self, key: str) -> CacheMetadata | None:
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
        metadata['key'] = key
        now = time.time()
        metadata.setdefault('created_at', now)
        metadata.setdefault('last_access', now)
        metadata.setdefault('access_count', 0)
        return metadata

    def tier_labels(self) -> list[str]:
        """Ordered labels of the storage tiers this backend exposes.

        Used by the notebook badge renderer to lay out one indicator dot
        per tier. The default returns ``[source_label]`` (or the class
        name as a fallback) — a single-tier label. Composite backends
        (``TieredBackend``, ``CascadingBackend``) override to return
        their children's labels in configured order.
        """
        return [getattr(type(self), "source_label", None) or type(self).__name__]

    def shutdown(self) -> None:  # noqa: B027 - intentional no-op default; subclasses override as needed
        """Perform any necessary cleanup before exit (e.g. waiting for async writes)."""

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        """Return a context manager that acquires a lock for the given key.

        Default implementation does nothing (no-op context manager).
        """
        from contextlib import nullcontext
        return nullcontext()
