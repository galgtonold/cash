"""Abstract base class and shared types for cache backends."""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["CacheMetadata", "CacheBackend"]


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

    Error contract
    --------------
    * ``get()`` returns ``(None, None)`` for missing or corrupt entries.
      Raises :class:`~cash.exceptions.CacheBackendError` on infrastructure
      failures (disk I/O, network, permission errors).
    * ``set()`` raises :class:`~cash.exceptions.CacheBackendError` on write
      failures.  Implementations should clean up partial writes before raising.
    * ``delete()`` and ``clear()`` raise
      :class:`~cash.exceptions.CacheBackendError` on infrastructure failures.
    """

    @abstractmethod
    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        """Retrieve (metadata, value) from the cache.

        Returns ``(None, None)`` when *key* is not found.  Raises
        :class:`~cash.exceptions.CacheBackendError` on infrastructure errors.
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

        Raises :class:`~cash.exceptions.CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache.

        Raises :class:`~cash.exceptions.CacheBackendError` on infrastructure errors.
        """
        ...

    @abstractmethod
    def list_entries(self) -> list[dict[str, Any]]:
        """List all cache entries with their metadata."""
        ...

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        """Iterate over all items and delete those where ``is_expired(metadata)`` is True.

        Returns the number of deleted items.  The default implementation
        scans :meth:`list_entries` and calls :meth:`delete` for each expired
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
        value.  Subclasses (e.g. :class:`FileBackend`) may override for a
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
