"""Cascading (multi-backend) and async wrapper backends."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from ._base import CacheBackend, CacheMetadata
from .serialization import Serializer

logger = logging.getLogger(__name__)

__all__ = ["_MultiBackendMixin", "CascadingBackend", "AsyncBackendWrapper"]


class _MultiBackendMixin:
    """Shared delegation logic for backends wrapping multiple sub-backends.

    Provides default implementations for methods that simply iterate over
    ``self.backends`` and delegate.  Used by :class:`CascadingBackend` and
    :class:`TieredBackend` (in ``tiered_backend.py``).
    """

    backends: list[CacheBackend]

    def get_metadata(self, key: str) -> dict | None:
        """Get only metadata for a cache key from the first backend that has it.

        Checks backends in order (fast → slow), returning the first hit.
        Supports metadata-only entries written by :meth:`set_metadata_only`.
        """
        for backend in self.backends:
            if hasattr(backend, 'get_metadata'):
                meta = backend.get_metadata(key)
                if meta is not None:
                    return meta
        return None

    def set_metadata_only(self, key: str, metadata: dict) -> None:
        """Persist metadata without data payload to all backends that support it.

        Shared implementation for both TieredBackend and CascadingBackend
        via _MultiBackendMixin — not duplicated in individual backend classes.
        """
        for backend in self.backends:
            if hasattr(backend, 'set_metadata_only'):
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
                key = entry.get('key')
                if key not in seen_keys:
                    seen_keys.add(key)
                    entries.append(entry)
        return entries

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

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        total = 0
        seen_keys = set()
        for backend in self.backends:
            for entry in list(backend.list_entries()):
                key = entry.get('key')
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if is_expired(entry):
                    for b in self.backends:
                        b.delete(key)
                    total += 1
        return total


class CascadingBackend(_MultiBackendMixin, CacheBackend):
    """A backend that cascades reads and writes across multiple backends.

    On ``get()``, tries backends in order and read-repairs lower-priority
    backens on hit.  On ``set()``, writes to all backends.
    """

    def __init__(self, backends: list[CacheBackend]) -> None:
        self.backends = backends

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        for i, backend in enumerate(self.backends):
            metadata, value = backend.get(key)
            if metadata is not None and value is not None:
                # Read-repair: write back to all earlier backends that missed
                for j in range(i):
                    try:
                        self.backends[j].set(key, value, metadata)
                    except (CacheBackendError, OSError) as exc:
                        logger.debug("Read-repair failed for key %r to backend %d: %s", key, j, exc)
                return metadata, value
        return None, None

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        for backend in self.backends:
            backend.set(key, value, metadata, serializer)


class AsyncBackendWrapper(CacheBackend):
    """
    Wrapper that performs write operations (set, delete) asynchronously
    using a ThreadPoolExecutor. Read operations are synchronous.
    """

    def __init__(self, backend: CacheBackend, max_workers: int = 1) -> None:
        self.backend = backend
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        # Note: value must be thread-safe or copied if we pass it to thread.
        # Since we are async, the main thread might mutate value after calling set.
        # We should probably copy it here if it's mutable?
        # But deepcopy is expensive.
        # For now, assume caller handles it or we rely on backend's serialization/copying.
        # But backend.set runs in another thread.
        # If we use InMemoryBackend, it deepcopies.
        # If we use FileBackend, it serializes.
        # Serialization might be affected if object changes during serialization?
        # Yes.
        # Ideally we deepcopy here if we want true safety, but that kills performance.
        # Let's proceed without deepcopy for now, assuming typical usage (cache result then don't mutate it immediately).

        future = self.executor.submit(self.backend.set, key, value, metadata, serializer)
        self._clean_futures()
        self._futures.append(future)

    def delete(self, key: str) -> None:
        future = self.executor.submit(self.backend.delete, key)
        self._clean_futures()
        self._futures.append(future)

    def clear(self) -> None:
        self.shutdown()
        self.backend.clear()

    def list_entries(self) -> list[dict[str, Any]]:
        return self.backend.list_entries()

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        return self.backend.cleanup_expired(is_expired)

    def _clean_futures(self) -> None:
        self._futures = [f for f in self._futures if not f.done()]

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        return self.backend.lock(key)
