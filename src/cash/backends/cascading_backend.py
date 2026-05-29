"""Cascading multi-backend (writes to every tier on set, reads first hit)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from ._base import CacheBackend, MetadataDict
from .serialization import Serializer

logger = logging.getLogger(__name__)

__all__ = ["_MultiBackendMixin", "CascadingBackend"]


class _MultiBackendMixin:
    """Shared delegation logic for backends wrapping multiple sub-backends.

    Provides default implementations for methods that simply iterate over
    ``self.backends`` and delegate.  Used by `CascadingBackend` and
    `TieredBackend` (in ``tiered_backend.py``).
    """

    backends: list[CacheBackend]

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
                    type(backend).__name__, e,
                )

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

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        for i, backend in enumerate(self.backends):
            metadata, value = backend.get(key)
            # Presence test: metadata-None means key absent. A non-None
            # metadata with value=None is a legitimately stored None.
            if metadata is not None:
                # Read-repair: write back to all earlier backends that missed
                for j in range(i):
                    try:
                        self.backends[j].set(key, value, metadata)
                    except (CacheBackendError, OSError) as exc:
                        logger.debug("Read-repair failed for key %r to backend %d: %s", key, j, exc)
                return metadata, value
        return None, None

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        for backend in self.backends:
            backend.set(key, value, metadata, serializer)
