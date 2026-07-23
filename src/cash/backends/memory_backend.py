"""In-memory cache backend with LRU eviction."""

from __future__ import annotations

import builtins
import copy
import ctypes
import logging
import pickle
import sys
import time
from collections.abc import Callable
from typing import Any

from ._base import CacheBackend, MetadataDict
from .serialization import Serializer

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

__all__ = ["InMemoryBackend"]


class InMemoryBackend(CacheBackend):
    """
    In-memory cache backend using a dictionary.
    Supports smart eviction based on memory pressure.
    """
    source_label: str = "RAM"

    def __init__(self, max_memory_percent: float = 0.9, check_interval: int = 10, max_entries: int | None = None,
                 max_size_bytes: int | None = None) -> None:
        """
        Args:
            max_memory_percent: Memory usage percentage (0.0 to 1.0) at which to trigger eviction.
            check_interval: Number of 'set' operations between memory checks.
            max_entries: Maximum number of cache entries. When exceeded, LRU eviction is triggered.
                         None means unlimited entries (eviction only via memory pressure).
            max_size_bytes: Soft byte cap for the RAM tier. When the tracked
                         total exceeds it, least-recently-used entries are evicted down to
                         ~90% of the cap. ``None`` (default) means unbounded — eviction is
                         driven only by ``max_entries`` and psutil memory pressure, exactly
                         as before. The factory sets this to a modest fraction of system RAM
                         (``adaptive_caps.resolve_ram_cap``) so the RAM tier is bounded
                         independently of the disk tier.
        """
        self._store: dict[str, tuple[MetadataDict, Any]] = {}  # Stores (metadata, value)
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.max_entries = max_entries
        self._max_size_bytes = max_size_bytes
        self._current_size_bytes = 0
        self._set_count = 0

    @staticmethod
    def _safe_deep_copy(value: Any, key: str = "<unknown>") -> Any:
        """Deep-copy *value* with a fast path for pandas and a fallback for uncopyable objects."""
        try:
            type_name = type(value).__name__
            if type_name in ('DataFrame', 'Series'):
                return value.copy(deep=True)
            return copy.deepcopy(value)
        except (TypeError, pickle.PicklingError, RecursionError, AttributeError):
            logger.debug("Could not deep-copy value for key %r, returning reference", key)
            return value

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        if key in self._store:
            metadata, value = self._store[key]

            metadata['last_access'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            metadata.setdefault('source', self.source_label)

            return metadata, self._safe_deep_copy(value, key)
        return None, None

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        metadata = self._init_metadata(metadata, key)

        if 'storage' not in metadata:
            metadata['storage'] = ['RAM']

        # Calculate size (approximate) - do this BEFORE copy to be fast logic-wise (size is same)
        size = self._get_object_size(value)
        metadata['size'] = size

        # Byte-cap bookkeeping: on replacement, discount the old entry's size
        # before recording the new one so the running total stays accurate.
        if key in self._store:
            self._current_size_bytes -= self._store[key][0].get('size', 0)
        self._store[key] = (metadata, self._safe_deep_copy(value, key))
        self._current_size_bytes += size

        # Check max_entries limit
        if self.max_entries is not None and len(self._store) > self.max_entries:
            self._evict_lru(len(self._store) - self.max_entries)

        # Enforce the soft byte cap (adaptive RAM cap; None = unbounded).
        if self._max_size_bytes is not None and self._current_size_bytes > self._max_size_bytes:
            self._evict_to_byte_cap()

        # Check memory pressure periodically
        self._set_count += 1
        if self._set_count % self.check_interval == 0:
            self._check_and_evict()

    def _drop(self, key: str) -> None:
        """Remove *key*, keeping the byte-cap running total in sync."""
        entry = self._store.pop(key, None)
        if entry is not None:
            self._current_size_bytes -= entry[0].get('size', 0)

    def delete(self, key: str) -> None:
        self._drop(key)

    def clear(self) -> None:
        self._store.clear()
        self._current_size_bytes = 0
        # Also try to free memory back to OS
        self._try_malloc_trim()

    def list_entries(self) -> list[dict[str, Any]]:
        return [meta for meta, _ in self._store.values()]

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        keys_to_delete = []
        for key, (meta, _) in self._store.items():
            if is_expired(meta):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._drop(key)

        if keys_to_delete:
            self._try_malloc_trim()

        return len(keys_to_delete)

    def _get_object_size(self, obj: Any, seen: builtins.set[int] | None = None) -> int:
        """Estimate object size in bytes (recursive)."""
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        try:
            # Prefer nbytes for numpy/pandas
            if hasattr(obj, 'nbytes'):
                return obj.nbytes
            if hasattr(obj, 'memory_usage'):
                # pandas DataFrame/Series
                # OPTIMIZATION: Use deep=False for speed (deep=True scans all object columns)
                try:
                    mem = obj.memory_usage(deep=False)
                    if hasattr(mem, 'sum'):
                        return mem.sum()
                    return mem
                except (TypeError, AttributeError):
                    # Fallback to sys.getsizeof below when memory_usage is unavailable
                    pass

            size = sys.getsizeof(obj)

            if isinstance(obj, dict):
                size += sum(self._get_object_size(v, seen) for v in obj.values())
                # Also keys
                size += sum(self._get_object_size(k, seen) for k in obj)
            elif isinstance(obj, (list, tuple, set)):
                size += sum(self._get_object_size(i, seen) for i in obj)

            return size
        except (TypeError, RecursionError, ValueError):
            logger.debug("Could not estimate size of %s object", type(obj).__name__, exc_info=True)
            return 0

    def _check_and_evict(self) -> None:
        """Check memory usage and evict items if threshold is exceeded."""
        if psutil is None:
            return

        try:
            mem = psutil.virtual_memory()
            if mem.percent / 100.0 > self.max_memory_percent:
                self._evict()
        except (OSError, AttributeError) as exc:
            logger.debug("Memory check failed: %s", exc)

    def _evict(self) -> None:
        """Evict items until memory usage is safe (target: 90% of max threshold)."""
        target_percent = self.max_memory_percent * 0.9

        items = []
        for key, (meta, _val) in self._store.items():
            exec_time = meta.get('execution_time', 0.001)
            if exec_time <= 0:
                exec_time = 0.001

            access_count = meta.get('access_count', 0)

            size = meta.get('size', 1)
            if size <= 0:
                size = 1

            score = (exec_time * access_count) / size
            last_access = meta.get('last_access', 0)
            items.append((score, last_access, key, size))

        # Sort: Primary=Score (asc), Secondary=LastAccess (asc) — lowest/oldest evicted first
        items.sort(key=lambda x: (x[0], x[1]))

        evicted_count = 0

        for _score, _last_access, key, _size in items:
            if key in self._store:
                self._drop(key)
                evicted_count += 1

                if psutil is None:
                    break
                mem = psutil.virtual_memory()
                if mem.percent / 100.0 <= target_percent:
                    break

        if evicted_count > 0:
            self._try_malloc_trim()

    def _evict_to_byte_cap(self) -> None:
        """Evict least-recently-used entries until under ~90% of the byte cap.

        Mirrors the file backend's LRU eviction: sort by ``last_access``
        (oldest first) and drop until the running total falls to 90% of the
        cap, giving headroom so the next few writes don't immediately
        re-trigger eviction. No-op when the cap is unset or already satisfied.
        """
        if not self._max_size_bytes or self._current_size_bytes <= self._max_size_bytes:
            return
        target = self._max_size_bytes * 0.9
        items = [
            (meta.get('last_access', 0), key)
            for key, (meta, _) in self._store.items()
        ]
        items.sort()  # oldest first
        evicted = 0
        for _last_access, key in items:
            if self._current_size_bytes <= target:
                break
            self._drop(key)
            evicted += 1
        if evicted:
            self._try_malloc_trim()

    def _evict_lru(self, count: int):
        """Evict the N least-recently-used entries."""
        if count <= 0:
            return
        items = []
        for key, (meta, _) in self._store.items():
            last_access = meta.get('last_access', 0)
            items.append((last_access, key))
        items.sort()  # oldest first
        for _, key in items[:count]:
            self._drop(key)

    def _try_malloc_trim(self) -> None:
        """Try to clean up memory on Linux."""
        if sys.platform.startswith('linux'):
            try:
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except (OSError, AttributeError):
                # Best-effort memory cleanup; safe to ignore on non-glibc systems
                pass
