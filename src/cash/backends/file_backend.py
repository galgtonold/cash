"""File-based cache backend with LRU eviction and optional compression."""

from __future__ import annotations

import glob
import gzip
import logging
import os
import pickle
import threading
import time
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError

from ._base import CacheBackend, CacheMetadata
from .serialization import PickleSerializer, Serializer

logger = logging.getLogger(__name__)

__all__ = ["FileBackend"]


class FileBackend(CacheBackend):
    """File-based cache backend.

    .. warning:: Uses :mod:`pickle` for serialization.  Cache files are
       assumed to originate from the local machine.  Do not load cache
       directories from untrusted sources — see :file:`SECURITY.md`.
    """
    source_label: str = "DISK"

    def __init__(self, cache_dir: str, compress: bool = False, max_size_bytes: int | None = None, flush_interval: int = 5, default_ttl: int | None = None) -> None:
        """
        Args:
            cache_dir: Directory for cache files.
            compress: Whether to gzip-compress data files.
            max_size_bytes: Maximum total cache size in bytes (triggers LRU eviction).
            flush_interval: Seconds between metadata flush cycles.
            default_ttl: Default time-to-live in seconds for cache entries. None = no expiration.
        """
        # Resolve to absolute path at init time so os.chdir() won't break file paths
        self.cache_dir = os.path.abspath(cache_dir)
        self.compress = compress
        self._max_size_bytes = max_size_bytes
        self._default_ttl = default_ttl
        self._current_size_bytes = 0
        self._dirty_metadata: set[str] = set()
        self._metadata_cache: dict[str, dict] = {}  # partial cache for LRU tracking
        self._lock = threading.RLock()
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()

        # Lazy initialization: defer directory creation, stat scanning,
        # and background thread to first actual use.
        self._initialized = False
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazily create cache directory, scan stats, and start flusher thread.

        Thread-safe via double-checked locking so the heavy I/O only
        happens once, on first real cache operation.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            os.makedirs(self.cache_dir, exist_ok=True)
            self._init_stats()
            if self._flush_interval > 0:
                self._flusher_thread = threading.Thread(
                    target=self._flush_periodically, daemon=True,
                )
                self._flusher_thread.start()
            self._initialized = True

    def _init_stats(self) -> None:
        """Scan directory to calculate total size and load metadata for LRU."""
        total_size = 0
        for meta_path in glob.glob(os.path.join(self.cache_dir, "*.meta")):
            try:
                # size of meta file
                total_size += os.path.getsize(meta_path)

                # Load meta to get data size and last access
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)

                key = meta.get('key')
                if key:
                    self._metadata_cache[key] = meta

                data_path = meta_path.replace(".meta", ".data")
                if os.path.exists(data_path):
                    total_size += os.path.getsize(data_path)

            except (OSError, pickle.PickleError):
                logger.debug("Skipping unreadable metadata file %s during init", meta_path, exc_info=True)
        self._current_size_bytes = total_size

    def _flush_periodically(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._flush_interval):
                break
            self._flush_metadata()

    def _flush_metadata(self) -> None:
        """Write dirty metadata to disk."""
        with self._lock:
            if not self._dirty_metadata:
                return

            keys_to_flush = list(self._dirty_metadata)
            self._dirty_metadata.clear()

        for key in keys_to_flush:
            try:
                meta = self._metadata_cache.get(key)
                if meta:
                    meta_path, _ = self._get_paths(key)
                    with open(meta_path, 'wb') as f:
                        pickle.dump(meta, f)
            except (OSError, pickle.PickleError) as exc:
                logger.debug("Failed to flush metadata for key %r: %s", key, exc)

    def _get_paths(self, key: str) -> tuple[str, str]:
        import hashlib
        safe_name = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return (
            os.path.join(self.cache_dir, f"{safe_name}.meta"),
            os.path.join(self.cache_dir, f"{safe_name}.data")
        )

    def get_metadata(self, key: str) -> dict | None:
        """Get only metadata for a cache key without deserializing the value.

        This is useful for lazy deserialization - check if a key exists and
        inspect its metadata before committing to deserialize the data.
        Also returns metadata-only entries (where data was skipped due to
        size-aware caching) — these still carry execution_time, output_lineages,
        etc. for badge display and upstream simulation.

        Returns:
            Metadata dict if key exists, None otherwise.
        """
        self._ensure_initialized()
        cached_meta = self._metadata_cache.get(key)

        meta_path, data_path = self._get_paths(key)
        if not os.path.exists(meta_path):
            return None

        try:
            if cached_meta:
                metadata = cached_meta
            else:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                self._metadata_cache[key] = metadata

            ttl = metadata.get('ttl', self._default_ttl)
            if ttl is not None:
                created_at = metadata.get('created_at', 0)
                if time.time() - created_at > ttl:
                    return None

            return metadata
        except (OSError, pickle.PickleError):
            return None

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        self._ensure_initialized()
        # Check memory cache first for metadata
        cached_meta = self._metadata_cache.get(key)

        meta_path, data_path = self._get_paths(key)

        if not os.path.exists(meta_path) or not os.path.exists(data_path):
            # Cleanup specific inconsistency
            if key in self._metadata_cache:
                with self._lock:
                    del self._metadata_cache[key]
            return None, None

        try:
            if cached_meta:
                metadata = cached_meta
            else:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                self._metadata_cache[key] = metadata

            ttl = metadata.get('ttl', self._default_ttl)
            if ttl is not None:
                created_at = metadata.get('created_at', 0)
                if time.time() - created_at > ttl:
                    # Entry expired - delete it
                    self.delete(key)
                    return None, None

            # Update Access Time (Async)
            metadata['last_access'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1

            with self._lock:
                self._dirty_metadata.add(key)

            # Check for compression
            is_compressed = metadata.get('compressed', False)
            opener = gzip.open if is_compressed else open

            serializer_cls = metadata.get('serializer_cls', PickleSerializer)
            serializer = serializer_cls()

            try:
                with opener(data_path, 'rb') as f:
                    data_bytes = f.read()
                    value = serializer.deserialize(data_bytes)
            except (OSError, gzip.BadGzipFile):
                # Fallback
                with open(data_path, 'rb') as f:
                    data_bytes = f.read()
                    value = serializer.deserialize(data_bytes)

            return metadata, value
        except (OSError, pickle.PickleError, ValueError) as exc:
            logger.debug("Cache get failed for key %r: %s", key, exc)
            return None, None

    def _write_cache_files(self, key: str, meta_path: str, data_path: str, metadata: dict, serialized_value: bytes) -> None:
        """Write serialized data and metadata to disk, updating size tracking.

        Raises:
            OSError, pickle.PickleError, ValueError: on write failure (caller handles cleanup).
        """
        opener = gzip.open if self.compress else open
        with opener(data_path, 'wb') as f:
            f.write(serialized_value)

        try:
            actual_data_size = os.path.getsize(data_path)
            metadata['size'] = actual_data_size
        except OSError:
            # Fall back to serialized byte count if stat fails
            metadata['size'] = len(serialized_value)
            logger.debug("Could not stat data file %s; using serialized length", data_path, exc_info=True)

        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        try:
            actual_meta_size = os.path.getsize(meta_path)
        except OSError:
            actual_meta_size = 0

        with self._lock:
            # Handle replacement: subtract old size
            if key in self._metadata_cache:
                old_meta = self._metadata_cache[key]
                self._current_size_bytes -= old_meta.get('size', 0)
                # Subtract estimated old meta size (using new as proxy)
                self._current_size_bytes -= actual_meta_size

            self._metadata_cache[key] = metadata
            self._current_size_bytes += metadata['size'] + actual_meta_size

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        self._ensure_initialized()
        meta_path, data_path = self._get_paths(key)

        metadata = self._init_metadata(metadata, key)

        # Set TTL if not already specified and we have a default
        if 'ttl' not in metadata and self._default_ttl is not None:
            metadata['ttl'] = self._default_ttl

        if serializer is None:
            serializer = PickleSerializer()

        # Serialize (let serialization errors propagate directly)
        serialized_value = serializer.serialize(value)

        metadata['compressed'] = self.compress

        try:
            self._write_cache_files(key, meta_path, data_path, metadata, serialized_value)

            if self._max_size_bytes:
                self._check_and_evict()

        except (OSError, pickle.PickleError, ValueError) as exc:
            # Cleanup partial files on write failure
            logger.debug("Cache set failed for key %r, cleaning up: %s", key, exc)
            try:
                if os.path.exists(data_path):
                    os.remove(data_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except OSError:
                # Best-effort cleanup; files may already be gone
                logger.debug("Cleanup of partial cache files failed for key %r", key, exc_info=True)
            raise CacheBackendError(f"Cache set failed for key {key!r}: {exc}") from exc

    def set_metadata_only(self, key: str, metadata: dict) -> None:
        """Store only metadata without data payload.

        Used when the data itself is too large to persist (size-aware caching skip)
        but we still want to retain execution_time, output_lineages, etc.
        for badge display and upstream simulation after kernel restart.

        Will NOT overwrite an existing entry that has full data (both .meta and .data),
        since that entry is more valuable.
        """
        self._ensure_initialized()
        meta_path, data_path = self._get_paths(key)

        # Don't overwrite a full cache entry (has both .meta and .data)
        if os.path.exists(meta_path) and os.path.exists(data_path):
            return

        metadata = dict(metadata)  # Don't mutate caller's dict
        metadata['key'] = key
        metadata['metadata_only'] = True
        metadata.setdefault('created_at', time.time())
        metadata.setdefault('last_access', time.time())
        metadata.setdefault('access_count', 0)
        metadata.setdefault('size', 0)

        try:
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            with self._lock:
                self._metadata_cache[key] = metadata
        except OSError as exc:
            logger.debug("Failed to write metadata-only entry for key %r: %s", key, exc)

    def delete(self, key: str) -> None:
        self._ensure_initialized()
        meta_path, data_path = self._get_paths(key)

        size_to_remove = 0
        if key in self._metadata_cache:
            size_to_remove = self._metadata_cache[key].get('size', 0)

        with self._lock:
            if key in self._metadata_cache:
                del self._metadata_cache[key]
            if key in self._dirty_metadata:
                self._dirty_metadata.remove(key)
            self._current_size_bytes -= size_to_remove

        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError as exc:
                logger.debug("Failed to remove metadata file %s: %s", meta_path, exc)
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
            except OSError as exc:
                logger.debug("Failed to remove data file %s: %s", data_path, exc)

    def _check_and_evict(self) -> None:
        """Evict items if over max size."""
        if not self._max_size_bytes or self._current_size_bytes <= self._max_size_bytes:
            return

        keys_to_delete = []
        with self._lock:
            items = []
            for k, m in self._metadata_cache.items():
                items.append((m.get('last_access', 0), k, m.get('size', 0)))

            items.sort(key=lambda x: x[0])

            evicted_size = 0
            needed = self._current_size_bytes - (self._max_size_bytes * 0.9)

            for _last_access, key, size in items:
                if evicted_size >= needed:
                    break
                keys_to_delete.append(key)
                evicted_size += size

        # Delete outside the first lock scope (delete acquires lock)
        for key in keys_to_delete:
            self.delete(key)

    def clear(self) -> None:
        self._ensure_initialized()
        for ext in ["*.meta", "*.data"]:
            for f in glob.glob(os.path.join(self.cache_dir, ext)):
                try:
                    os.remove(f)
                except OSError:
                    # Best-effort removal during cache clear; file may be locked
                    logger.debug("Could not remove cache file %s during clear", f, exc_info=True)
        with self._lock:
            self._metadata_cache.clear()
            self._dirty_metadata.clear()
            self._current_size_bytes = 0

    def shutdown(self) -> None:
        self._stop_event.set()
        if hasattr(self, '_flusher_thread'):
            self._flusher_thread.join(timeout=1.0)
        if self._initialized:
            self._flush_metadata()

    def list_entries(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        entries = []
        for meta_path in glob.glob(os.path.join(self.cache_dir, "*.meta")):
            try:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                    entries.append(metadata)
            except (OSError, pickle.PickleError):
                logger.debug("Skipping unreadable metadata file %s in list_entries", meta_path, exc_info=True)
        return entries

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        self._ensure_initialized()
        count = 0
        for meta_path in glob.glob(os.path.join(self.cache_dir, "*.meta")):
            try:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)

                if is_expired(metadata):
                    # Delete both
                    data_path = meta_path.replace(".meta", ".data")
                    os.remove(meta_path)
                    if os.path.exists(data_path):
                        os.remove(data_path)
                    count += 1
            except (OSError, pickle.PickleError):
                logger.debug("Skipping unreadable metadata file %s during cleanup", meta_path, exc_info=True)
        return count
