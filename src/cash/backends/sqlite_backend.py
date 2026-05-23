"""SQLite-based cache backend for efficient storage of many small entries."""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheSerializationError

from ._base import CacheBackend, CacheMetadata
from .serialization import PickleSerializer, Serializer

logger = logging.getLogger(__name__)

__all__ = ["SQLiteBackend"]


class SQLiteBackend(CacheBackend):
    """
    SQLite-based cache backend.

    Uses a single SQLite database file for all cache entries.
    Better than FileBackend for many small entries.

    Args:
        db_path: Path to the SQLite database file.
        default_ttl: Default time-to-live in seconds. None = no expiration.
        max_size_bytes: Maximum total data size. None = unlimited.
        wal_mode: Use WAL journal mode for better concurrency (default: True).
    """
    source_label: str = "SQLITE"
    # Tier-promotion hint: SQLite's row/blob handling degrades on
    # multi-hundred-MB values, so the tiered pipeline skips it past
    # this cap. Bare-backend writes are not gated.
    max_size_bytes: int | None = 100 * 1024 * 1024

    def __init__(self, db_path: str = '.cash/cache.db', default_ttl: int = None,
                 max_size_bytes: int = None, wal_mode: bool = True):
        self.db_path = db_path
        self._default_ttl = default_ttl
        self._max_size_bytes = max_size_bytes
        self._lock = threading.RLock()

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

        # Initialize database
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        if wal_mode:
            self._conn.execute('PRAGMA journal_mode=WAL')

        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._create_tables()

    def _create_tables(self) -> None:
        """Create cache tables if they don't exist."""
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                metadata BLOB NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                ttl INTEGER DEFAULT NULL,
                serializer_cls TEXT DEFAULT 'PickleSerializer'
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
        ''')
        self._conn.commit()

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
        with self._lock:
            cursor = self._conn.execute(
                'SELECT data, metadata, created_at, ttl, access_count FROM cache_entries WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None, None

            data_bytes = row['data']
            meta_bytes = row['metadata']
            created_at = row['created_at']
            ttl = row['ttl']
            current_access_count = row['access_count']

            # Check TTL
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None and time.time() - created_at > effective_ttl:
                # Expired - delete and return None
                self._conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                self._conn.commit()
                return None, None

            # Update access stats
            now = time.time()
            self._conn.execute(
                'UPDATE cache_entries SET last_access = ?, access_count = access_count + 1 WHERE key = ?',
                (now, key)
            )
            self._conn.commit()

            # Deserialize
            try:
                metadata = pickle.loads(meta_bytes)
                metadata['last_access'] = now
                metadata['access_count'] = current_access_count + 1

                serializer_cls = metadata.get('serializer_cls', PickleSerializer)
                serializer = serializer_cls()
                value = serializer.deserialize(data_bytes)

                metadata.setdefault('source', self.source_label)
                return metadata, value
            except (pickle.UnpicklingError, KeyError, TypeError, ValueError, EOFError) as e:
                logger.debug("Error deserializing cache entry %s: %s", key, e)
                raise CacheSerializationError(
                    f"Failed to deserialize cache entry '{key}': {e}"
                ) from e

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        if metadata is None:
            metadata = {}

        metadata['key'] = key

        if serializer is None:
            serializer = PickleSerializer()

        # Serialize value
        serialized_value = serializer.serialize(value)
        data_size = len(serialized_value)
        metadata['size'] = data_size

        now = time.time()
        metadata['created_at'] = now
        metadata['last_access'] = now
        metadata['access_count'] = 0

        # Set TTL
        ttl = metadata.get('ttl', self._default_ttl)
        if 'ttl' not in metadata and self._default_ttl is not None:
            metadata['ttl'] = self._default_ttl

        # Inject storage info
        if 'storage' not in metadata:
            metadata['storage'] = [self.source_label]

        meta_bytes = pickle.dumps(metadata)

        with self._lock:
            self._conn.execute('''
                INSERT OR REPLACE INTO cache_entries
                (key, data, metadata, size_bytes, created_at, last_access, access_count, ttl)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ''', (key, serialized_value, meta_bytes, data_size, now, now, ttl))
            self._conn.commit()

            # Check size limit
            if self._max_size_bytes is not None:
                self._check_and_evict()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute('DELETE FROM cache_entries')
            self._conn.commit()
            # VACUUM must run outside a transaction
            try:
                self._conn.execute('VACUUM')
            except sqlite3.OperationalError:
                logger.debug("VACUUM failed after clearing SQLite cache")

    def list_entries(self) -> list[dict]:
        entries = []
        with self._lock:
            cursor = self._conn.execute('SELECT metadata FROM cache_entries')
            for row in cursor:
                try:
                    entries.append(pickle.loads(row['metadata']))
                except (pickle.UnpicklingError, KeyError, TypeError, EOFError):
                    logger.debug("Failed to deserialize SQLite cache metadata")
        return entries

    def cleanup_expired(self, is_expired: Callable[[dict], bool]) -> int:
        """Clean up expired entries using TTL and custom predicate."""
        count = 0
        with self._lock:
            # First, clean up TTL-expired entries
            now = time.time()
            if self._default_ttl is not None:
                cursor = self._conn.execute(
                    'DELETE FROM cache_entries WHERE ttl IS NOT NULL AND (? - created_at) > ttl',
                    (now,)
                )
                count += cursor.rowcount

            # Then check custom predicate
            cursor = self._conn.execute('SELECT key, metadata FROM cache_entries')
            keys_to_delete = []
            for row in cursor:
                try:
                    meta = pickle.loads(row['metadata'])
                    if is_expired(meta):
                        keys_to_delete.append(row['key'])
                except (pickle.UnpicklingError, KeyError, TypeError, EOFError):
                    logger.debug("Failed to deserialize metadata during cleanup for key %s", row['key'])

            for key in keys_to_delete:
                self._conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                count += 1

            if count > 0:
                self._conn.commit()

        return count

    def _check_and_evict(self) -> None:
        """Evict LRU entries if over size limit."""
        if self._max_size_bytes is None:
            return

        cursor = self._conn.execute('SELECT SUM(size_bytes) FROM cache_entries')
        row = cursor.fetchone()
        total_size = row[0] or 0

        if total_size <= self._max_size_bytes:
            return

        # Evict oldest-accessed entries until under 90% of limit
        target = self._max_size_bytes * 0.9
        cursor = self._conn.execute(
            'SELECT key, size_bytes FROM cache_entries ORDER BY last_access ASC'
        )

        evicted = 0
        for row in cursor:
            if total_size <= target:
                break
            self._conn.execute('DELETE FROM cache_entries WHERE key = ?', (row['key'],))
            total_size -= row['size_bytes']
            evicted += 1

        if evicted > 0:
            self._conn.commit()

    def entry_count(self) -> int:
        """Get the number of cache entries."""
        with self._lock:
            cursor = self._conn.execute('SELECT COUNT(*) FROM cache_entries')
            return cursor.fetchone()[0]

    def total_size(self) -> int:
        """Get total data size in bytes."""
        with self._lock:
            cursor = self._conn.execute('SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries')
            return cursor.fetchone()[0]

    def shutdown(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            logger.debug("Error closing SQLite connection")

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        """SQLite handles its own locking via the RLock."""
        return contextlib.nullcontext()
