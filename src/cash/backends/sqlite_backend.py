"""SQLite-based cache backend for efficient storage of many small entries."""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

from ._base import CacheBackend, MetadataDict, entry_expired
from ._writes import PendingWrites
from .cache_dir import warn_unusable
from .serialization import RESTORE_ERRORS, PickleSerializer, Serializer, restore_value

logger = logging.getLogger(__name__)

__all__ = ["SQLiteBackend"]


class SQLiteBackend(CacheBackend):
    """All entries in one SQLite database file; suits many small entries.

    Entries are pickled, as with `FileBackend`. As a tier, it does not take
    values over 100 MiB. The database is opened on first use; one that cannot
    be opened turns the backend off for the run, with a warning.

    The simpler of the two disk backends: it evicts the least recently used
    entries rather than ranking them by value per byte, gives no notice when
    its cap evicts something, and a running process does not notice the
    database being cleared under it.

    Args:
        db_path: The database file; its directory is created if missing.
        default_ttl: Seconds an entry stays valid when the caller gives no
            ``ttl``. ``None``: no expiry.
        max_size_bytes: Byte cap for stored data. ``None``: no cap.
        wal_mode: Use SQLite's WAL journal mode, so readers do not block
            the writer.
    """

    source_label: str = "SQLITE"
    # Tier-promotion hint: SQLite's row/blob handling degrades on
    # multi-hundred-MB values, so the tiered pipeline skips it past
    # this cap. Bare-backend writes are not gated.
    max_size_bytes: int | None = 100 * 1024 * 1024

    def __init__(
        self,
        db_path: str = ".cash/cache.db",
        default_ttl: int = None,
        max_size_bytes: int = None,
        wal_mode: bool = True,
    ):
        self.db_path = db_path
        self._default_ttl = default_ttl
        self._max_size_bytes = max_size_bytes
        self._lock = threading.RLock()

        # Per-backend async writes: serialization happens on the calling
        # thread, the actual INSERT runs in this executor.
        self._writes = PendingWrites()

        self._wal_mode = wal_mode
        #: Opened on first use (`_open`), so building the backend does no I/O.
        self._connection: sqlite3.Connection | None = None
        #: Set when the database turned out unusable. Every operation then
        #: answers as an empty cache would: a miss, a no-op write.
        self._unusable = False

    def _open(self) -> sqlite3.Connection | None:
        """The connection, opened on first use, or None when the database
        cannot be opened.

        A database that cannot be created (a read-only volume, a cache_dir
        under a regular file) turns this tier off, with one warning, instead
        of failing the caller's program, as the file tier does.
        """
        conn = self._connection
        if conn is not None or self._unusable:
            return conn
        with self._lock:
            if self._connection is None and not self._unusable:
                try:
                    os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    if self._wal_mode:
                        conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    self._create_tables(conn)
                    self._connection = conn
                except (OSError, sqlite3.Error) as exc:
                    self._unusable = True
                    warn_unusable(self.db_path, exc)
            return self._connection

    @property
    def _conn(self) -> sqlite3.Connection | None:
        return self._open()

    #: Column order is load-bearing: SQLite lays a row out in declaration
    #: order and spills what does not fit onto a chain of overflow pages, so
    #: reading a column means walking past everything declared before it. With
    #: ``data`` first, ``SELECT metadata`` on a 16MB entry walked 16MB:
    #: measured at 7.398ms against 0.003ms with the two swapped -- 2845x, for a
    #: change that moves no bytes. Changing the table means bumping
    #: `SCHEMA_VERSION`.
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS cache_entries (
            key TEXT PRIMARY KEY,
            metadata BLOB NOT NULL,
            size_bytes INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            last_access REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            ttl INTEGER DEFAULT NULL,
            serializer_cls TEXT DEFAULT 'PickleSerializer',
            data BLOB NOT NULL
        )
    """

    #: Version of `_SCHEMA`, kept in the database's ``user_version``. A
    #: database stamped with any other version has its table dropped: this is
    #: a cache, so every entry can be recomputed, and copying a table of
    #: large values forward would cost more than recomputing them.
    SCHEMA_VERSION = 2

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create the cache table, dropping one written in another schema."""
        (stored,) = conn.execute("PRAGMA user_version").fetchone()
        if stored != self.SCHEMA_VERSION:
            conn.execute("DROP TABLE IF EXISTS cache_entries")
            conn.execute(f"PRAGMA user_version = {int(self.SCHEMA_VERSION)}")
        conn.execute(self._SCHEMA)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
        """)
        conn.commit()

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        # Wait for any pending write for this key so we never miss it.
        self._writes.wait(key)
        conn = self._open()
        if conn is None:
            return None, None
        with self._lock:
            cursor = conn.execute("SELECT data, metadata, access_count FROM cache_entries WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row is None:
                return None, None

            try:
                metadata = pickle.loads(row["metadata"])
                expired = entry_expired(metadata, self._default_ttl)
                value = None if expired else restore_value(metadata, row["data"])
            except RESTORE_ERRORS as e:
                # Unrestorable, so absent; dropped, so the recomputed value
                # replaces it instead of failing every later read the same way.
                logger.debug("Unrestorable cache entry %s, dropped: %s", key, e)
                expired = True
            if expired:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                return None, None

            now = time.time()
            conn.execute(
                "UPDATE cache_entries SET last_access = ?, access_count = access_count + 1 WHERE key = ?", (now, key)
            )
            conn.commit()
            metadata["last_access"] = now
            metadata["access_count"] = row["access_count"] + 1
            metadata.setdefault("source", self.source_label)
            return metadata, value

    def get_metadata(self, key: str) -> MetadataDict | None:
        """Read an entry's metadata without touching its value.

        The base implementation performs a full ``get()`` and discards the
        value, which means answering "when was this last accessed?" unpickles
        the whole cached object. Measured at 35.9ms for a 16MB entry against
        0.226ms for the file backend, whose metadata read is bounded by the
        metadata.

        Selecting one column instead makes the cost independent of the value,
        which is what every caller of this method already assumes: it exists
        so badges and listings can inspect entries they have no intention of
        restoring.
        """
        self._writes.wait(key)
        conn = self._open()
        if conn is None:
            return None
        with self._lock:
            cursor = conn.execute("SELECT metadata FROM cache_entries WHERE key = ?", (key,))
            row = cursor.fetchone()

        if row is None:
            return None

        try:
            metadata = pickle.loads(row["metadata"])
        except RESTORE_ERRORS as e:
            logger.debug("Error deserializing metadata for %s: %s", key, e)
            return None
        if entry_expired(metadata, self._default_ttl):
            return None

        metadata.setdefault("source", self.source_label)
        return metadata

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> None:
        """Serialize on the calling thread, INSERT in the background."""
        if self._open() is None:
            return
        # The entry keeps the ``created_at`` it was first written with: a copy
        # promoted from another tier must not restart its ttl.
        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        # IMPORTANT: serialize on the calling thread.
        serialized_value = serializer.serialize(value)
        data_size = len(serialized_value)
        metadata["size"] = data_size

        created_at = metadata["created_at"]

        # Set TTL
        ttl = metadata.get("ttl", self._default_ttl)
        if "ttl" not in metadata and self._default_ttl is not None:
            metadata["ttl"] = self._default_ttl

        # Inject storage info
        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        meta_bytes = pickle.dumps(metadata)

        self._writes.submit(
            key,
            self._do_set_sync,
            key,
            serialized_value,
            meta_bytes,
            data_size,
            created_at,
            ttl,
        )

    def _do_set_sync(
        self, key: str, serialized_value: bytes, meta_bytes: bytes, data_size: int, created_at: float, ttl: int | None
    ) -> None:
        """The actual INSERT — runs in the PendingWrites worker thread."""
        conn = self._open()
        if conn is None:
            return
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, data, metadata, size_bytes, created_at, last_access, access_count, ttl)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
                (key, serialized_value, meta_bytes, data_size, created_at, time.time(), ttl),
            )
            conn.commit()

            # Check size limit
            if self._max_size_bytes is not None:
                self._check_and_evict()

    def delete(self, key: str) -> None:
        # Drain any pending write so the delete actually deletes.
        self._writes.drain(key)
        conn = self._open()
        if conn is None:
            return
        with self._lock:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        self._writes.wait_all()
        conn = self._open()
        if conn is None:
            return
        with self._lock:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
            # VACUUM must run outside a transaction
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError:
                logger.debug("VACUUM failed after clearing SQLite cache")

    def list_entries(self) -> list[dict]:
        self._writes.wait_all()
        entries = []
        conn = self._open()
        if conn is None:
            return entries
        with self._lock:
            cursor = conn.execute("SELECT metadata FROM cache_entries")
            for row in cursor:
                try:
                    entries.append(pickle.loads(row["metadata"]))
                except RESTORE_ERRORS:
                    logger.debug("Failed to deserialize SQLite cache metadata")
        return entries

    def cleanup_expired(self, is_expired: Callable[[dict], bool]) -> int:
        """Clean up expired entries using TTL and custom predicate."""
        count = 0
        conn = self._open()
        if conn is None:
            return count
        with self._lock:
            # This tier's own ttl, as a read applies it, then the caller's predicate.
            now = time.time()
            cursor = conn.execute("SELECT key, metadata FROM cache_entries")
            keys_to_delete = []
            for row in cursor:
                try:
                    meta = pickle.loads(row["metadata"])
                    if entry_expired(meta, self._default_ttl, now) or is_expired(meta):
                        keys_to_delete.append(row["key"])
                except RESTORE_ERRORS:
                    logger.debug("Failed to deserialize metadata during cleanup for key %s", row["key"])

            for key in keys_to_delete:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                count += 1

            if count > 0:
                conn.commit()

        return count

    def _check_and_evict(self) -> None:
        """Evict LRU entries if over size limit."""
        if self._max_size_bytes is None:
            return

        cursor = self._connection.execute("SELECT SUM(size_bytes) FROM cache_entries")
        row = cursor.fetchone()
        total_size = row[0] or 0

        if total_size <= self._max_size_bytes:
            return

        # Evict oldest-accessed entries until under 90% of limit
        target = self._max_size_bytes * 0.9
        cursor = self._connection.execute("SELECT key, size_bytes FROM cache_entries ORDER BY last_access ASC")

        evicted = 0
        for row in cursor:
            if total_size <= target:
                break
            self._connection.execute("DELETE FROM cache_entries WHERE key = ?", (row["key"],))
            total_size -= row["size_bytes"]
            evicted += 1

        if evicted > 0:
            self._connection.commit()

    def entry_count(self) -> int:
        """Get the number of cache entries."""
        # Bulk read: drain in-flight writes first so the count reflects
        # every set() that has already returned (same barrier as
        # list_entries / clear / cleanup_expired).
        self._writes.wait_all()
        conn = self._open()
        if conn is None:
            return 0
        with self._lock:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            return cursor.fetchone()[0]

    def promotion_size_cap(self) -> int | None:
        """The smaller of the class-level hint and this tier's own size cap:
        a value larger than the cap would be written only to be evicted."""
        hint = type(self).max_size_bytes
        own = self._max_size_bytes
        if own is None or hint is None:
            return hint if own is None else own
        return min(hint, own)

    def shutdown(self) -> None:
        """Wait for pending writes, then close the database connection."""
        self._writes.shutdown(wait=True)
        if self._connection is None:
            return
        try:
            self._connection.close()
        except sqlite3.Error:
            logger.debug("Error closing SQLite connection")

    # NOTE: no ``lock()`` override. The internal ``self._lock`` RLock guards
    # storage integrity (concurrent DB writes), which is a distinct concern
    # from compute single-flight. We inherit ``CacheBackend.lock()`` so that
    # ``use_locking=True`` gets a real in-process per-key lock.
