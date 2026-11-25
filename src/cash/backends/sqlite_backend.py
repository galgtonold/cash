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

from . import cache_dir as _cache_dir
from ..exceptions import CacheBackendError
from ._base import CacheBackend, MetadataDict, entry_expired
from ._writes import PendingWrites
from .cache_dir import NewerCacheFormat, warn_newer_format, warn_unusable
from .serialization import RESTORE_ERRORS, PickleSerializer, Serializer, restore_value

logger = logging.getLogger(__name__)

__all__ = ["SQLiteBackend"]


class SQLiteBackend(CacheBackend):
    """All entries in one SQLite database file; suits many small entries.

    Entries are pickled, as with `FileBackend`. As a tier, it does not take
    values over 100 MiB. The database is opened on first use; one that cannot
    be opened turns the backend off for the run, with a warning.

    Several processes can share the database. A read never waits for another
    process's write, a write waits its turn (`BUSY_TIMEOUT_S`, tried
    `WRITE_ATTEMPTS` times), and a database that stays busy costs a recompute,
    never an exception in the caller: a read that fails is a miss, the
    access-time note of a hit is kept for the next write, and a store that
    fails is reported like any failed store (STORE-FAILED).

    The simpler of the two disk backends: it evicts the least recently used
    entries rather than ranking them by value per byte, gives no notice when
    its cap evicts something, and a running process does not notice the
    database being cleared under it. The database is stamped with the cache
    format like a cache folder: one written in an older format is emptied on
    open, one written by a newer cash is left alone and the tier is off.

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
        #: Hits whose access time is not in the database yet: key -> (when,
        #: how many). Another process held the write lock when they happened;
        #: the next write this process gets through carries them.
        self._unwritten_hits: dict[str, tuple[float, int]] = {}

    #: Seconds a write waits for another process's write to finish, per attempt.
    BUSY_TIMEOUT_S = 5.0
    #: Times a write is tried before it counts as failed.
    WRITE_ATTEMPTS = 3
    #: Milliseconds a hit waits to note its access time before leaving the
    #: note for a later write: a hit must not queue behind other processes'
    #: stores for bookkeeping.
    HIT_NOTE_TIMEOUT_MS = 20

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
                    # Autocommit: every write below opens its own ``BEGIN
                    # IMMEDIATE`` (`_writing`), which takes the write lock
                    # first and so waits for a busy database. A transaction
                    # that reads first and writes later fails at once when
                    # another process committed in between, timeout or not.
                    conn = sqlite3.connect(
                        self.db_path, check_same_thread=False, timeout=self.BUSY_TIMEOUT_S, isolation_level=None
                    )
                    conn.row_factory = sqlite3.Row
                    if self._wal_mode:
                        try:
                            conn.execute("PRAGMA journal_mode=WAL")
                        except sqlite3.OperationalError:
                            # Another process is in the database right now,
                            # or the volume cannot do WAL: whatever mode the
                            # database is in works, only slower.
                            logger.debug("Could not switch %s to WAL", self.db_path, exc_info=True)
                    conn.execute("PRAGMA synchronous=NORMAL")
                    self._retrying(lambda: self._create_tables(conn))
                    self._connection = conn
                except NewerCacheFormat as newer:
                    # A newer cash's database: off for this run, left as it is.
                    self._unusable = True
                    conn.close()
                    warn_newer_format(self.db_path, newer.stored)
                except (OSError, sqlite3.Error) as exc:
                    self._unusable = True
                    warn_unusable(self.db_path, exc)
            return self._connection

    @property
    def _conn(self) -> sqlite3.Connection | None:
        return self._open()

    #: Column order is load-bearing: SQLite lays a row out in declaration
    #: order and spills what does not fit onto a chain of overflow pages, so
    #: reading a column means walking past everything declared before it.
    #: ``data`` goes last, so ``SELECT metadata`` never walks a large value.
    #: Changing the table means bumping `SCHEMA_VERSION`.
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

    #: Version of `_SCHEMA`. Changing the table means bumping it.
    SCHEMA_VERSION = 2

    #: The database's ``user_version`` is ``SCHEMA_VERSION * _STAMP_STRIDE +
    #: CACHE_FORMAT_VERSION``: the table layout and the format of what the
    #: rows hold, in one number that grows with either.
    _STAMP_STRIDE = 1000

    @classmethod
    def _stamp(cls) -> int:
        return cls.SCHEMA_VERSION * cls._STAMP_STRIDE + _cache_dir.CACHE_FORMAT_VERSION

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create the cache table, dropping one written in an older schema
        or cache format.

        The same rule as the file tier's stamp (`CacheDirStamp.check`). A
        lower stamp (one from before the stamp carried the cache format
        included) has its table dropped: this is a cache, so every entry can
        be recomputed, entries of an older format would be served wrongly,
        and copying a table of large values forward would cost more than
        recomputing them. A higher stamp is a newer cash's database, which
        is left as it is (`NewerCacheFormat`).
        """
        current = self._stamp()
        if conn.execute("PRAGMA user_version").fetchall()[0][0] == current:
            return  # stamped with its table, in one transaction, by an earlier open
        # One transaction, so two processes opening an old database at once
        # do not each drop the table the other has begun to fill.
        with self._writing(conn):
            (stored,) = conn.execute("PRAGMA user_version").fetchall()[0]
            if stored > current:
                raise NewerCacheFormat(stored % self._STAMP_STRIDE)
            if stored != current:
                conn.execute("DROP TABLE IF EXISTS cache_entries")
                conn.execute(f"PRAGMA user_version = {int(current)}")
            conn.execute(self._SCHEMA)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)

    @contextlib.contextmanager
    def _writing(self, conn: sqlite3.Connection, wait_ms: int | None = None):
        """One write transaction: ``BEGIN IMMEDIATE`` ... ``COMMIT``.

        Taking the write lock first is what lets SQLite wait for another
        process (the busy timeout, or *wait_ms* for this one transaction).
        Raises ``sqlite3.OperationalError`` when the database stayed busy;
        nothing was written then.
        """
        if wait_ms is not None:
            conn.execute(f"PRAGMA busy_timeout = {int(wait_ms)}")
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise
        finally:
            if wait_ms is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout = {int(self.BUSY_TIMEOUT_S * 1000)}")

    def _retrying(self, write: Callable[[], Any]) -> Any:
        """Run *write*, again while the database is busy, `WRITE_ATTEMPTS` times."""
        for attempt in range(self.WRITE_ATTEMPTS):
            try:
                return write()
            except sqlite3.OperationalError as exc:
                if attempt + 1 == self.WRITE_ATTEMPTS or not _is_busy(exc):
                    raise
                logger.debug("SQLite cache %s is busy, trying the write again", self.db_path)
        return None

    def _read(self, conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list | None:
        """Every row of a query, or None when the database could not be read.

        All rows at once: a cursor left part-read keeps its read transaction
        open on the connection, and the next write on it then fails at once
        if another process committed meanwhile. A failed read is logged and
        answered as "nothing there": the caller recomputes.
        """
        try:
            return conn.execute(sql, args).fetchall()
        except sqlite3.Error as exc:
            logger.debug("SQLite cache %s could not be read (%s); a miss", self.db_path, exc)
            return None

    def _note_hit(self, conn: sqlite3.Connection, key: str, now: float) -> None:
        """Record a hit's access time, now if the database is free, else
        with the next write (`_unwritten_hits`). Never raises: the value was
        read, and when it was last used only ranks it for eviction."""
        when, count = self._unwritten_hits.get(key, (now, 0))
        self._unwritten_hits[key] = (max(when, now), count + 1)
        try:
            with self._writing(conn, wait_ms=self.HIT_NOTE_TIMEOUT_MS):
                self._write_hits(conn)
        except sqlite3.Error as exc:
            logger.debug("Access time of %r not written yet (%s)", key, exc)
            if len(self._unwritten_hits) > self._MAX_UNWRITTEN_HITS:
                self._unwritten_hits.clear()

    #: Most hits kept for a later write; past it the notes are dropped.
    _MAX_UNWRITTEN_HITS = 10_000

    def _write_hits(self, conn: sqlite3.Connection) -> None:
        """Write the kept access times, inside the caller's transaction."""
        hits = self._unwritten_hits
        if not hits:
            return
        conn.executemany(
            "UPDATE cache_entries SET last_access = MAX(last_access, ?), access_count = access_count + ? WHERE key = ?",
            [(when, count, key) for key, (when, count) in hits.items()],
        )
        hits.clear()

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        # Wait for any pending write for this key so we never miss it.
        self._writes.wait(key)
        conn = self._open()
        if conn is None:
            return None, None
        with self._lock:
            rows = self._read(conn, "SELECT data, metadata, access_count FROM cache_entries WHERE key = ?", (key,))
            if not rows:
                return None, None
            row = rows[0]

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
                try:
                    with self._writing(conn):
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                except sqlite3.Error as exc:
                    # Still a miss; the recomputed value replaces the row.
                    logger.debug("Could not drop the unusable entry %s: %s", key, exc)
                return None, None

            now = time.time()
            unwritten = self._unwritten_hits.get(key, (now, 0))[1]
            self._note_hit(conn, key, now)
            metadata["last_access"] = now
            metadata["access_count"] = row["access_count"] + unwritten + 1
            metadata.setdefault("source", self.source_label)
            return metadata, value

    def get_metadata(self, key: str) -> MetadataDict | None:
        """Read an entry's metadata without touching its value.

        Not the base ``get()``, which unpickles the whole cached object to
        answer "when was this last accessed?". Selecting one column makes the
        cost independent of the value,
        which is what every caller of this method already assumes: it exists
        so badges and listings can inspect entries they have no intention of
        restoring.
        """
        self._writes.wait(key)
        conn = self._open()
        if conn is None:
            return None
        with self._lock:
            rows = self._read(conn, "SELECT metadata FROM cache_entries WHERE key = ?", (key,))

        if not rows:
            return None
        row = rows[0]

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

        self._writes.submit_sized(
            key,
            len(serialized_value),
            True,
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
        """The actual INSERT — runs in the PendingWrites worker thread.

        A database another process keeps busy past every attempt raises, and
        the write queue reports it as a failed store (STORE-FAILED): the call
        already returned its result, and the next one computes again.
        """
        conn = self._open()
        if conn is None:
            return

        def write() -> None:
            with self._lock, self._writing(conn):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, data, metadata, size_bytes, created_at, last_access, access_count, ttl)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                    (key, serialized_value, meta_bytes, data_size, created_at, time.time(), ttl),
                )
                self._unwritten_hits.pop(key, None)
                self._write_hits(conn)
                if self._max_size_bytes is not None:
                    self._check_and_evict(conn)

        self._retrying(write)

    def _modify(self, what: str, sql: str, args: tuple = ()) -> None:
        """One statement that changes the database, for a method the user
        calls to remove entries: a database that stays busy is an error they
        must see, or the entries they removed would still be served."""
        conn = self._open()
        if conn is None:
            return

        def write() -> None:
            with self._lock, self._writing(conn):
                conn.execute(sql, args)

        try:
            self._retrying(write)
        except sqlite3.Error as exc:
            raise CacheBackendError(f"SQLite {what} failed for {self.db_path}: {exc}") from exc

    def delete(self, key: str) -> None:
        # Drain any pending write so the delete actually deletes.
        self._writes.drain(key)
        with self._lock:
            self._unwritten_hits.pop(key, None)
        self._modify(f"delete of {key!r}", "DELETE FROM cache_entries WHERE key = ?", (key,))

    def clear(self) -> None:
        self._writes.wait_all()
        conn = self._open()
        if conn is None:
            return
        with self._lock:
            self._unwritten_hits.clear()
        self._modify("clear", "DELETE FROM cache_entries")
        with self._lock:
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
            rows = self._read(conn, "SELECT metadata FROM cache_entries") or ()
        for row in rows:
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
            keys_to_delete = []
            for row in self._read(conn, "SELECT key, metadata FROM cache_entries") or ():
                try:
                    meta = pickle.loads(row["metadata"])
                    if entry_expired(meta, self._default_ttl, now) or is_expired(meta):
                        keys_to_delete.append(row["key"])
                except RESTORE_ERRORS:
                    logger.debug("Failed to deserialize metadata during cleanup for key %s", row["key"])

            if not keys_to_delete:
                return 0

            def write() -> None:
                with self._writing(conn):
                    conn.executemany("DELETE FROM cache_entries WHERE key = ?", [(k,) for k in keys_to_delete])

            try:
                self._retrying(write)
            except sqlite3.Error as exc:
                # Housekeeping: an expired entry is not served either way.
                logger.debug("Could not remove expired entries from %s: %s", self.db_path, exc)
                return 0
            count = len(keys_to_delete)

        return count

    def _check_and_evict(self, conn: sqlite3.Connection) -> None:
        """Evict LRU entries if over size limit, inside the caller's transaction."""
        if self._max_size_bytes is None:
            return

        total_size = conn.execute("SELECT SUM(size_bytes) FROM cache_entries").fetchall()[0][0] or 0
        if total_size <= self._max_size_bytes:
            return

        # Evict oldest-accessed entries until under 90% of limit
        target = self._max_size_bytes * 0.9
        evict = []
        for row in conn.execute("SELECT key, size_bytes FROM cache_entries ORDER BY last_access ASC").fetchall():
            if total_size <= target:
                break
            evict.append((row["key"],))
            total_size -= row["size_bytes"]
        conn.executemany("DELETE FROM cache_entries WHERE key = ?", evict)

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
            rows = self._read(conn, "SELECT COUNT(*) FROM cache_entries")
            return rows[0][0] if rows else 0

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
            with self._lock:
                if self._unwritten_hits:
                    # The last chance for the access times of this run's hits.
                    with contextlib.suppress(sqlite3.Error), self._writing(self._connection, wait_ms=200):
                        self._write_hits(self._connection)
            self._connection.close()
        except sqlite3.Error:
            logger.debug("Error closing SQLite connection")

    # NOTE: no ``lock()`` override. The internal ``self._lock`` RLock guards
    # storage integrity (concurrent DB writes), which is a distinct concern
    # from compute single-flight. We inherit ``CacheBackend.lock()`` so that
    # ``use_locking=True`` gets a real in-process per-key lock.


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    """Did *exc* say that another connection holds the database?"""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    text = str(exc)
    return "locked" in text or "busy" in text
