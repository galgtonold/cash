"""Persistent analytics and cache-usage telemetry.

Collects per-statement metrics (hit/miss, timing) in a local SQLite
database. The notebook statement processor
(``cash.notebook.statement.processor``) writes the events, and the analytics
dashboard (``cash.ui.dashboard``, behind ``cash.show_stats()``) reads them.

The database lives in the per-user cache root (``~/.cache/cash`` on Linux).
Set ``analytics = false`` in the config, or ``CASH_ANALYTICS=0``, to stop
recording.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

from ._location import per_user_cache_root
from .backends.cache_dir import ANALYTICS_DB_FILENAME, is_cash_file
from .tracking.read_classification import register_cache_dir

logger = logging.getLogger(__name__)

# Sanity cap on the telemetry db file. An events log this large is runaway
# growth or a corrupt/garbage file (saw a 2.8 GB SQLITE_NOTADB file that
# made every ``import cash`` print an error). Analytics is best-effort
# observability, so a file over the cap is dropped and recreated rather than
# carried forever.
_MAX_DB_BYTES = 64 * 1024 * 1024  # 64 MiB


def default_db_path() -> Path:
    """Where the analytics db lives unless a path is given: the per-user cache root."""

    return per_user_cache_root() / ANALYTICS_DB_FILENAME


def _write_events(db_path: str, buffer: list[tuple]) -> None:
    """Move every event in *buffer* into the db at *db_path*. Best-effort."""
    if not buffer:
        return
    events = buffer[:]
    buffer.clear()
    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
            conn.executemany(
                """
                INSERT INTO events (session_id, timestamp, status, execution_time, saved_time, code_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                events,
            )
    except sqlite3.Error as exc:
        logger.debug("Failed to flush analytics events: %s", exc)


class AnalyticsManager:
    """
    Manages persistent storage of analytics data for the cash library.
    Uses a local SQLite database to track execution events, savings, and usage patterns.
    """

    def __init__(self, db_path: str | None = None, *, enabled: bool = True):
        """
        Initialize the AnalyticsManager.

        Args:
            db_path: Path to the SQLite database. Defaults to ``analytics.db``
                in the per-user cache root (`default_db_path`).
            enabled: False records nothing and creates no file (the
                ``analytics`` setting).
        """
        self.db_path = str(db_path if db_path is not None else default_db_path())
        self.session_id = str(uuid.uuid4())
        self._event_buffer: list[tuple] = []
        # Events per commit. A commit is an fsync (~12 ms on Windows), and a
        # loop records thousands of statements. What a hard kill can lose is
        # this many telemetry rows, never a cached result.
        self._flush_threshold = 1000
        # True when analytics is switched off, or the db cannot be created even
        # after a recreate (read-only dir, disk full). Analytics then no-ops
        # for the session rather than retrying a doomed connect on every event.
        self._disabled = not enabled
        if enabled:
            try:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.debug("Analytics disabled this session (cannot create %s: %s)", self.db_path, e)
                self._disabled = True
            else:
                # cash's own file: the flush below can run from a finalizer in
                # the middle of a user's statement, and its connect was
                # recorded as a file that statement read.
                register_cache_dir(str(Path(self.db_path).parent), is_cash_file)
                self._init_db()
        # Buffered events are written when the manager is collected or, on a
        # clean interpreter exit, at exit. A hard kill loses at most one
        # buffer of telemetry, never a cached result. The finalizer holds the
        # path and the buffer, not the manager, so it never keeps it alive.
        self._finalizer = weakref.finalize(self, _write_events, self.db_path, self._event_buffer)

    def _init_db(self) -> None:
        """Create the schema, self-healing an unreadable or runaway db.

        Analytics is best-effort observability, never correctness, so
        a pre-existing db that cannot be opened — corrupt pages, a truncated
        write, a non-sqlite or oversized file — must NEVER surface a raw sqlite
        error to the user on every ``import cash`` (which saw a 2.8 GB
        ``SQLITE_NOTADB`` file warn forever). The recovery ladder:

          1. Drop a file over ``_MAX_DB_BYTES`` up front (runaway / bloat).
          2. Try to create the schema.
          3. On any sqlite error (corrupt / not-a-db), unlink and recreate ONCE.
          4. If that still fails, disable analytics silently for the session.

        Every failure logs at ``debug`` — a first-time user must not see an
        alarming error for a telemetry subsystem that changes no cached result.
        """
        # (1) A file over the sanity cap is corruption or runaway growth; drop
        # it rather than carry a multi-GB/garbage file forever.
        with contextlib.suppress(OSError):
            if Path(self.db_path).stat().st_size > _MAX_DB_BYTES:
                logger.debug("Analytics db %s exceeds %d bytes; recreating", self.db_path, _MAX_DB_BYTES)
                Path(self.db_path).unlink()

        try:  # (2)
            self._create_schema()
            return
        except sqlite3.Error as e:  # (3)
            logger.debug("Analytics db at %s is unreadable (%s); recreating", self.db_path, e)
        try:
            with contextlib.suppress(FileNotFoundError):
                Path(self.db_path).unlink()
            self._create_schema()
        except (sqlite3.Error, OSError) as e:  # (4)
            logger.debug("Analytics disabled this session (db init failed: %s)", e)
            self._disabled = True

    def _create_schema(self) -> None:
        """Create the events table + indexes (idempotent). Raises on a db that
        cannot be opened; the caller (:meth:`_init_db`) owns recovery.

        The connection is CLOSED even on failure (``contextlib.closing``): a
        ``sqlite3.connect`` context manager only manages the transaction, not the
        handle, and on Windows a still-open handle to the corrupt file would
        block :meth:`_init_db`'s ``unlink`` recovery."""
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    execution_time REAL,
                    saved_time REAL,
                    code_hash TEXT
                )
            """)
            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON events (session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON events (timestamp)
            """)
            conn.commit()

    def record_event(self, status: str, execution_time: float, saved_time: float = 0.0, code_hash: str | None = None):
        """
        Record an execution event.

        Events are buffered in memory and flushed to SQLite in batches to
        avoid per-statement connection/commit overhead (which can dominate
        wall-clock time in tight loops).

        Args:
            status: 'HIT', 'MISS', or 'EXECUTION'
            execution_time: Time taken for the operation (overhead or actual execution)
            saved_time: Time saved by using cache (for hits)
            code_hash: Optional hash of the code executed
        """
        if self._disabled:
            # No writable db this session; drop telemetry instead of
            # growing an in-memory buffer that can never flush.
            return
        self._event_buffer.append((self.session_id, time.time(), status, execution_time, saved_time, code_hash))
        if len(self._event_buffer) >= self._flush_threshold:
            self.flush()

    def flush(self):
        """Flush buffered events to the SQLite database."""
        _write_events(self.db_path, self._event_buffer)

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for the current session."""
        self.flush()  # Ensure buffered events are persisted before querying
        return self.get_stats_for_session(self.session_id)

    def get_stats_for_session(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a specific session."""
        if self._disabled:
            return {}
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Basic counts
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_events,
                        SUM(CASE WHEN status = 'HIT' THEN 1 ELSE 0 END) as hits,
                        SUM(CASE WHEN status != 'HIT' THEN 1 ELSE 0 END) as misses,
                        SUM(saved_time) as total_saved_time,
                        SUM(execution_time) as total_execution_time
                    FROM events
                    WHERE session_id = ?
                """,
                    (session_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return {}

                stats = dict(row)
                # handle None values from SUM
                stats["total_saved_time"] = stats["total_saved_time"] or 0.0
                stats["total_execution_time"] = stats["total_execution_time"] or 0.0

                return stats
        except sqlite3.Error:
            logger.debug("Failed to query session stats for %s", session_id, exc_info=True)
            return {}

    def get_global_stats(self) -> dict[str, Any]:
        """Get statistics across all recorded sessions."""
        if self._disabled:
            return {}
        self.flush()  # Ensure buffered events are persisted before querying
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT session_id) as total_sessions,
                        COUNT(*) as total_events,
                        SUM(CASE WHEN status = 'HIT' THEN 1 ELSE 0 END) as hits,
                        SUM(CASE WHEN status != 'HIT' THEN 1 ELSE 0 END) as misses,
                        SUM(saved_time) as total_saved_time,
                        SUM(execution_time) as total_execution_time
                    FROM events
                """)

                row = cursor.fetchone()
                if not row:
                    return {}

                stats = dict(row)
                # handle None values from SUM
                stats["total_saved_time"] = stats["total_saved_time"] or 0.0
                stats["total_execution_time"] = stats["total_execution_time"] or 0.0

                return stats
        except sqlite3.Error:
            logger.debug("Failed to query global analytics stats", exc_info=True)
            return {}

    def get_daily_savings(self, limit: int = 7) -> list[tuple[str, float]]:
        """Get total saved time per day for the last 'limit' days."""
        if self._disabled:
            return []
        self.flush()  # Ensure buffered events are persisted before querying
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()

                # SQLite doesn't have a simple date function for unix timestamps in all versions,
                # but 'unixepoch' modifier is available in newer ones.
                # To be safe and compatible, we can use strftime with 'unixepoch' or just process in python if volume is low.
                # Let's try standard SQLite date function.
                cursor.execute(
                    """
                    SELECT
                        date(timestamp, 'unixepoch', 'localtime') as day,
                        SUM(saved_time) as saved
                    FROM events
                    GROUP BY day
                    ORDER BY day DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                return cursor.fetchall()
        except sqlite3.Error:
            logger.debug("Failed to query daily savings", exc_info=True)
            return []
