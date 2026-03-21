"""Persistent analytics and cache-usage telemetry.

Collects per-statement metrics (hit/miss, timing, storage) in a
local SQLite database for performance analysis and dashboard display.

Used by both the decorator-based caching (``core.py``) and the notebook
caching subsystem (``statement_processor.py``), so it lives at the
package root rather than inside ``notebook/``.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """
    Manages persistent storage of analytics data for the cash library.
    Uses a local SQLite database to track execution events, savings, and usage patterns.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the AnalyticsManager.

        Args:
            db_path: Path to the SQLite database. If None, uses default ~/.cash/analytics.db
        """
        if db_path is None:
            home_dir = Path.home()
            cash_dir = home_dir / ".cash"
            cash_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(cash_dir / "analytics.db")
        else:
            self.db_path = db_path
            # Ensure directory exists if manually specified
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.session_id = str(uuid.uuid4())
        self._event_buffer: list[tuple] = []
        self._flush_threshold = 50  # Flush every 50 events
        self._init_db()

    def __del__(self):
        """Flush remaining buffered events on garbage collection."""
        # Best-effort flush during GC; logging may itself fail during
        # interpreter shutdown, so we intentionally swallow all errors.
        with contextlib.suppress(sqlite3.Error, OSError):
            self.flush()

    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
        except sqlite3.Error as e:
            logger.warning("Failed to initialize analytics database: %s", e)

    def record_event(self,
                    status: str,
                    execution_time: float,
                    saved_time: float = 0.0,
                    code_hash: str | None = None):
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
        self._event_buffer.append((
            self.session_id,
            time.time(),
            status,
            execution_time,
            saved_time,
            code_hash
        ))
        if len(self._event_buffer) >= self._flush_threshold:
            self.flush()

    def flush(self):
        """Flush buffered events to the SQLite database."""
        if not self._event_buffer:
            return
        events = self._event_buffer[:]
        self._event_buffer.clear()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO events (session_id, timestamp, status, execution_time, saved_time, code_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, events)
                conn.commit()
        except sqlite3.Error as exc:
            # analytics should be best-effort
            logger.debug("Failed to flush analytics events: %s", exc)

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for the current session."""
        self.flush()  # Ensure buffered events are persisted before querying
        return self.get_stats_for_session(self.session_id)

    def get_stats_for_session(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a specific session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Basic counts
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_events,
                        SUM(CASE WHEN status = 'HIT' THEN 1 ELSE 0 END) as hits,
                        SUM(CASE WHEN status != 'HIT' THEN 1 ELSE 0 END) as misses,
                        SUM(saved_time) as total_saved_time,
                        SUM(execution_time) as total_execution_time
                    FROM events
                    WHERE session_id = ?
                """, (session_id,))

                row = cursor.fetchone()
                if not row:
                    return {}

                stats = dict(row)
                # handle None values from SUM
                stats['total_saved_time'] = stats['total_saved_time'] or 0.0
                stats['total_execution_time'] = stats['total_execution_time'] or 0.0

                return stats
        except sqlite3.Error:
            logger.debug("Failed to query session stats for %s", session_id, exc_info=True)
            return {}

    def get_global_stats(self) -> dict[str, Any]:
        """Get statistics across all recorded sessions."""
        self.flush()  # Ensure buffered events are persisted before querying
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                stats['total_saved_time'] = stats['total_saved_time'] or 0.0
                stats['total_execution_time'] = stats['total_execution_time'] or 0.0

                return stats
        except sqlite3.Error:
            logger.debug("Failed to query global analytics stats", exc_info=True)
            return {}

    def get_daily_savings(self, limit: int = 7) -> list[tuple[str, float]]:
        """Get total saved time per day for the last 'limit' days."""
        self.flush()  # Ensure buffered events are persisted before querying
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # SQLite doesn't have a simple date function for unix timestamps in all versions,
                # but 'unixepoch' modifier is available in newer ones.
                # To be safe and compatible, we can use strftime with 'unixepoch' or just process in python if volume is low.
                # Let's try standard SQLite date function.
                cursor.execute("""
                    SELECT
                        date(timestamp, 'unixepoch', 'localtime') as day,
                        SUM(saved_time) as saved
                    FROM events
                    GROUP BY day
                    ORDER BY day DESC
                    LIMIT ?
                """, (limit,))

                return cursor.fetchall()
        except sqlite3.Error:
            logger.debug("Failed to query daily savings", exc_info=True)
            return []
