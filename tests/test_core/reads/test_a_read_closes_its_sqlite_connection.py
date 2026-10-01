"""A read of an analytics or sqlite cache database closes its connection.

``with sqlite3.connect(path) as conn`` commits or rolls back, but does not
close: the connection lived until a garbage collection freed it. Until then,
on Windows, its handle kept the file open, and from Python 3.13 freeing it
warns (ResourceWarning: unclosed database) in whatever code runs when the
collector does -- in a test, in whichever one was recording warnings.
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def opened(monkeypatch):
    """Every connection ``sqlite3.connect`` makes during the test."""
    made = []
    connect = sqlite3.connect

    def tracking(*args, **kwargs):
        conn = connect(*args, **kwargs)
        made.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking)
    return made


def _open(connections):
    still_open = []
    for conn in connections:
        try:
            conn.execute("SELECT 1")  # raises once the connection is closed
        except sqlite3.ProgrammingError:
            continue
        still_open.append(conn)
    return still_open


@pytest.mark.parametrize("read", ["get_session_stats", "get_global_stats", "get_daily_savings"])
def test_an_analytics_read_closes_its_connection(read, tmp_path, opened):
    from cash.analytics import AnalyticsManager

    am = AnalyticsManager(db_path=str(tmp_path / "analytics.db"))
    am.record_event("HIT", 0.1, 1.0, "abc")
    del opened[:]
    getattr(am, read)()
    assert opened, "the read made no connection"
    assert not _open(opened)


def test_the_cli_count_of_a_sqlite_cache_closes_its_connection(tmp_path, opened):
    from cash.__main__ import _sqlite_cache
    from cash.backends.sqlite_backend import SQLiteBackend

    cache = tmp_path / "sq"
    db = SQLiteBackend(str(cache / "cache.db"))
    db.set("f:1", 1)
    db.shutdown()
    del opened[:]
    assert _sqlite_cache(str(cache))[0] == 1
    assert opened, "the count made no connection"
    assert not _open(opened)
