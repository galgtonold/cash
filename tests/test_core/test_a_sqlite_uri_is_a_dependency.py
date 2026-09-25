"""A SQLite database opened by ``file:`` URI is a dependency of the call.

``sqlite3.connect("file:d.db?mode=ro", uri=True)`` -- the read-only spelling
-- recorded the URI as if it were a path. No such file exists, so the snapshot
dropped it: after an INSERT the cached query kept returning the old sum.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from cash import Cash
from cash.tracking.reader_patches import _sqlite_uri_path


def _db(path, value):
    conn = sqlite3.connect(path)
    conn.execute("create table if not exists t (x int)")
    conn.execute("insert into t values (?)", (value,))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("spelling", ["relative", "absolute", "positional"])
def test_an_insert_invalidates_a_query_over_a_uri_connection(tmp_path, monkeypatch, spelling):
    monkeypatch.chdir(tmp_path)
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    _db(tmp_path / "d.db", 1)
    uri = "file:d.db?mode=ro" if spelling != "absolute" else f"file:{tmp_path / 'd.db'}?mode=ro"
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def total():
        runs.append(1)
        if spelling == "positional":
            conn = sqlite3.connect(uri, 5.0, 0, "DEFERRED", True, sqlite3.Connection, 128, True)
        else:
            conn = sqlite3.connect(uri, uri=True)
        try:
            return conn.execute("select sum(x) from t").fetchone()[0]
        finally:
            conn.close()

    assert total() == 1
    assert total() == 1
    assert len(runs) == 1
    _db(tmp_path / "d.db", 100)
    assert total() == 101


@pytest.mark.parametrize(
    ("uri", "path"),
    [
        ("file:d.db?mode=ro", "d.db"),
        ("file:/data/d%20b.db", "/data/d b.db"),
        ("file://localhost/data/d.db", "/data/d.db"),
        ("file::memory:", None),
        ("file:x.db?mode=memory&cache=shared", None),
        ("file://elsewhere/d.db", None),
        ("plain.db", "plain.db"),
    ],
)
def test_the_uri_names_the_file(uri, path):
    if os.name == "nt" and path is not None and path.startswith("/"):
        pytest.skip("POSIX absolute paths")
    assert _sqlite_uri_path(uri) == path
