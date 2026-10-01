"""A cached function reading a SQLite database depends on that file.

``sqlite3.connect`` opens the file in C, so no patched reader sees it; the
connection itself records the file, whether it is named by path or by a
``file:`` URI (the read-only spelling). An INSERT then invalidates the cached
query, for a plain ``sqlite3`` cursor and for ``pd.read_sql_query`` over the
same connection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import textwrap

import pytest

from cash import Cash
from cash.tracking.reader_patches import _sqlite_uri_path
from tests._scripts import run_python

PROGRAM = textwrap.dedent("""
    import os, sqlite3, time, json
    import pandas as pd
    import cash
    cash.configure(cache_dir=CACHE_DIR)
    DB = PATH

    @cash.cache
    def total():
        with sqlite3.connect(DB) as conn:
            return conn.execute("select coalesce(sum(x), 0) from t").fetchone()[0]

    @cash.cache
    def total_pandas():
        with sqlite3.connect(DB) as conn:
            return int(pd.read_sql_query("select coalesce(sum(x), 0) as s from t", conn)["s"][0])

    print(json.dumps([total(), total_pandas()]))
""")


@pytest.mark.timeout(300)
def test_an_insert_invalidates_the_cached_query(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("create table t (x int)")
    conn.execute("insert into t values (1)")
    conn.commit()
    conn.close()

    script = tmp_path / "run.py"
    script.write_text(
        PROGRAM.replace("CACHE_DIR", repr(str(tmp_path / ".cash"))).replace("PATH", repr(str(db))), encoding="utf-8"
    )

    def run():
        done = run_python(script, cwd=tmp_path, timeout=180)
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert run() == [1, 1]
    conn = sqlite3.connect(db)
    conn.execute("insert into t values (100)")
    conn.commit()
    conn.close()
    assert run() == [101, 101]


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
