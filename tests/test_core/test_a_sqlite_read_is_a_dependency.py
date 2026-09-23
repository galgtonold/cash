"""A cached function reading a SQLite database depends on that file.

Found while stress-testing the decorator: ``sqlite3.connect`` opens
the file in C, so nothing passed through a patched reader and a cached query
returned 1 where an uncached run returned 101 after an INSERT -- with no
warning. ``pd.read_sql_query`` over a ``sqlite3`` connection has the same
shape, and looks like the ``pd.read_*`` family that IS tracked.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import os, sqlite3, time, json
    import pandas as pd
    import cash
    cash.configure(cache_dir=CACHE_DIR)
    DB = PATH

    @cash.cache
    def total():
        time.sleep(0.3)
        with sqlite3.connect(DB) as conn:
            return conn.execute("select coalesce(sum(x), 0) from t").fetchone()[0]

    @cash.cache
    def total_pandas():
        time.sleep(0.3)
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
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path)
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert run() == [1, 1]
    conn = sqlite3.connect(db)
    conn.execute("insert into t values (100)")
    conn.commit()
    conn.close()
    assert run() == [101, 101]
