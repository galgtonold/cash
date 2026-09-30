"""A SQLite database in WAL mode is a dependency together with its ``-wal``.

A commit in WAL mode goes to ``<db>-wal``; the main file is untouched until a
checkpoint. With a writer holding its connection open, as an app server does,
a cached ``select sum(x)`` returned 1 after ``insert 100`` where an uncached
run returned 101.
"""

from __future__ import annotations

import sqlite3

from cash import Cash


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _wal_db(path):
    writer = sqlite3.connect(path)
    writer.execute("pragma journal_mode=wal")
    writer.execute("create table t(x)")
    writer.execute("insert into t values (1)")
    writer.commit()
    return writer


def test_a_commit_that_sits_in_the_wal_file_recomputes(tmp_path):
    c = _cash(tmp_path)
    db = str(tmp_path / "w.db")
    writer = _wal_db(db)
    runs = []

    @c.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        con = sqlite3.connect(p)
        try:
            return con.execute("select sum(x) from t").fetchone()[0]
        finally:
            con.close()

    try:
        assert total(db) == 1
        assert total(db) == 1
        assert len(runs) == 1
        writer.execute("insert into t values (100)")
        writer.commit()
        assert total(db) == 101
    finally:
        writer.close()


def test_a_wal_database_read_alone_still_hits(tmp_path):
    """Control: the reader's own connection makes and removes the ``-wal``."""
    c = _cash(tmp_path)
    db = str(tmp_path / "w.db")
    _wal_db(db).close()
    runs = []

    @c.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        con = sqlite3.connect(p)
        try:
            return con.execute("select sum(x) from t").fetchone()[0]
        finally:
            con.close()

    assert [total(db) for _ in range(3)] == [1, 1, 1]
    assert len(runs) == 1


def test_a_rollback_journal_database_still_hits(tmp_path):
    """Control: no ``-wal`` is looked for in the default journal mode."""
    c = _cash(tmp_path)
    db = str(tmp_path / "r.db")
    con = sqlite3.connect(db)
    con.execute("create table t(x)")
    con.execute("insert into t values (1)")
    con.commit()
    con.close()
    runs = []

    @c.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        con = sqlite3.connect(p)
        try:
            return con.execute("select sum(x) from t").fetchone()[0]
        finally:
            con.close()

    assert [total(db) for _ in range(3)] == [1, 1, 1]
    assert len(runs) == 1
