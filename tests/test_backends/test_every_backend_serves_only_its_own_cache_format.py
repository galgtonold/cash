"""A cache format change must reach every backend, not only the cache folder.

The format number changes when entries written before would be served
wrongly by the code after. Only the file tier checked it: a SQLite database
kept serving its old rows after an upgrade, and a Redis server or an S3
prefix (the team caches, where teammates on different versions meet all the
time) served every format to everyone.

* SQLite carries the format in its database stamp: an older one is emptied on
  open, a newer one is left alone and the tier is off, as for a cache folder.
* Redis and S3 carry it in each entry's name, so the formats live side by
  side: a build reads, lists and replaces only its own.
"""

from __future__ import annotations

import sqlite3
import textwrap
import warnings
from unittest.mock import patch

import pytest

from cash.backends import cache_dir
from cash.backends.cache_dir import CACHE_FORMAT_VERSION
from cash.backends.sqlite_backend import SQLiteBackend
from tests._scripts import run_python

from .remote_doubles import FakeRedisClient, FakeS3Client


@pytest.fixture
def as_format(monkeypatch):
    """Run what follows as a cash whose cache format is *version*."""

    def switch(version: int) -> None:
        monkeypatch.setattr(cache_dir, "CACHE_FORMAT_VERSION", version)

    return switch


def _stored(backend, key="k", value=None):
    backend.set(key, {"v": 1} if value is None else value, {"execution_time": 1.0})
    backend._writes.wait_all()


# -- SQLite ------------------------------------------------------------------


def _rows(db: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
    finally:
        conn.close()


def _user_version(db: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_sqlite_drops_the_entries_of_an_older_format(tmp_path, as_format):
    db = str(tmp_path / "c.db")
    as_format(CACHE_FORMAT_VERSION - 1)
    old = SQLiteBackend(db)
    _stored(old)
    assert old.get("k")[1] == {"v": 1}, "precondition: the older cash reads its own entry"
    old.shutdown()

    as_format(CACHE_FORMAT_VERSION)
    new = SQLiteBackend(db)
    try:
        assert new.get("k") == (None, None), "an entry of an older cache format was served"
        assert new.get_metadata("k") is None
        assert new.entry_count() == 0
        _stored(new, value={"v": 2})
        assert new.get("k")[1] == {"v": 2}
    finally:
        new.shutdown()


def test_sqlite_drops_a_database_stamped_before_the_stamp_held_the_format(tmp_path):
    """A database of a build that stamped the table layout only."""
    db = str(tmp_path / "c.db")
    first = SQLiteBackend(db)
    _stored(first)
    first.shutdown()
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SQLiteBackend.SCHEMA_VERSION}")
    conn.commit()
    conn.close()

    again = SQLiteBackend(db)
    try:
        assert again.get("k") == (None, None)
    finally:
        again.shutdown()


def test_sqlite_keeps_a_database_of_its_own_format(tmp_path):
    """The control: a stamp check that fired on every open would pass the rest."""
    db = str(tmp_path / "c.db")
    first = SQLiteBackend(db)
    _stored(first)
    first.shutdown()
    second = SQLiteBackend(db)
    try:
        assert second.get("k")[1] == {"v": 1}
    finally:
        second.shutdown()


def test_sqlite_leaves_a_newer_cashs_database_alone(tmp_path, as_format):
    db = str(tmp_path / "c.db")
    as_format(CACHE_FORMAT_VERSION + 1)
    newer = SQLiteBackend(db)
    _stored(newer)
    newer.shutdown()
    stamp = _user_version(db)

    as_format(CACHE_FORMAT_VERSION)
    older = SQLiteBackend(db)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert older.get("k") == (None, None)
        older.set("mine", 1, {})
        assert older.get("mine") == (None, None)
        older.clear()
        older.shutdown()

    assert _rows(db) == 1, "the newer cash's entry was removed"
    assert _user_version(db) == stamp
    said = [str(w.message) for w in rec if getattr(w.message, "code", "") == "CACHE-FORMAT-NEWER"]
    assert len(said) == 1 and db in said[0], [str(w.message) for w in rec]

    as_format(CACHE_FORMAT_VERSION + 1)
    back = SQLiteBackend(db)
    try:
        assert back.get("k")[1] == {"v": 1}, "the newer cash lost its entry"
    finally:
        back.shutdown()


def test_a_script_after_a_format_change_computes_again_on_sqlite(tmp_path):
    """The whole path: the same script run by an 'older' cash (format one
    lower) and then by this one, over one SQLite cache."""
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import sys
            import cash
            import cash.backends.cache_dir as cd

            @cash.cache
            def f(n):
                print("computing", n, flush=True)
                return n * 2

            if sys.argv[1] == "old":
                cd.CACHE_FORMAT_VERSION -= 1
            print("result", f(1))
            """
        ),
        encoding="utf-8",
    )
    env = {"CASH_BACKEND": "sqlite"}
    assert "computing 1" in run_python("job.py", "old", cwd=tmp_path, env=env).stdout
    assert "computing" not in run_python("job.py", "old", cwd=tmp_path, env=env).stdout, "nothing was stored"

    after = run_python("job.py", "new", cwd=tmp_path, env=env).stdout

    assert "computing 1" in after and "result 2" in after, "the older format's entry was served"
    assert "computing" not in run_python("job.py", "new", cwd=tmp_path, env=env).stdout


# -- Redis and S3 -------------------------------------------------------------


@pytest.fixture
def redis():
    pytest.importorskip("redis")
    from cash.backends.redis_backend import RedisBackend

    with patch("redis.Redis"):
        backend = RedisBackend(prefix="p:")
    backend.client = FakeRedisClient()
    return backend


@pytest.fixture
def s3():
    pytest.importorskip("boto3")
    from cash.backends.s3_backend import S3Backend

    with patch("boto3.client"):
        backend = S3Backend(bucket="b", prefix="p/")
    backend.s3 = FakeS3Client()
    return backend


def _names(backend) -> set:
    return set(backend.client.store if hasattr(backend, "client") else backend.s3.store)


@pytest.mark.parametrize("kind", ["redis", "s3"])
def test_a_shared_store_keeps_each_format_apart(kind, request, as_format):
    backend = request.getfixturevalue(kind)
    as_format(CACHE_FORMAT_VERSION - 1)
    _stored(backend, value="from the older cash")
    assert backend.get("k")[1] == "from the older cash", "precondition"
    older_names = _names(backend)

    as_format(CACHE_FORMAT_VERSION)
    assert backend.get("k") == (None, None), "an entry of another cache format was served"
    assert backend.get_metadata("k") is None
    assert backend.list_entries() == []
    _stored(backend, value="from this cash")
    assert backend.get("k")[1] == "from this cash"
    assert [m["key"] for m in backend.list_entries()] == ["k"]
    assert older_names < _names(backend), "storing in one format replaced the other format's entry"
    backend.delete("k")
    assert _names(backend) == older_names, "a delete reached another format's entry"

    as_format(CACHE_FORMAT_VERSION - 1)
    assert backend.get("k")[1] == "from the older cash", "the older cash lost its entry to the newer one"


@pytest.mark.parametrize("kind", ["redis", "s3"])
def test_an_entry_named_without_a_format_is_not_served(kind, request):
    """What a cash from before entries carried the format left in the store."""
    backend = request.getfixturevalue(kind)
    _stored(backend)
    tag = cache_dir.format_tag()
    if kind == "redis":
        store = backend.client.store
        for name in list(store):
            store[name.replace(f"{tag}:", "", 1)] = store.pop(name)
    else:
        store = backend.s3.store
        for bucket, name in list(store):
            store[(bucket, name.replace(f"{tag}/", "", 1))] = store.pop((bucket, name))
    assert all(tag not in str(name) for name in _names(backend)), "precondition: the names were rewritten"

    assert backend.get("k") == (None, None)
    assert backend.get_metadata("k") is None
    assert backend.list_entries() == []
