"""Processes sharing one SQLite cache must never see its lock errors.

Every hit read the row and then wrote its access time on the same
connection. With a few processes on one database that write failed with
``sqlite3.OperationalError: database is locked``, and the error left ``get``
into the caller of the cached function: hundreds of calls of a correct
program raised.

A busy database now costs time or a recompute: a hit returns its value and
keeps the access-time note for a later write, a read that cannot be done is a
miss, and a store that cannot be done is reported as a failed store.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import time
import warnings

import pytest

from cash.backends.sqlite_backend import SQLiteBackend
from cash.exceptions import CacheBackendError
from tests._scripts import child_env


@pytest.fixture
def quick(monkeypatch):
    """A database that stays busy is given up on in a fraction of a second."""
    monkeypatch.setattr(SQLiteBackend, "BUSY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(SQLiteBackend, "WRITE_ATTEMPTS", 2)


def _stored(db, key="k", value=None, **kwargs) -> SQLiteBackend:
    backend = SQLiteBackend(db, **kwargs)
    backend.set(key, {"v": 1} if value is None else value, {"execution_time": 1.0})
    backend._writes.wait_all()
    return backend


def _another_process_writing(db: str, how: str = "IMMEDIATE") -> sqlite3.Connection:
    """A second connection in the middle of a write transaction."""
    other = sqlite3.connect(db, isolation_level=None, timeout=0.05)
    other.execute(f"BEGIN {how}")
    return other


def _access_count(db: str, key: str = "k") -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT access_count FROM cache_entries WHERE key = ?", (key,)).fetchone()[0]
    finally:
        conn.close()


def test_a_hit_while_another_process_writes_returns_its_value(tmp_path, quick):
    db = str(tmp_path / "c.db")
    backend = _stored(db)
    other = _another_process_writing(db)
    try:
        started = time.perf_counter()
        metadata, value = backend.get("k")
        again = backend.get("k")[0]
        waited = time.perf_counter() - started
    finally:
        other.execute("ROLLBACK")
        other.close()

    assert value == {"v": 1}
    assert (metadata["access_count"], again["access_count"]) == (1, 2)
    assert waited < 2.0, f"two hits waited {waited:.1f} s for another process's write to note their access time"
    assert _access_count(db) == 0, "precondition: the notes could not be written while the database was busy"

    # The notes travel with the next write this process gets through.
    backend.set("other", 1, {})
    backend._writes.wait_all()
    assert _access_count(db) == 2
    backend.shutdown()


def test_the_access_times_of_a_run_that_only_reads_are_written_at_shutdown(tmp_path, quick):
    db = str(tmp_path / "c.db")
    _stored(db).shutdown()
    reader = SQLiteBackend(db)
    other = _another_process_writing(db)
    try:
        assert reader.get("k")[1] == {"v": 1}
    finally:
        other.execute("ROLLBACK")
        other.close()
    reader.shutdown()
    assert _access_count(db) == 1


def test_a_hit_on_a_free_database_notes_its_access_at_once(tmp_path):
    """The control: with nobody else in the database nothing is deferred."""
    db = str(tmp_path / "c.db")
    backend = _stored(db)
    backend.get("k")
    backend.get("k")
    assert _access_count(db) == 2
    backend.shutdown()


def test_a_read_the_database_refuses_is_a_miss(tmp_path, quick):
    """Without WAL a writer shuts readers out; so does any I/O error."""
    db = str(tmp_path / "c.db")
    backend = _stored(db, wal_mode=False)
    other = _another_process_writing(db, "EXCLUSIVE")
    try:
        assert backend.get("k") == (None, None)
        assert backend.get_metadata("k") is None
        assert backend.list_entries() == []
        assert backend.entry_count() == 0
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert backend.get("k")[1] == {"v": 1}, "the entry is served again once the database is free"
    backend.shutdown()


@pytest.mark.expects_failed_writes
def test_a_store_into_a_database_that_stays_busy_is_a_failed_store(tmp_path, quick):
    db = str(tmp_path / "c.db")
    backend = _stored(db)
    other = _another_process_writing(db)
    try:
        backend.set("new", {"v": 2}, {"execution_time": 1.0})  # never raises: the write is queued
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert backend.get("new") == (None, None)
    finally:
        other.execute("ROLLBACK")
        other.close()
    said = [str(w.message) for w in rec if getattr(w.message, "code", "") == "STORE-FAILED"]
    assert len(said) == 1, [str(w.message) for w in rec]

    backend.set("new", {"v": 2}, {"execution_time": 1.0})
    assert backend.get("new")[1] == {"v": 2}, "a failed store must not poison the key"
    backend.shutdown()


def test_a_store_waits_its_turn_behind_another_process(tmp_path, monkeypatch):
    """Several short attempts add up: a writer that lets go in time loses nothing."""
    import threading

    monkeypatch.setattr(SQLiteBackend, "BUSY_TIMEOUT_S", 0.2)
    monkeypatch.setattr(SQLiteBackend, "WRITE_ATTEMPTS", 5)
    db = str(tmp_path / "c.db")
    backend = _stored(db)
    other = sqlite3.connect(db, isolation_level=None, timeout=0.05, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    release = threading.Timer(0.5, lambda: other.execute("ROLLBACK"))
    release.start()
    try:
        backend.set("new", {"v": 2}, {"execution_time": 1.0})
        assert backend.get("new")[1] == {"v": 2}
    finally:
        release.join()
        other.close()
        backend.shutdown()


def test_a_delete_the_database_refuses_says_so(tmp_path, quick):
    """Removing is the one thing that may not fail quietly: the entry the
    caller asked to drop would still be served."""
    db = str(tmp_path / "c.db")
    backend = _stored(db)
    other = _another_process_writing(db)
    try:
        with pytest.raises(CacheBackendError, match="delete"):
            backend.delete("k")
        with pytest.raises(CacheBackendError, match="clear"):
            backend.clear()
    finally:
        other.execute("ROLLBACK")
        other.close()
    backend.delete("k")
    assert backend.get("k") == (None, None)
    backend.shutdown()


def test_the_cap_still_evicts_the_least_recently_used(tmp_path):
    db = str(tmp_path / "c.db")
    backend = SQLiteBackend(db, max_size_bytes=3000)
    for i in range(6):
        backend.set(f"k{i}", b"x" * 900, {})
        backend._writes.wait_all()
        if i == 0:
            continue
        backend.get("k0")  # in use throughout
    kept = sorted(m["key"] for m in backend.list_entries())
    assert "k0" in kept and "k5" in kept and len(kept) < 6, kept
    backend.shutdown()


JOB = """
import random, sys, time, warnings
warnings.simplefilter("ignore")
import cash

@cash.cache
def f(n):
    return bytes([n]) * 400_000

rng = random.Random(int(sys.argv[1]))
# Four seconds, and at least 30 calls each: a process the others keep
# waiting on the database's lock still makes enough calls to be tested.
end = time.time() + 4
give_up = time.time() + 60
errors = {}
calls = 0
while (time.time() < end or calls < 30) and time.time() < give_up:
    n = rng.randrange(40)
    try:
        assert f(n)[0] == n
    except Exception as exc:
        name = type(exc).__name__ + ": " + str(exc)
        errors[name] = errors.get(name, 0) + 1
    calls += 1
print(calls, errors)
"""


@pytest.mark.timeout(120)
def test_processes_sharing_a_sqlite_cache_raise_nothing_into_cached_calls(tmp_path):
    """Four processes, 40 keys, a cap that keeps all of them evicting."""
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    env = child_env(tmp_path, env={"CASH_BACKEND": "sqlite", "CASH_MAX_CACHE_SIZE": "6MB"})
    procs = [
        subprocess.Popen(
            [sys.executable, "job.py", str(i)],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for i in range(4)
    ]
    outs = [p.communicate(timeout=90)[0].strip() for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    counts = [int(out.split()[0]) for out in outs]
    assert all(out.endswith("{}") for out in outs), f"cached calls raised: {outs}"
    assert min(counts) > 20, f"precondition: every process made calls: {outs}"
