"""A file read moments after it was written is not hashed on every hit forever.

An unchanged file is trusted by its metadata only if it had settled before its
digest was taken. The digest's time was never moved on, so an input read
within seconds of being written -- a stage's output read by the next stage --
was hashed in full on every hit, in every process, for the life of the entry:
0.6 s a hit for a 200 MB file behind a microsecond body. Once a check re-hashes
it after it has settled and finds it unchanged, that check's time is kept.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types

import pytest

from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def _no_digest_reuse(monkeypatch):
    """No in-process memo: what is under test is whether the file is read at all."""
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_TTL_SECONDS", 0.0)
    file_dep_snapshot._HASH_MEMO.clear()


@pytest.fixture
def reads(monkeypatch):
    real = file_dep_snapshot.file_content_hash
    seen: list[str] = []

    def counting(path, *a, **k):
        seen.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(file_dep_snapshot, "file_content_hash", counting)
    return seen


def _later(monkeypatch, seconds):
    real = file_dep_snapshot.time
    monkeypatch.setattr(
        file_dep_snapshot, "time", types.SimpleNamespace(time=lambda: real.time() + seconds, monotonic=real.monotonic)
    )


def test_a_settled_recheck_is_remembered(tmp_path, reads, monkeypatch):
    path = str(tmp_path / "fresh.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("a,b\n1,2\n")
    stored = snapshot_file_deps({path})[path]  # hashed the moment it was written
    _later(monkeypatch, 3600)
    reads.clear()

    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [path], "not settled when it was hashed: the content decides once"
    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [path], "hashed again though it had settled and nothing moved"


def test_a_recheck_before_it_settles_is_not_remembered(tmp_path, reads):
    """Control: still inside the window, the next check reads it again."""
    path = str(tmp_path / "fresh.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("a,b\n1,2\n")
    stored = snapshot_file_deps({path})[path]
    reads.clear()
    assert file_dep_is_fresh(path, stored) == (True, None)
    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [path, path]


def test_an_edit_after_the_recheck_is_still_caught(tmp_path, monkeypatch):
    path = str(tmp_path / "fresh.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("a,b\n1,2\n")
    stored = snapshot_file_deps({path})[path]
    _later(monkeypatch, 3600)
    assert file_dep_is_fresh(path, stored) == (True, None)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("a,b\n9,9\n")  # same size
    assert file_dep_is_fresh(path, stored)[0] is False


_SCRIPT = """
import os, sys, time
import cash
from cash.tracking import file_dep_snapshot
file_dep_snapshot._HASH_MEMO_MIN_AGE_SECONDS = 1.0
reads = []
real = file_dep_snapshot.file_content_hash
def counting(path, *a, **k):
    reads.append(path)
    return real(path, *a, **k)
file_dep_snapshot.file_content_hash = counting
c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
@c.cache
def head(p):
    time.sleep({floor})
    with open(p, "rb") as fh:
        return fh.read(4)
data = sys.argv[2]
if sys.argv[3] == "make":
    with open(data, "wb") as fh:
        fh.write(b"abcdefgh")
    # Written "just now" as of the body's read, however long the first
    # call's analysis takes: a slow machine let the file settle before it.
    t = time.time_ns() + 3_000_000_000
    os.utime(data, ns=(t, t))
head(data)
reads = [r for r in reads if r == data]
print(len(reads), head.cache_info()["hits"])
"""


def test_the_remembered_check_reaches_the_next_process(tmp_path):
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(_SCRIPT).replace("{floor}", str(ABOVE_PERSISTENCE_FLOOR_S)), encoding="utf-8")
    cache, data = str(tmp_path / "cache"), str(tmp_path / "data.bin")

    def run(mode):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run(
            [sys.executable, str(script), cache, data, mode], capture_output=True, text=True, env=env, check=True
        )
        return out.stdout.split()

    run("make")
    import time

    time.sleep(4.2)  # past the future mtime plus the 1 s window
    assert run("call") == ["1", "1"], "the first settled check reads it once"
    assert run("call") == ["0", "1"], "the next process read the unchanged file again"
