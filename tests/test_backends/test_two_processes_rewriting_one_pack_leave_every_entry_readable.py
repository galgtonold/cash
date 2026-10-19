"""Two processes rewriting one pack leave every entry readable.

Evicting an entry that shared a pack rewrites the pack without that entry's
own chunks (`ChunkStore._compact`). Two processes evicting it at once both
wrote the same new pack; the one that found the entries already moved then
removed that pack as its own, unheld, and every moved entry was left
pointing at nothing. A process now claims a pack (``<pack>.claim``) before
rewriting it; the other leaves it alone. Within a process, a rewrite never
lands between another thread's write of an entry and its rename: it takes
the entry's lock, and skips an entry another thread holds.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from cash.backends.file_backend import FileBackend
from tests._scripts import child_env

ROWS = 200_000  # 1.6 MB per float64 column: each one a chunk


def _values():
    rng = np.random.default_rng(1)
    shared = [rng.random(ROWS) for _ in range(2)]
    return {
        "first": [*shared, *[rng.random(ROWS) for _ in range(3)]],
        "second": [*shared, np.ones(ROWS)],
        "third": [*shared, np.zeros(ROWS)],
    }


def _store_all(cache):
    backend = FileBackend(str(cache), flush_interval=0)
    for key, value in _values().items():
        backend.set(key, value, {"execution_time": 1.0})
        backend._writes.wait_all()
    return backend


def _packs(cache):
    return sorted(p.name for p in (cache / "chunks").iterdir() if p.suffix == ".pack")


def _wait_for(path, proc=None, limit=60.0):
    end = time.monotonic() + limit
    while not path.exists():
        assert proc is None or proc.poll() is None, proc.communicate()
        assert time.monotonic() < end, f"{path.name} never came"
        time.sleep(0.02)


# Process A: evicts "first", and stops once its new pack is written until
# the test says go.
PAUSED_REWRITE = """
import os, time
from cash.backends.file_backend import FileBackend

backend = FileBackend({cache!r}, flush_interval=0)
store = backend.chunks
real = store._write_pack

def write_pack(fresh):
    made = real(fresh)
    open({wrote!r}, "w").close()
    while not os.path.exists({go!r}):
        time.sleep(0.02)
    return made

store._write_pack = write_pack
backend.delete("first")
"""


def test_a_second_process_leaves_a_claimed_pack_alone(tmp_path):
    cache = tmp_path / "cache"
    _store_all(cache)
    old = _packs(cache)
    other = FileBackend(str(cache), flush_interval=0)
    other.chunks.ensure_index()  # knows "first" holds the pack, as A does
    wrote, go = tmp_path / "a-wrote", tmp_path / "go"
    script = PAUSED_REWRITE.format(cache=str(cache), wrote=str(wrote), go=str(go))
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        env=child_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(wrote, proc)
        # The second process evicts "first" too, A's rewrite half done: it
        # would write the same new pack and, finding the entries moved by A,
        # remove it as its own.
        rewrote = []
        real = other.chunks._write_pack

        def write_pack(fresh):
            made = real(fresh)
            rewrote.append(made[0])
            go.touch()
            proc.wait(timeout=60)
            return made

        other.chunks._write_pack = write_pack
        other.delete("first")
        assert rewrote == [], "the claimed pack was rewritten by a second process"
        go.touch()
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
    finally:
        if proc.poll() is None:
            proc.kill()
    values = _values()
    reader = FileBackend(str(cache), flush_interval=0)
    for key in ("second", "third"):
        got = reader.get(key)
        assert got != (None, None), f"{key} lost its chunks"
        assert [a.tobytes() for a in got[1]] == [a.tobytes() for a in values[key]], key
    now = _packs(cache)
    assert len(set(old) - set(now)) == 1 and len(now) == len(old), "first's pack rewritten without its chunks"
    assert not list((cache / "chunks").glob("*.claim"))


def test_an_entry_another_thread_holds_is_not_rewritten(tmp_path):
    """Its write holds the entry's lock: the rewrite skips it, so the old
    pack stays for it, and it still reads."""
    cache = tmp_path / "cache"
    backend = _store_all(cache)
    old = _packs(cache)
    held = threading.Event()
    done = threading.Event()

    def hold():
        with backend.chunks.entry_lock(backend._stem("second")):
            held.set()
            done.wait(60)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        held.wait(60)
        backend.delete("first")
    finally:
        done.set()
        thread.join()
    values = _values()
    reader = FileBackend(str(cache), flush_interval=0)
    for key in ("second", "third"):
        got = reader.get(key)
        assert [a.tobytes() for a in got[1]] == [a.tobytes() for a in values[key]], key
    assert set(old) <= set(_packs(cache)), "the old pack went while second still names it"
