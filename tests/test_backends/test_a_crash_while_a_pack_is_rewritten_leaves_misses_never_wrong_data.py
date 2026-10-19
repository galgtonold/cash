"""A crash while a pack is rewritten leaves misses, never wrong data.

Evicting an entry rewrites a pack it shared without that entry's own chunks
(`ChunkStore._compact`): 1. the new pack is written beside the old one;
2. each entry holding the old one is rewritten whole and renamed into place,
pointing at the new one; 3. the old pack is removed once no entry points
at it. The process is killed after each step, and between two holders.
Every entry left reads back its own value or misses, and the next process
removes the pack no entry holds any more.
"""

from __future__ import annotations

import textwrap

import pytest

np = pytest.importorskip("numpy")

from cash.backends import chunk_store
from cash.backends.file_backend import FileBackend
from tests._scripts import run_python

ROWS = 200_000  # 1.6 MB per float64 column: each one a chunk

SETUP = """
import os
import numpy as np
from cash.backends import chunk_store
from cash.backends.file_backend import FileBackend

ROWS = {rows}
rng = np.random.default_rng(1)
shared = [rng.random(ROWS) for _ in range(2)]
values = {{
    "first": [*shared, *[rng.random(ROWS) for _ in range(3)]],
    "second": [*shared, np.ones(ROWS)],
    "third": [*shared, np.zeros(ROWS)],
}}
np.save("values.npy", np.stack([a for v in values.values() for a in v]))
backend = FileBackend("cache", flush_interval=0)
for key, value in values.items():
    backend.set(key, value, {{"execution_time": 1.0}})
    backend._writes.wait_all()

step = {step!r}
store = backend.chunks
real_write, real_repoint, real_drop = store._write_pack, store._repoint, store._drop
moved = []

def write_pack(fresh):
    made = real_write(fresh)
    if step == "after the new pack":
        os._exit(0)
    return made

def repoint(*args):
    done = real_repoint(*args)
    moved.append(done)
    if step == "between holders" and len(moved) == 1:
        os._exit(0)
    if step == "after every holder" and len(moved) == 2:
        os._exit(0)
    return done

def drop(names):
    names = list(names)
    if step == "while removing the old pack" and any(n.endswith(".pack") for n in names):
        os._exit(0)
    return real_drop(names)

store._write_pack, store._repoint, store._drop = write_pack, repoint, drop
backend.delete("first")
raise SystemExit("the rewrite never reached the step")
"""

STEPS = ["after the new pack", "between holders", "after every holder", "while removing the old pack"]


def _values(tmp_path):
    arrays = list(np.load(tmp_path / "values.npy"))
    return {"first": arrays[:5], "second": arrays[5:8], "third": arrays[8:11]}


@pytest.mark.parametrize("step", STEPS)
def test_each_entry_reads_its_own_value_or_misses(tmp_path, step, monkeypatch):
    run_python("-c", textwrap.dedent(SETUP.format(rows=ROWS, step=step)), cwd=tmp_path)
    values = _values(tmp_path)
    reader = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    served = 0
    for key in ("second", "third"):
        got = reader.get(key)
        if got == (None, None):
            continue
        served += 1
        assert [a.tobytes() for a in got[1]] == [a.tobytes() for a in values[key]], key
    assert served == 2, "a holder is whole before and after its rename: none should miss"
    # The next process sweeps what no entry holds; what is left is what they hold.
    monkeypatch.setattr(chunk_store, "ORPHAN_CHUNK_AGE", -1.0)
    sweeper = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    sweeper.chunks.ensure_index()
    held = {name for stem, names in sweeper.chunks._held_by.items() for name in names}
    on_disk = {p.name for p in (tmp_path / "cache" / "chunks").iterdir()}
    assert on_disk == held
    for key in ("second", "third"):
        got = sweeper.get(key)
        assert [a.tobytes() for a in got[1]] == [a.tobytes() for a in values[key]], key
