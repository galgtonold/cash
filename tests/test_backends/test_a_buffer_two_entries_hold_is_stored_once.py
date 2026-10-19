"""A big buffer several entries hold is written to disk once.

A frame with one column added, the same data stored by two statements, each
iteration of a loop returning its input's columns: each entry wrote every
column again, gigabytes over a session of which most were already on disk.
The file tier now keeps every buffer of a megabyte or more in the folder's
chunk store, named by its content digest, and an entry names the chunks it
holds. The chunks one write adds share one file (a pack), which goes when no
entry holds a chunk in it. These tests pin that a value comes back byte for
byte, that removing one holder keeps what another still needs, that a
missing or damaged chunk is a miss (never other data), and that another
process reads the chunks.
"""

from __future__ import annotations

import os
import pickle
import textwrap
import time

import pytest

np = pytest.importorskip("numpy")

from cash import bulk_digest as bulk_digest_module
from cash.backends.cache_dir import entry_totals, is_cash_file
from cash.backends.entry_format import read_entry, update_metadata_in_place
from cash.backends.file_backend import FileBackend
from tests._scripts import run_python

ROWS = 400_000  # 3.2 MB per float64 column: well above CHUNK_MIN_BYTES
#: Where the chunks are, and the metadata field listing an entry's.
CHUNK_DIRNAME = "chunks"
CHUNKS_FIELD = "chunks"
PLACES_FIELD = "chunk_places"


def _chunk_store():
    from cash.backends import chunk_store

    return chunk_store


def _backend(tmp_path, **kwargs):
    return FileBackend(str(tmp_path / "cache"), flush_interval=0, **kwargs)


def _store(backend, key, value, seconds=1.0):
    backend.set(key, value, {"execution_time": seconds})
    backend._writes.wait_all()


def _chunks(tmp_path):
    folder = tmp_path / "cache" / CHUNK_DIRNAME
    return sorted(p for p in folder.iterdir() if p.suffix in (".chunk", ".pack")) if folder.exists() else []


def _placed(tmp_path):
    """The chunks the entries in the folder hold, by digest."""
    folder = tmp_path / "cache"
    found = set()
    for path in folder.glob("*.entry"):
        found.update(read_entry(str(path), with_payload=False)[0].get(PLACES_FIELD) or {})
    return found


def _chunk_bytes(tmp_path):
    return sum(p.stat().st_size for p in _chunks(tmp_path))


def _arrays(seed, n=3):
    rng = np.random.default_rng(seed)
    return [rng.random(ROWS) for _ in range(n)]


def _same(got, want):
    assert type(got) is type(want)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def test_an_entry_that_adds_one_array_writes_only_that_array(tmp_path):
    base = _arrays(1)
    derived = [*base, np.arange(ROWS, dtype=np.float64)]
    backend = _backend(tmp_path)
    _store(backend, "base", base)
    after_base = _chunk_bytes(tmp_path)
    _store(backend, "derived", derived)
    grown = _chunk_bytes(tmp_path) - after_base
    assert grown == derived[-1].nbytes + _chunk_store().TRAILER, f"the derived entry added {grown} bytes of chunks"
    reader = _backend(tmp_path)
    _same(reader.get("base")[1], base)
    _same(reader.get("derived")[1], derived)


def test_a_frame_with_a_column_added_comes_back_as_it_was_stored(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": np.arange(ROWS, dtype=float), "b": np.random.default_rng(0).random(ROWS)})
    wider = frame.assign(c=frame["a"] * 2)
    backend = _backend(tmp_path)
    _store(backend, "frame", frame)
    _store(backend, "wider", wider)
    reader = _backend(tmp_path)
    pd.testing.assert_frame_equal(reader.get("frame")[1], frame)
    pd.testing.assert_frame_equal(reader.get("wider")[1], wider)
    # Both entries are on disk together in less than either written whole twice.
    total = entry_totals(str(tmp_path / "cache"))[1]
    assert total < frame.memory_usage().sum() + wider.memory_usage().sum()


def test_one_value_holding_equal_arrays_writes_them_once(tmp_path):
    arr = np.random.default_rng(3).random(ROWS)
    value = {"x": arr, "y": arr.copy()}
    backend = _backend(tmp_path)
    _store(backend, "k", value)
    assert len(_chunks(tmp_path)) == 1
    assert _chunk_bytes(tmp_path) == arr.nbytes + _chunk_store().TRAILER
    got = _backend(tmp_path).get("k")[1]
    assert got["x"].tobytes() == arr.tobytes() and got["y"].tobytes() == arr.tobytes()
    got["x"][0] = -1.0  # two arrays, not one shared by both
    assert got["y"][0] == arr[0]


def test_removing_one_holder_keeps_what_the_other_needs(tmp_path):
    shared = _arrays(2)
    first = [*shared, np.zeros(ROWS)]
    second = [*shared, np.ones(ROWS)]
    backend = _backend(tmp_path)
    _store(backend, "first", first)
    _store(backend, "second", second)
    assert len(_placed(tmp_path)) == 5
    assert len(_chunks(tmp_path)) == 2  # first's pack, and second's of the one it added
    backend.delete("first")
    assert len(_placed(tmp_path)) == 4
    assert len(_chunks(tmp_path)) == 2, "second still holds chunks in first's pack"
    _same(_backend(tmp_path).get("second")[1], second)
    backend.delete("second")
    assert _chunks(tmp_path) == []


def test_another_process_removing_a_holder_keeps_what_the_other_needs(tmp_path):
    """The remover did not write either entry: it learns the holders from
    the entries' metadata."""
    shared = _arrays(4)
    writer = _backend(tmp_path)
    _store(writer, "first", [*shared, np.zeros(ROWS)])
    _store(writer, "second", [*shared, np.ones(ROWS)])
    writer.shutdown()
    remover = _backend(tmp_path)
    remover.delete("first")
    assert len(_placed(tmp_path)) == 4
    assert len(_chunks(tmp_path)) == 2
    _same(_backend(tmp_path).get("second")[1], [*shared, np.ones(ROWS)])


def test_rewriting_an_entry_frees_the_chunks_it_no_longer_holds(tmp_path):
    backend = _backend(tmp_path)
    _store(backend, "k", _arrays(5))
    _store(backend, "k", _arrays(6))
    assert len(_chunks(tmp_path)) == 1 and len(_placed(tmp_path)) == 3
    other = _backend(tmp_path)
    _store(other, "k", _arrays(7))  # replaces an entry this process never saw
    assert len(_chunks(tmp_path)) == 1 and len(_placed(tmp_path)) == 3
    _same(_backend(tmp_path).get("k")[1], _arrays(7))


def test_eviction_keeps_every_chunk_a_remaining_entry_needs(tmp_path):
    """Under a cap, entries sharing chunks are evicted by what they hold, and
    each one left still restores; the byte count matches the folder."""
    shared = _arrays(8, n=4)
    cap = 12 * ROWS * 8  # room for the shared arrays and a few own ones
    backend = _backend(tmp_path, max_size_bytes=cap)
    values = {}
    for i in range(12):
        values[f"k{i}"] = [*shared, np.full(ROWS, float(i))]
        _store(backend, f"k{i}", values[f"k{i}"], seconds=0.01 * (i + 1))
    reader = _backend(tmp_path)
    left = [k for k in values if reader.get_metadata(k) is not None]
    assert 0 < len(left) < len(values), "the cap evicted nothing, or everything"
    for key in left:
        _same(reader.get(key)[1], values[key])
    folder = str(tmp_path / "cache")
    on_disk = entry_totals(folder)[1]
    assert on_disk <= cap
    assert backend.evictor.current_bytes == on_disk
    held = set()
    for key in left:
        held.update(name for name, _size in read_entry(backend._get_path(key), with_payload=False)[0][CHUNKS_FIELD])
    assert {p.name for p in _chunks(tmp_path)} == held, "a chunk no entry holds was left behind"


def test_a_missing_chunk_is_a_miss_and_the_next_store_heals_it(tmp_path):
    value = _arrays(9)
    backend = _backend(tmp_path)
    _store(backend, "k", value)
    victim = _chunks(tmp_path)[0]
    victim.unlink()
    assert _backend(tmp_path).get("k") == (None, None)
    _store(_backend(tmp_path), "k", value)
    _same(_backend(tmp_path).get("k")[1], value)


@pytest.mark.parametrize("where", [0, 0.5, -1])
def test_a_damaged_chunk_is_a_miss_never_other_data(tmp_path, where):
    value = _arrays(10)
    backend = _backend(tmp_path)
    _store(backend, "k", value)
    victim = _chunks(tmp_path)[0]
    data = bytearray(victim.read_bytes())
    data[int(len(data) * where) if where else 0] ^= 0x01  # -1: the crc trailer
    victim.write_bytes(data)
    assert _backend(tmp_path).get("k") == (None, None)
    assert not victim.exists(), "a damaged chunk stays, so every later store of it would miss"
    _store(_backend(tmp_path), "k", value)
    _same(_backend(tmp_path).get("k")[1], value)


def test_a_chunk_of_the_wrong_size_is_a_miss(tmp_path):
    backend = _backend(tmp_path)
    _store(backend, "k", _arrays(11))
    victim = _chunks(tmp_path)[0]
    with open(victim, "r+b") as fh:
        fh.truncate(victim.stat().st_size - 1000)
    assert _backend(tmp_path).get("k") == (None, None)


def test_a_chunk_is_named_by_the_digest_scheme(tmp_path, monkeypatch):
    """The scheme's tag is part of the name: an install hashing with another
    algorithm never takes this chunk for its own."""
    arr = np.random.default_rng(12).random(ROWS)
    _store(_backend(tmp_path), "k", [arr])
    (first,) = _chunks(tmp_path)
    assert first.name == _chunk_store().pack_name([bulk_digest_module.bulk_digest(arr)])
    monkeypatch.setattr(bulk_digest_module, "TAG", b"cash-bulk-other-tree-1\x00")
    _store(_backend(tmp_path), "k2", [arr])
    assert len(_chunks(tmp_path)) == 2


def test_the_flusher_keeps_the_entrys_own_chunk_list(tmp_path):
    """Access stamps are written back over the metadata; which chunks the
    payload holds is the payload's, whatever dict is written back."""
    backend = _backend(tmp_path)
    _store(backend, "k", _arrays(13))
    path = backend._get_path("k")
    meta, _ = read_entry(path, with_payload=False)
    held = meta[CHUNKS_FIELD]
    stale = {k: v for k, v in meta.items() if k != CHUNKS_FIELD}
    stale["access_count"] = 7
    assert update_metadata_in_place(path, stale)
    meta, _ = read_entry(path, with_payload=False)
    assert meta[CHUNKS_FIELD] == held and meta["access_count"] == 7


def test_clear_removes_the_chunks(tmp_path):
    backend = _backend(tmp_path)
    _store(backend, "k", _arrays(14))
    assert _chunks(tmp_path)
    backend.clear()
    assert _chunks(tmp_path) == []
    assert entry_totals(str(tmp_path / "cache")) == (0, 0)


def test_chunks_are_cash_files():
    """``cash clear`` removes them, and reading one is never a dependency."""
    assert is_cash_file(f"{CHUNK_DIRNAME}/{'ab' * 32}.chunk")
    assert is_cash_file(f"{CHUNK_DIRNAME}/.tmp-1a2b.123-abcdef.part")
    assert not is_cash_file(f"{CHUNK_DIRNAME}/notes.txt")


def test_an_unheld_chunk_left_by_a_killed_writer_is_swept(tmp_path):
    backend = _backend(tmp_path)
    _store(backend, "k", _arrays(15))
    folder = tmp_path / "cache" / CHUNK_DIRNAME
    orphan = folder / ("cd" * 32 + ".chunk")
    orphan.write_bytes(bytes(100))
    old = time.time() - _chunk_store().ORPHAN_CHUNK_AGE - 60
    os.utime(orphan, (old, old))
    young = folder / ("ef" * 32 + ".chunk")
    young.write_bytes(bytes(100))  # maybe a write in flight in another process
    fresh = _backend(tmp_path)
    fresh.delete("k")  # needs the holders: reads the folder once
    assert not orphan.exists()
    assert young.exists()
    assert len(_chunks(tmp_path)) == 1


def test_a_chunk_another_process_just_touched_is_left_alone(tmp_path):
    """Another process stored an entry holding the chunk after this one read
    the folder: its touch makes the chunk newer than that reading."""
    shared = _arrays(16)
    backend = _backend(tmp_path)
    _store(backend, "mine", shared)
    backend.chunks.ensure_index()
    time.sleep(0.01)
    other = _backend(tmp_path)
    _store(other, "theirs", [*shared, np.zeros(ROWS)])
    backend.delete("mine")
    _same(_backend(tmp_path).get("theirs")[1], [*shared, np.zeros(ROWS)])


def test_a_new_process_restores_values_whose_buffers_are_chunks(tmp_path):
    pytest.importorskip("pandas")
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import os, pickle, sys
            import numpy as np, pandas as pd
            import cash

            def ran(name):
                fd = os.open("ran.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, name.encode() + b"\\n")
                os.close(fd)

            @cash.cache
            def frame(n):
                ran("frame")
                rng = np.random.default_rng(0)
                return pd.DataFrame({"a": rng.random(n), "b": np.arange(n, dtype=float)})

            @cash.cache
            def wider(n):
                ran("wider")
                f = frame(n)
                return f.assign(c=f["a"] * 3)

            out = (frame(400_000), wider(400_000))
            sys.stdout.write(pickle.dumps(out).hex())
            """
        ),
        encoding="utf-8",
    )
    first = run_python("-W", "ignore", "job.py", cwd=tmp_path, timeout=120)
    assert any((tmp_path / ".cash" / CHUNK_DIRNAME).iterdir())
    ran = (tmp_path / "ran.log").read_text(encoding="utf-8")
    second = run_python("-W", "ignore", "job.py", cwd=tmp_path, timeout=120)
    assert (tmp_path / "ran.log").read_text(encoding="utf-8") == ran, "the second process computed again"
    want = pickle.loads(bytes.fromhex(first.stdout))
    got = pickle.loads(bytes.fromhex(second.stdout))
    for a, b in zip(got, want):
        assert a.equals(b)
        for col in a:
            assert a[col].to_numpy().tobytes() == b[col].to_numpy().tobytes()


def test_a_buffer_is_named_by_its_digest_however_many_are_hashed_together():
    """The writer hashes an entry's buffers on several threads at once; each
    name is the buffer's own digest, whatever was hashed beside it."""
    store = _chunk_store()
    rng = np.random.default_rng(17)
    arrays = [rng.random(n) for n in (200_000, 150_001, 2_000_000, 300_000, 131_072)]
    views = [store.as_bytes(a) for a in arrays]
    assert store.digests(views) == [bulk_digest_module.bulk_digest(v) for v in views]
