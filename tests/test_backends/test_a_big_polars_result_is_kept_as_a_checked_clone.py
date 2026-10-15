"""The RAM tier keeps a big polars result as a clone, checked on each hit.

It copied every buffer of a polars result it stored, because a frame made
over a numpy array (``pl.Series(a)``) holds the array's memory, and a clone
would change with ``a``. Copying two results of 1.2 GB took 7-9 s of a 15 s
cell. Now it keeps a clone with the digest of what it held, and a hit whose
clone no longer hashes so is a miss; a disk write of it is dropped the same
way.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

from .test_a_polars_frame_read_from_disk_is_kept_in_ram import MUTATORS

ROWS = 300_000  # over a MiB: kept as a clone


def _frame():
    return pl.DataFrame({"x": np.arange(ROWS), "s": ["a", "bb", "ccc"] * (ROWS // 3)})


def test_a_frame_over_an_array_written_after_its_store_is_a_miss():
    backend = InMemoryBackend()
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)  # polars holds a's memory itself
    backend.set("k", series, {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    assert hit is not None and hit[0] == 0.0
    a[0] = -1.0  # reaches the series, and a clone of it
    meta, hit = backend.get("k")
    assert meta is None and hit is None
    assert "k" not in backend._store


def test_a_store_copies_no_buffer_and_a_hit_serves_the_value(monkeypatch):
    from cash.backends import memory_backend

    copies = []
    owned = memory_backend._owned_polars
    monkeypatch.setattr(memory_backend, "_owned_polars", lambda frame: copies.append(frame) or owned(frame))
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    assert copies == []
    assert hit.equals(_frame())


@pytest.mark.parametrize("mutate", MUTATORS, ids=lambda f: f.__name__.strip("_"))
def test_what_the_caller_does_to_its_frame_or_a_hit_never_comes_back(mutate):
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    assert backend._checked["k"]  # kept as a clone, not copied
    mutate(df)
    _meta, hit = backend.get("k")
    assert hit.equals(_frame())
    mutate(hit)
    _meta, again = backend.get("k")
    assert again.equals(_frame())


def test_a_disk_write_whose_check_fails_writes_nothing(tmp_path):
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    value = np.arange(ROWS)
    disk.set("k", value, {"execution_time": 1.0}, private=True, check=lambda: False)
    disk.set("ok", value, {"execution_time": 1.0}, private=True, check=lambda: True)
    assert disk.get("k") == (None, None)
    _meta, kept = disk.get("ok")
    disk.shutdown()
    assert np.array_equal(kept, value)
    # The chunk the dropped write put is held by the one written after it.
    assert len(list((tmp_path / "c" / "chunks").iterdir())) == 1


def test_the_disk_tier_gets_the_clone_with_its_check(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)
    metadata = {"execution_time": 5.0, "copy_required": True}
    tiers.set("k", series, metadata)
    assert "DISK" in metadata["storage"]
    found = tiers.backends[0].checked_copy("k", tiers.backends[0]._store["k"][0])
    assert found is not None
    stored, check = found
    assert check()
    a[0] = -1.0
    assert not check()
    tiers.backends[0].delete("k")
    _meta, read = tiers.get("k")  # written before the change
    tiers.shutdown()
    assert read is None or read[0] == 0.0
