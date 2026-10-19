"""The RAM tier copies a polars result when it stores it.

It kept a big one as a clone, with the digest of what it held, and a hit
whose clone no longer hashed so was a miss: a frame made over a numpy array
(``pl.Series(a)``) holds the array's memory, and a write to ``a`` changed
the clone too. A caller writing the array right after each call had the
function run on every call. Polars does not say which memory it allocated
itself, so every frame is copied at its store unless the tier made it
(`memory_backend._store_polars`): the caller's writes never reach the
entry, and nothing is hashed at the store or at a hit.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import content_hashers
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

from .test_a_polars_frame_read_from_disk_is_kept_in_ram import MUTATORS

ROWS = 300_000  # over a MiB: was kept as a clone


def _frame():
    return pl.DataFrame({"x": np.arange(ROWS), "s": ["a", "bb", "ccc"] * (ROWS // 3)})


@pytest.fixture
def hashed(monkeypatch):
    calls = []
    real = content_hashers.hash_polars
    monkeypatch.setattr(content_hashers, "hash_polars", lambda f: calls.append(f) or real(f))
    return calls


def test_a_frame_over_an_array_written_after_its_store_is_still_served():
    backend = InMemoryBackend()
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)  # polars holds a's memory itself
    backend.set("k", series, {"execution_time": 1.0, "copy_required": True})
    a[0] = -1.0  # reaches the series, not the entry
    _meta, hit = backend.get("k")
    assert hit is not None and hit[0] == 0.0
    assert backend._checked == {}


def test_nothing_is_hashed_at_the_store_or_a_hit(hashed):
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    _meta, again = backend.get("k")
    assert hashed == []
    assert hit.equals(_frame()) and again.equals(_frame())


@pytest.mark.parametrize("mutate", MUTATORS, ids=lambda f: f.__name__.strip("_"))
def test_what_the_caller_does_to_its_frame_or_a_hit_never_comes_back(mutate):
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
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


def test_the_disk_tier_gets_the_values_from_before_the_write(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    a = np.arange(ROWS, dtype=np.float64)
    series = pl.Series("x", a)
    metadata = {"execution_time": 5.0, "copy_required": True}
    tiers.set("k", series, metadata)
    a[0] = -1.0
    assert "DISK" in metadata["storage"]
    tiers.backends[0].delete("k")
    _meta, read = tiers.get("k")
    tiers.shutdown()
    assert read[0] == 0.0
