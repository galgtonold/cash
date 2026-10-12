"""A polars result is held by the RAM tier and served from there.

The tier sent every type named ``DataFrame`` or ``Series`` to pandas'
``copy(deep=...)``, which a polars frame does not have. The store was refused,
so every hit in the same process read the entry back from disk (240 ms for an
88 MB frame, against about 1 ms from RAM) and logged "Failed to promote ... to
tier 0" each time.
"""

import logging

import pytest

pl = pytest.importorskip("polars")

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

from .test_a_polars_frame_read_from_disk_is_kept_in_ram import MUTATORS


def _frame():
    return pl.DataFrame({"x": list(range(1000)), "s": ["a"] * 1000})


def test_the_ram_tier_stores_a_polars_frame_when_a_copy_is_required():
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    assert hit is not None and hit.equals(df)


def test_a_polars_hit_is_served_from_ram_without_a_warning(tmp_path, caplog):
    backend = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    df = _frame()
    with caplog.at_level(logging.WARNING):
        backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
        meta, hit = backend.get("k")
        _meta, again = backend.get("k")
    backend.shutdown()
    assert meta["source"] == "RAM"
    assert hit.equals(df) and again.equals(df)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text


def test_editing_a_polars_hit_does_not_reach_the_cache():
    backend = InMemoryBackend()
    df = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    df[0, "x"] = -1  # the caller edits their frame after the store
    _meta, hit = backend.get("k")
    assert hit[0, "x"] == 0
    hit[1, "x"] = -2  # and the frame a hit handed back
    _meta, again = backend.get("k")
    assert again[1, "x"] == 1


@pytest.mark.parametrize("mutate", MUTATORS, ids=lambda f: f.__name__.strip("_"))
def test_what_the_caller_does_to_a_ram_hit_or_the_stored_frame_stays_out(mutate):
    backend = InMemoryBackend()
    df = _frame()
    kept = _frame()
    backend.set("k", df, {"execution_time": 1.0, "copy_required": True})
    mutate(df)
    _meta, hit = backend.get("k")
    assert hit.equals(kept)
    mutate(hit)
    _meta, again = backend.get("k")
    assert again.equals(kept)


def test_a_series_edited_in_place_does_not_reach_the_cache():
    backend = InMemoryBackend()
    s = pl.Series("x", [1, 2, 3])
    backend.set("k", s, {"execution_time": 1.0, "copy_required": True})
    s.scatter(0, 9)
    s.append(pl.Series("x", [4]))
    _meta, hit = backend.get("k")
    assert hit.to_list() == [1, 2, 3]
    hit.scatter(1, 9)
    assert backend.get("k")[1].to_list() == [1, 2, 3]
