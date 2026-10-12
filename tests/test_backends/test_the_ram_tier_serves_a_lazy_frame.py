"""A polars LazyFrame result is held by the RAM tier as a clone.

The tier cloned a polars DataFrame or Series but pickled a LazyFrame, its
plan and the data the plan holds, on every store and every hit: about 2 s
a hit for a plan over an 80 MB frame, where a clone takes well under a
millisecond. A plan has no in-place API, so the clone shares it safely.
"""

from __future__ import annotations

import time

import pytest

pl = pytest.importorskip("polars")

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


def _source():
    return pl.DataFrame({"x": list(range(10_000)), "g": [i % 5 for i in range(10_000)]})


def _plan(df):
    return df.lazy().filter(pl.col("x") > 10).group_by("g").agg(pl.col("x").sum()).sort("g")


def _tiers(path) -> TieredBackend:
    return TieredBackend([InMemoryBackend(), FileBackend(str(path), flush_interval=0)])


def test_a_ram_hit_gives_the_plan_back():
    backend = InMemoryBackend()
    backend.set("k", _plan(_source()), {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    assert isinstance(hit, pl.LazyFrame)
    assert hit.collect().equals(_plan(_source()).collect())


def test_editing_the_frame_the_plan_was_made_from_does_not_reach_the_cache():
    backend = InMemoryBackend()
    df = _source()
    backend.set("k", df.lazy(), {"execution_time": 1.0, "copy_required": True})
    df[0, "x"] = -1
    df.insert_column(0, pl.Series("new", [0] * df.height))
    _meta, hit = backend.get("k")
    assert hit.collect().equals(_source())
    hit.collect()[1, "x"] = -2  # a frame collected from a hit is the caller's
    assert backend.get("k")[1].collect().equals(_source())


def test_a_plan_read_from_disk_is_kept_in_ram_on_its_first_read(tmp_path):
    writer = _tiers(tmp_path)
    writer.set("k", _plan(_source()), {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = _tiers(tmp_path)
    meta, first = reader.get("k")
    assert meta["source"] == "DISK"
    meta, second = reader.get("k")
    reader.shutdown()
    assert meta["source"] == "RAM"
    assert first.collect().equals(_plan(_source()).collect())
    assert second.collect().equals(_plan(_source()).collect())


@pytest.mark.perf
def test_a_ram_hit_of_a_plan_over_a_large_frame_costs_no_pickle():
    big = pl.DataFrame({f"c{i}": pl.int_range(2_000_000, eager=True) for i in range(5)})  # 80 MB
    backend = InMemoryBackend()
    backend.set("k", big.lazy().select(pl.all() * 2), {"execution_time": 1.0, "copy_required": True})
    started = time.perf_counter()
    for _ in range(5):
        _meta, hit = backend.get("k")
    per_hit = (time.perf_counter() - started) / 5
    assert hit is not None
    assert per_hit < 0.05, f"{per_hit * 1000:.1f} ms a hit"  # pickled: ~2,000 ms
