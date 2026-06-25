"""Cross-mechanic staleness: where two caching features interact.

* iterator caching x file deps - a cached generator that reads a file must
  invalidate when the file changes (its reads happen lazily during chunk
  materialization, which must be inside the file-tracking scope).
* TTL x downstream lineage - a TTL'd value's identity isn't captured by its
  cache key, so it must NOT carry a lineage hash; otherwise a downstream cached
  function returns a stale result after the upstream's TTL refresh.
"""
from __future__ import annotations

import time

import pytest

from cash import Cash, FileBackend, InMemoryBackend


def test_cached_generator_invalidates_on_file_change(tmp_path):
    data = tmp_path / "lines.txt"
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache
    def stream(path):
        with open(path) as fh:
            for line in fh:
                yield line.strip()

    data.write_text("a\nb\nc\n")
    assert list(stream(str(data))) == ["a", "b", "c"]
    assert list(stream(str(data))) == ["a", "b", "c"]   # replay from chunks

    time.sleep(0.02)
    data.write_text("x\ny\nz\n")
    assert list(stream(str(data))) == ["x", "y", "z"], "stale cached generator"


def test_generator_manifest_records_file_dep(tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("1\n2\n")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache
    def stream(path):
        with open(path) as fh:
            yield from fh

    list(stream(str(data)))
    raw_meta, _ = c.backend.get(stream.explain(str(data)).cache_key)
    recorded = (raw_meta or {}).get("auto_file_deps") or {}
    assert any(p.endswith("d.txt") for p in recorded), recorded


def test_ttl_value_has_no_lineage_hash(tmp_path):
    pd = pytest.importorskip("pandas")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache(ttl=60, assume_safe=True)
    def fetch():
        return pd.DataFrame({"x": [1]})

    df = fetch()
    assert not hasattr(df, "_cash_lineage_hash"), "TTL'd value must not carry lineage"


def test_downstream_not_stale_after_ttl_refresh():
    pd = pytest.importorskip("pandas")
    c = Cash(backend=InMemoryBackend())
    state = {"v": 1}

    @c.cache(ttl=1, assume_safe=True)
    def fetch():
        return pd.DataFrame({"x": [state["v"]]})

    @c.cache
    def double(df):
        return int(df["x"].sum()) * 2

    assert double(fetch()) == 2
    state["v"] = 100
    time.sleep(1.1)                       # expire fetch's TTL
    assert int(fetch()["x"].sum()) == 100  # upstream refreshed
    assert double(fetch()) == 200, "downstream returned a stale lineage-keyed result"
