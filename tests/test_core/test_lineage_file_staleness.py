"""Regression: the lineage short-circuit must not return stale results after an
upstream file change.

When a cached function's result is passed to another cached function, the
downstream keys on the upstream's ``_cash_lineage_hash`` instead of re-hashing
the (possibly large) value. That hash was the producer's cache key, which omits
file-dependency state (files invalidate via a freshness re-stat, not via the
key). So after an upstream file changed, the producer recomputed but its new
output carried the SAME lineage hash as the old one - and the downstream
returned a stale wrong answer. The lineage hash now folds in a fingerprint of
the files the producer read.
"""
from __future__ import annotations

import time

import pytest

from cash import Cash, FileBackend


def test_downstream_recomputes_after_upstream_file_change(tmp_path):
    pd = pytest.importorskip("pandas")
    csv = tmp_path / "data.csv"
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache
    def load(path):
        return pd.read_csv(path)

    calls = {"n": 0}

    @c.cache
    def summarize(df):
        calls["n"] += 1
        return int(df["v"].sum())

    pd.DataFrame({"v": [1, 2, 3]}).to_csv(csv, index=False)
    assert summarize(load(str(csv))) == 6

    time.sleep(0.02)
    pd.DataFrame({"v": [10, 20, 30]}).to_csv(csv, index=False)   # upstream file changed

    # load recomputes (file dep); summarize must see the NEW data, not stale 6.
    result = summarize(load(str(csv)))
    assert result == 60, "downstream returned a stale lineage-keyed result"
    assert calls["n"] == 2


def test_lineage_survives_disk_restore(tmp_path):
    """The lineage hash is a plain attribute that doesn't survive pickling, so a
    disk-restored value used to lose it and the downstream fell back to content
    hashing under a different key (an extra recompute). It is now re-attached on
    the hit path, so the downstream hits across the compute/restore boundary."""
    pd = pytest.importorskip("pandas")
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / "cache")))

    @c.cache
    def load(n):
        return pd.DataFrame({"v": range(n)})

    calls = {"n": 0}

    @c.cache
    def total(df):
        calls["n"] += 1
        return int(df["v"].sum())

    d1 = load(5)                       # freshly computed
    assert hasattr(d1, "_cash_lineage_hash")
    total(d1)

    d2 = load(5)                       # restored from disk
    assert hasattr(d2, "_cash_lineage_hash"), "lineage hash lost on disk restore"
    total(d2)

    assert calls["n"] == 1, "downstream recomputed across the compute/restore boundary"


def test_lineage_short_circuit_still_hits_in_memory(tmp_path):
    """The fix must not break the lineage optimisation for in-memory values
    with no file dependency."""
    pd = pytest.importorskip("pandas")
    from cash import InMemoryBackend
    c = Cash(backend=InMemoryBackend())

    @c.cache
    def gen(n):
        return pd.DataFrame({"v": range(n)})

    calls = {"n": 0}

    @c.cache
    def total(df):
        calls["n"] += 1
        return int(df["v"].sum())

    df = gen(5)
    total(df)
    total(df)                      # same in-memory object -> lineage hit
    assert calls["n"] == 1
