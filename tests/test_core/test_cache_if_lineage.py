"""A cache_if-rejected value must not carry a lineage hash.

The lineage hash a value carries points downstream at the producer's cache
entry. When cache_if rejects a result, nothing is stored - so tagging the
returned value with a lineage hash would reference an entry that does not
exist. A downstream cached function would then key on a lineage hash with no
backing entry. Lineage is now attached only when the value is actually stored.
"""
from __future__ import annotations

import tempfile

import pytest

from cash import Cash, FileBackend, InMemoryBackend


class Result:
    def __init__(self, payload):
        self.payload = payload


def test_rejected_value_has_no_lineage_hash():
    c = Cash(backend=InMemoryBackend())

    @c.cache(cache_if=lambda r: False)        # never store
    def produce(n):
        return Result(list(range(n)))

    r = produce(5)
    assert not hasattr(r, "_cash_lineage_hash")


def test_accepted_value_still_has_lineage_hash():
    c = Cash(backend=InMemoryBackend())

    @c.cache(cache_if=lambda r: True)         # always store
    def produce(n):
        return Result(list(range(n)))

    r = produce(5)
    assert hasattr(r, "_cash_lineage_hash")


def test_downstream_recomputes_for_rejected_upstream():
    """With the upstream rejected (uncached), the downstream must content-hash
    the value rather than key on a dangling lineage hash - and still be correct
    across recomputes."""
    pd = pytest.importorskip("pandas")
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    @c.cache(cache_if=lambda df: False)       # upstream never cached
    def load(n):
        return pd.DataFrame({"v": range(n)})

    calls = {"n": 0}

    @c.cache
    def total(df):
        calls["n"] += 1
        return int(df["v"].sum())

    assert total(load(5)) == 10
    assert total(load(5)) == 10               # equal content -> downstream hit by content
    assert calls["n"] == 1
    assert total(load(6)) == 15               # different content -> recompute
    assert calls["n"] == 2
