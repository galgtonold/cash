"""A RAM hit hands back what a disk hit does: an independent copy that is
otherwise the stored result as it was.

A disk hit is a pickle round trip, which keeps a value's attributes, flags,
sharing and read-only arrays, and copies every Python object in it. A RAM hit
that kept less made the answer depend on which tier served it.
"""

from __future__ import annotations

import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]


def _hits(value, n: int = 2) -> list:
    """*value* stored in a fresh RAM tier, and *n* hits of it."""
    b = InMemoryBackend()
    b.set("k", value, {"copy_required": True})
    return [b.get("k")[1] for _ in range(n)]


def test_a_series_of_lists_keeps_its_attrs_and_flags():
    pd = pytest.importorskip("pandas")
    s = pd.Series([["a"], ["b", "c"]], name="tags").set_flags(allows_duplicate_labels=False)
    s.attrs["source"] = "crm-export"
    for got in _hits(s):
        assert got.attrs == {"source": "crm-export"}
        assert got.flags.allows_duplicate_labels is False
        assert got.name == "tags"
        assert got.tolist() == [["a"], ["b", "c"]]
