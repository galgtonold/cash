"""Keying an object frame by frame keeps every answer it gave when pickled whole.

These pass on the pickled path too: they pin that opening objects up
(`object_hashing.holds_content_data`) still sees every edit, still shares an entry
between equal objects, and still caches an object that reaches itself.
The speed-up itself is pinned in test_objects_holding_frames.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

import cash

pytestmark = pytest.mark.core


class Bundle:
    def __init__(self, n=1000):
        self.name = "sales"
        self.frames = {"a": pd.DataFrame({"x": range(n)}), "b": pd.DataFrame({"y": range(n)})}
        self.extra = [pd.Series(range(n))]


class Node:
    """Holds a frame and a reference back to itself."""

    def __init__(self):
        self.frame = pd.DataFrame({"x": range(10)})
        self.me = self


@pytest.fixture
def c(tmp_path):
    return cash.Cash(cache_dir=str(tmp_path / "cache"))


def test_equal_objects_share_an_entry_and_edits_are_seen(c):
    @c.cache
    def total(b):
        return int(b.frames["a"]["x"].sum() + b.frames["b"]["y"].sum() + b.extra[0].sum())

    first = Bundle()
    assert total(first) == 3 * 499500
    assert total(Bundle()) == 3 * 499500
    assert total.cache_info()["hits"] == 1

    first.frames["a"].loc[0, "x"] = 1000
    assert total(first) == 3 * 499500 + 1000
    first.frames["b"] = pd.DataFrame({"y": [1]})
    assert total(first) == 499500 + 1000 + 1 + 499500
    first.extra[0].iloc[1] = 0
    assert total(first) == 499500 + 1000 + 1 + 499500 - 1


def test_self_holding_frames_is_keyed_by_them(c):
    class Holder(Bundle):
        @c.cache
        def rows(self):
            return len(self.frames["a"])

    h = Holder()
    assert h.rows() == 1000
    h.frames["a"] = h.frames["a"].iloc[:10]
    assert h.rows() == 10


def test_an_object_that_reaches_itself_still_caches(c):
    runs = []

    @c.cache
    def f(node):
        runs.append(1)
        return int(node.frame["x"].sum())

    n = Node()
    assert f(n) == 45
    assert f(n) == 45
    assert len(runs) == 1
