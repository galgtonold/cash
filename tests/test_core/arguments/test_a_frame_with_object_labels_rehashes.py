"""A frame whose index or columns hold Python objects is hashed afresh each call.

The content-hash memo trusts copy-on-write: a frame changes only by getting new
arrays. That holds for an object-dtype block, which the memo already refuses,
but it skipped the axes too: a Series indexed by ``Site`` objects was served
the first call's 15 after ``site.capacity = 100``, where plain Python gives 105.
An axis of strings, numbers or dates cannot change in place and keeps the memo.
Column labels were keyed by their ``repr`` alone, an object's address, so
the columns case was stale even without the memo.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing


class Site:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def test_an_edited_object_in_the_index_is_seen(cash):
    @cash.cache
    def total_capacity(load):
        return sum(site.capacity for site in load.index)

    a = Site("a", 10)
    load = pd.Series([0.5, 0.7], index=pd.Index([a, Site("b", 5)], dtype=object))
    assert total_capacity(load) == 15
    a.capacity = 100
    assert total_capacity(load) == 105


def test_an_edited_object_in_the_columns_is_seen(cash):
    @cash.cache
    def column_capacity(frame):
        return sum(site.capacity for site in frame.columns)

    a = Site("a", 1)
    frame = pd.DataFrame([[1.0, 2.0]], columns=pd.Index([a, Site("b", 2)], dtype=object))
    assert column_capacity(frame) == 3
    a.capacity = 50
    assert column_capacity(frame) == 52


def test_an_edited_object_in_a_multiindex_level_is_seen(cash):
    @cash.cache
    def level_capacity(load):
        return sum(site.capacity for site in load.index.levels[0])

    a = Site("a", 10)
    index = pd.MultiIndex.from_arrays([pd.Index([a, Site("b", 5)], dtype=object), [1, 2]])
    load = pd.Series([0.5, 0.7], index=index)
    assert level_capacity(load) == 15
    a.capacity = 100
    assert level_capacity(load) == 105


def test_labels_that_cannot_change_keep_the_memo():
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["a", "b"], dtype=object))
    assert arg_hashing._frame_memory(frame, None) is not None
    frame.index = pd.Index([Site("a", 1), Site("b", 2)], dtype=object)
    assert arg_hashing._frame_memory(frame, None) is None
