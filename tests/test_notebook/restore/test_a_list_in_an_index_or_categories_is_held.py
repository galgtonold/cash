"""An object a frame holds in an object index, a MultiIndex level or a
categorical column's categories is one it holds, as one in an object
column is.

``g = pd.DataFrame(..., index=pd.Index(list(f['o']), dtype=object))``:
``g.index[4].append(9)`` writes into the list ``f['o'][4]`` is. Only the
column blocks were read, so ``f`` and ``g`` were restored apart, and the
next Run All left ``f['o'][4]`` at ``[4]`` where plain Python gives
``[4, 9]``.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import _pandas_objects

pd = pytest.importorskip("pandas")


def _holds(value, item) -> bool:
    return any(held is item for held in _pandas_objects(value))


def test_an_object_index_label():
    lists = [[i] for i in range(5)]
    g = pd.DataFrame({"v": range(5)}, index=pd.Index(lists, dtype=object))
    assert _holds(g, lists[4])
    assert _holds(g["v"], lists[4])


class Tag:
    """A label that hashes by identity and holds a list it can change."""

    def __init__(self, i):
        self.items = [i]


def test_a_multiindex_level():
    a, b = Tag(1), Tag(2)
    s = pd.Series([1, 2], index=pd.MultiIndex.from_arrays([pd.Index([a, b], dtype=object), [0, 1]]))
    assert _holds(s, a)


def test_a_categorical_category():
    a, b = Tag(1), Tag(2)
    assert _holds(pd.Series(pd.Categorical([a, b, a])), a)
    assert _holds(pd.DataFrame({"c": pd.Categorical([a, b])}), b)


def test_a_plain_index_adds_nothing():
    """The control: numbers and text in an index are values, not held."""
    assert _pandas_objects(pd.DataFrame({"v": [1, 2]}, index=["a", "b"])) == []
