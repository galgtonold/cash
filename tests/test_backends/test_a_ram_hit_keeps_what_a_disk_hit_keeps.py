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


def test_a_user_frame_subclass_gets_its_cells_copied():
    pd = pytest.importorskip("pandas")

    class Orders(pd.DataFrame):
        @property
        def _constructor(self):
            return Orders

    first, second = _hits(Orders({"id": [1, 2], "items": [["apple"], ["pear"]]}))
    assert type(first) is Orders
    first["items"].iloc[0].append("plum")
    assert second["items"].iloc[0] == ["apple"]


class _Sensor:
    """Hashable by name, with mutable calibration."""

    def __init__(self, name):
        self.name = name
        self.offsets = [0.0]

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, _Sensor) and other.name == self.name


@pytest.mark.parametrize(
    "make, first_label",
    [
        (lambda pd, a, b: pd.Series(pd.Categorical([a, b, a])), lambda got: got.cat.categories[0]),
        (lambda pd, a, b: pd.Series([1.0, 2.0], index=pd.Index([a, b], dtype=object)), lambda got: got.index[0]),
        (lambda pd, a, b: pd.DataFrame([[1, 2]], columns=pd.Index([a, b], dtype=object)), lambda got: got.columns[0]),
        (
            lambda pd, a, b: pd.Series([1, 2], index=pd.MultiIndex.from_tuples([(a, 1), (b, 2)])),
            lambda got: got.index[0][0],
        ),
        (
            lambda pd, a, b: pd.DataFrame({"s": pd.Categorical([a, b])}, index=pd.CategoricalIndex([b, a])),
            lambda got: got.index[0],
        ),
    ],
    ids=["categories", "index", "columns", "multiindex", "categorical-index"],
)
def test_objects_in_labels_and_categories_are_copied(make, first_label):
    pd = pytest.importorskip("pandas")
    first, second = _hits(make(pd, _Sensor("a"), _Sensor("b")))
    first_label(first).offsets.append(0.5)
    assert first_label(second).offsets == [0.0]
