"""A RAM hit hands back what a disk hit does: an independent copy that is
otherwise the stored result as it was.

A disk hit is a pickle round trip, which keeps a value's attributes, flags,
sharing and read-only arrays, and copies every Python object in it. A RAM hit
that kept less made the answer depend on which tier served it.
"""

from __future__ import annotations

import datetime

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a test dependency
    np = None

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


class _Settings:
    """Treated as immutable by its author: copying it returns itself."""

    def __init__(self):
        self.layers = [64]

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


@pytest.mark.parametrize("wrap", [lambda s: s, lambda s: {"cfg": s}, lambda s: [s, 1]], ids=["bare", "dict", "list"])
def test_a_deepcopy_that_returns_self_does_not_share_the_entry(wrap):
    settings = _Settings()
    first, second = _hits(wrap(settings))
    unwrap = (lambda v: v) if type(first) is _Settings else (lambda v: v["cfg"] if type(v) is dict else v[0])
    assert unwrap(first) is not settings
    unwrap(first).layers.append(128)
    assert unwrap(second).layers == [64]


class _Day(datetime.date):
    pass


if np is not None:

    class _Quantity(np.ndarray):
        """Units carried as an attribute, without ``__array_finalize__``."""


def test_attributes_of_a_c_type_subclass_survive_a_hit():
    if np is None:
        pytest.skip("numpy not installed")
    q = np.array([1.0, 2.0]).view(_Quantity)
    q.unit = "kg"
    d = _Day(2024, 1, 31)
    d.holiday = True
    for got_q, got_d in _hits((q, d)):
        assert type(got_q) is _Quantity and got_q.unit == "kg"
        assert type(got_d) is _Day and got_d.holiday is True


@pytest.mark.parametrize(
    "wrap, unwrap",
    [
        (lambda a: {"lut": a, "n": 5}, lambda v: v["lut"]),
        (lambda a: (a, 5), lambda v: v[0]),
        (lambda a: [a], lambda v: v[0]),
        (lambda a: a, lambda v: v),
    ],
    ids=["dict", "tuple", "list", "bare"],
)
def test_a_read_only_array_comes_back_read_only(wrap, unwrap):
    if np is None:
        pytest.skip("numpy not installed")
    lut = np.arange(5.0)
    lut.flags.writeable = False
    writable = np.arange(3.0)
    for got in _hits(wrap(lut)):
        assert unwrap(got).flags.writeable is False
        assert unwrap(got).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    for got in _hits(wrap(writable)):
        unwrap(got)[0] = 9.0  # still writable, and a copy
    assert writable[0] == 0.0


@pytest.mark.parametrize("cell_first", [False, True])
def test_a_cell_returned_beside_its_frame_stays_the_frames(cell_first):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"tags": [["a"], ["b"]]})
    value = (df["tags"].iloc[0], df) if cell_first else (df, df["tags"].iloc[0])
    for got in _hits(value):
        frame, first = (got[1], got[0]) if cell_first else got
        first.append("z")
        assert frame["tags"].iloc[0] == ["a", "z"]
        assert first is not df["tags"].iloc[0]
    assert df["tags"].iloc[0] == ["a"]


class _Node:
    def __init__(self, v, nxt):
        self.v = v
        self.next = nxt


def test_a_long_chain_of_objects_is_copied():
    """deepcopy recursed a few frames per link and gave up at a few hundred
    links; the store was refused, though pickle (the disk tier) copies it."""
    head = None
    for i in range(2000):
        head = _Node(i, head)
    first, second = _hits(head)
    assert first is not head and first is not second
    n, node = 0, first
    while node is not None:
        n, node = n + 1, node.next
    assert n == 2000 and first.v == 1999


@pytest.mark.parametrize("as_frame", [False, True])
def test_the_objects_in_a_polars_object_column_are_copied(as_frame):
    """No disk tier can hold an Object column, so the RAM entry is the only
    copy: a clone shared its objects with every hit."""
    pl = pytest.importorskip("polars")
    s = pl.Series("o", [[1], [2]], dtype=pl.Object)
    value = pl.DataFrame({"o": s, "n": [1, 2]}) if as_frame else s
    first, second = _hits(value)
    column = (lambda v: v["o"]) if as_frame else (lambda v: v)
    column(first)[0].append(9)
    assert column(second)[0] == [1]
    assert s[0] == [1]
    if as_frame:
        assert first.columns == ["o", "n"] and first["n"].to_list() == [1, 2]
