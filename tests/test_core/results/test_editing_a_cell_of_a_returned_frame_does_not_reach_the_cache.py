"""Editing a list or dict inside a returned frame never changes the next hit.

The RAM tier copied a frame shallowly (under copy-on-write that already
isolates the frame's own data), but neither a shallow nor a deep pandas copy
copies the Python objects in an object column. A list in a cell was one
object shared by the caller, the entry and every later hit:
``load()["tags"].iloc[0].append("X")`` and the next ``load()`` returned the
appended list. Frames holding such cells are now copied with their cells.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


@dataclasses.dataclass
class Report:
    table: object
    n: int


#: How each shape wraps the frame, and how to get the frame back out.
SHAPES = {
    "frame": (lambda r: r, lambda r: r),
    "frame in a dict": (lambda r: {"df": r}, lambda r: r["df"]),
    "frame in a list": (lambda r: [r, 1], lambda r: r[0]),
    "frame in a tuple": (lambda r: (r, 1), lambda r: r[0]),
    "frame in a dataclass": (lambda r: Report(r, 2), lambda r: r.table),
    "frame in an object": (lambda r: types.SimpleNamespace(table=r), lambda r: r.table),
    "frame in a tuple in a list": (lambda r: [(r, "summary")], lambda r: r[0][0]),
    "frame in a nested tuple": (lambda r: ((r,),), lambda r: r[0][0]),
    "frame in a deep dict": (
        lambda r: {"a": {"b": {"c": {"d": {"e": {"f": r}}}}}},
        lambda r: r["a"]["b"]["c"]["d"]["e"]["f"],
    ),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("edited", ["miss", "hit"])
def test_an_edited_cell_does_not_reach_later_hits(disk_cash, shape, edited):
    wrap, unwrap = SHAPES[shape]

    @disk_cash.cache
    def load():
        return wrap(pd.DataFrame({"tags": [["a"], ["b"]], "rec": [{"k": 1}, {"k": 2}], "n": [1.0, 2.0]}))

    first = load()
    frame = unwrap(first if edited == "miss" else load())
    frame["tags"].iloc[0].append("X")
    frame["rec"].iloc[1]["k"] = 99

    frame = unwrap(load())
    assert frame["tags"].tolist() == [["a"], ["b"]]
    assert frame["rec"].tolist() == [{"k": 1}, {"k": 2}]
    assert load.cache_info()["hits"] >= 1


@pytest.mark.parametrize("edited", ["miss", "hit"])
@pytest.mark.parametrize("where", ["alone", "in a list in a dict"])
def test_an_edited_cell_of_a_series_does_not_reach_later_hits(disk_cash, edited, where):
    @disk_cash.cache
    def load():
        series = pd.Series([["a"], ["b"]])
        return series if where == "alone" else {"k": [series]}

    first = load()
    target = first if edited == "miss" else load()
    (target if where == "alone" else target["k"][0]).iloc[0].append("X")
    again = load()
    assert (again if where == "alone" else again["k"][0]).tolist() == [["a"], ["b"]]
    assert load.cache_info()["hits"] >= 1
