"""The disk entry is what the call returned, whatever the caller does after.

A store hands the persistent tiers the RAM tier's own copy, which they
serialize later, on the writer thread (``private``). That is safe only for a
copy that keeps nothing of the caller's. A table pandas copies deep (one
that cannot be frozen: a categorical or nullable column, a subclass) keeps
the Python objects of its object cells, and ``deepcopy`` trusts a class's
``__deepcopy__``: a part the caller still holds was pickled with whatever the
caller did to it while the write waited in the queue. Such a value is
serialized on the calling thread, as the caller's value always was. A table
whose columns and labels the RAM tier copies in full (a nullable column,
Python strings: arrays and labels copied, no Python object that can change)
is its own copy, and goes to the writer as one; so is one whose object cells,
labels or categories hold a subclass of str with attributes (a tag carrying
a note): pandas calls those strings, and the RAM tier copies them through
pickle.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from tests.test_backends.test_a_miss_hands_the_disk_write_the_ram_copy import _KEEP, _NOTEBOOK, _tiers


class Tag(str):
    """A str whose instances carry a writable attribute: pandas treats a
    column of them as text and shares the cells in a deep copy."""


def _tag(note: str = "fresh") -> Tag:
    tag = Tag("x")
    tag.note = note
    return tag


def _tagged_table(n: int = 5) -> pd.DataFrame:
    tag = _tag()
    # The categorical column keeps the table from being frozen: copied deep.
    return pd.DataFrame({"c": pd.Categorical(["a"] * n), "t": pd.Series([tag] * n, dtype=object)})


def _cells(table: pd.DataFrame) -> Tag:
    return table["t"].iloc[0]


def _label(table: pd.DataFrame) -> Tag:
    return table.index[0]


def _category(table: pd.DataFrame) -> Tag:
    return table["c"].cat.categories[0]


def _tagged_categories() -> pd.CategoricalDtype:
    # Given as an index: built from the values, the categories are plain str.
    return pd.CategoricalDtype(pd.Index([_tag(), "y"], dtype=object))


#: Where the tag sits, and how to reach it again.
TAGGED = {
    "cells, with a categorical": (_tagged_table, _cells),
    "cells, of a table of numbers": (
        lambda: pd.DataFrame({"x": np.arange(5.0), "t": pd.Series([_tag() for _ in range(5)], dtype=object)}),
        _cells,
    ),
    "labels": (lambda: pd.DataFrame({"x": np.arange(2.0)}, index=pd.Index([_tag(), _tag()], dtype=object)), _label),
    "categories": (lambda: pd.DataFrame({"c": pd.Categorical(["y"], dtype=_tagged_categories())}), _category),
}


class Subclass(pd.DataFrame):
    @property
    def _constructor(self):
        return Subclass


class TrustsItself:
    """deepcopy hands back the object itself; pickle refuses the lambda."""

    def __init__(self):
        self.steps = []
        self.fn = lambda: None

    def __deepcopy__(self, memo):
        return self


COPIED_IN_FULL = {
    "nullable column": lambda: pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")}),
    "python strings": lambda: pd.DataFrame(
        {"s": pd.Series(["a", None, "c"], dtype="string[python]")}, index=pd.Index(["x", "y", "z"], dtype=object)
    ),
}


@pytest.mark.parametrize("name", COPIED_IN_FULL)
def test_a_table_the_ram_tier_copies_in_full_goes_to_disk_as_that_copy(name):
    tiers, disk = _tiers()
    value = COPIED_IN_FULL[name]()
    tiers.set("k", value, dict(_KEEP))
    assert tiers.backends[0].holds_own_copy("k")
    ((_key, written, private),) = disk.calls
    assert private and disk.copied == [False] and written is not value
    column = value.columns[0]
    value[column].array[0] = value[column].iloc[2]  # writes in place
    if not isinstance(value.index, pd.RangeIndex):
        value.index.array[0] = "edited"  # a deep pandas copy would keep these labels
    pd.testing.assert_frame_equal(written, COPIED_IN_FULL[name]())


class Unpicklable:
    """A cell pickle refuses, and no copy but ``deepcopy``'s reaches."""

    def __init__(self):
        self.steps = []
        self.fn = lambda: None


def _unpicklable_cells() -> pd.DataFrame:
    return pd.DataFrame({"c": pd.Categorical(["a", "b"]), "u": pd.Series([Unpicklable(), Unpicklable()], dtype=object)})


NOT_OWN = {
    "table with cells pickle refuses": _unpicklable_cells,
    "subclass": lambda: Subclass({"a": np.arange(3.0)}),
    "copied by deepcopy": TrustsItself,
}


@pytest.mark.parametrize("name", NOT_OWN)
def test_a_value_that_may_share_a_part_goes_to_disk_as_the_caller_gave_it(name):
    tiers, disk = _tiers()
    value = NOT_OWN[name]()
    tiers.set("k", value, dict(_KEEP))
    ((_key, written, private),) = disk.calls
    assert not private and written is value
    assert not tiers.backends[0].holds_own_copy("k")


OWN = {
    "array": lambda: np.arange(1000.0),
    "records": lambda: [{"id": i, "tags": [i]} for i in range(10_000)],
    "small dict": lambda: {"rows": list(range(100))},
    "object copied by pickle": lambda: {"when": pd.Timestamp("2024-01-01"), "arr": np.zeros(3), "s": {1, 2}},
}
if frame_sharing.enabled():
    OWN["frozen table"] = lambda: pd.DataFrame({"x": np.arange(100.0)})
    OWN["frozen table in a notebook entry"] = lambda: {"variables": {"df": pd.DataFrame({"x": np.arange(9.0)})}}


@pytest.mark.parametrize("name", OWN)
def test_a_value_that_is_cash_s_own_still_goes_to_disk_as_the_ram_copy(name):
    """The positive control: what is copied in full keeps its free write."""
    tiers, disk = _tiers()
    value = OWN[name]()
    tiers.set("k", value, dict(_KEEP))
    ((_key, written, private),) = disk.calls
    assert private and written is not value


@pytest.mark.parametrize("name", TAGGED)
def test_a_tag_in_a_table_is_copied_not_shared(name):
    """``infer_dtype`` calls a subclass of str "string", by ``isinstance``:
    such cells were taken for immutable, the table frozen or copied deep,
    and the tag object itself shared with the caller. A note the caller set
    on it after the call reached the RAM hit and a later disk write."""
    make, reach = TAGGED[name]
    tiers, disk = _tiers()
    table = make()
    tiers.set("k", {"variables": {"v": table}}, dict(_NOTEBOOK))
    reach(table).note = "EDITED"
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    ((_key, written, _private),) = disk.calls
    assert reach(written["variables"]["v"]).note == "fresh"
    _metadata, hit = tiers.backends[0].get("k")
    assert reach(hit["variables"]["v"]).note == "fresh"
    assert tiers.late_writes_refused == 0


def test_an_edit_made_while_the_write_waits_stays_out_of_the_disk_entry(tmp_path):
    disk = FileBackend(str(tmp_path / "cache"))
    tiers = TieredBackend([InMemoryBackend(), disk])
    tiers.set("warm", np.zeros(3), dict(_KEEP))
    disk.get("warm")  # the writer is up
    gate = threading.Event()
    disk._writes.submit("gate", gate.wait, 30)  # an earlier write, still running
    try:
        table = _tagged_table()
        tiers.set("k", table, dict(_KEEP))
        table["t"].iloc[0].note = "edited by the caller after the call"
    finally:
        gate.set()
    _metadata, restored = disk.get("k")
    assert restored["t"].iloc[0].note == "fresh"


def _make_helper():
    g = np.random.default_rng(0)

    def helper():
        return float(g.normal())

    return helper


def _edit_a_cell(table: pd.DataFrame) -> None:
    table["u"].iloc[0].steps.append(1)


LATER = {
    "a helper's closure": (_make_helper, lambda h: h()),
    "copied by deepcopy": (TrustsItself, lambda v: v.steps.append(1)),
    "table with cells pickle refuses": (_unpicklable_cells, _edit_a_cell),
}


@pytest.mark.parametrize("name", LATER)
def test_a_later_persist_of_a_value_that_may_share_a_part_is_not_written(name):
    """``h, peek = make_helper()``: deepcopy hands back the helpers
    themselves, closure and generator with them. Written once a later cell
    made it worth its disk, the entry held the generator as the draws below
    had left it, and after a restart the next draw was not plain Python's.
    A table whose cells pickle refuses shares them the same way."""
    make, edit = LATER[name]
    tiers, disk = _tiers()
    made = make()
    tiers.set("k", {"variables": {"v": made}}, dict(_NOTEBOOK))
    edit(made)
    assert not tiers.persist_from_memory("k", rebuild_seconds=5.0)
    assert disk.calls == []
    assert tiers.backends[0].peek_entry("k", value=False)[0]["persist_skipped"] == "shared"
    assert tiers.late_writes_refused == 1


class Box:
    def __init__(self, items):
        self.items = items

    def __eq__(self, other):
        return type(other) is Box and other.items == self.items

    __hash__ = None


COPIED_WITH_CELLS = {
    "cells of a user class": (
        lambda: pd.DataFrame({"x": np.arange(3.0), "b": pd.Series([Box([i]) for i in range(3)], dtype=object)}),
        lambda v: v["b"].iloc[0].items,
    ),
    "list cells, with a categorical": (
        lambda: pd.DataFrame({"c": pd.Categorical(list("abc")), "l": pd.Series([[i] for i in range(3)], dtype=object)}),
        lambda v: v["l"].iloc[0],
    ),
    "dict cells, with a categorical": (
        lambda: pd.DataFrame(
            {"c": pd.Categorical(list("abc")), "d": pd.Series([{"k": [i]} for i in range(3)], dtype=object)}
        ),
        lambda v: v["d"].iloc[0]["k"],
    ),
    "a categorical": (lambda: pd.DataFrame({"c": pd.Categorical(list("ab") * 50), "x": np.arange(100.0)}), None),
}


@pytest.mark.parametrize("name", COPIED_WITH_CELLS)
def test_a_later_persist_of_a_table_copied_with_its_cells_is_written(name):
    """The positive control of the refusal: a table whose cells the RAM tier
    copied (through pickle, or a container at a time) holds nothing of the
    caller's, and a later write of it is the value as it was returned."""
    make, reach = COPIED_WITH_CELLS[name]
    tiers, disk = _tiers()
    table = make()
    tiers.set("k", {"variables": {"v": table}}, dict(_NOTEBOOK))
    if reach is not None:
        reach(table).append("EDITED")
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    ((_key, written, _private),) = disk.calls
    pd.testing.assert_frame_equal(written["variables"]["v"], make())
    assert tiers.late_writes_refused == 0


def test_a_refused_late_write_is_counted_in_cash_status(mock_shell):
    from cash import Cash
    from cash.notebook.ipython.magics import CashMagics

    tiers, _disk = _tiers()
    magics = CashMagics(mock_shell, Cash(backend=tiers, register_magic=False))
    tiers.set("k", {"variables": {"v": TrustsItself()}}, dict(_NOTEBOOK))
    assert magics.cash_status("dict")["cache_stats"]["late_writes_refused"] == 0
    assert not tiers.persist_from_memory("k", rebuild_seconds=5.0)

    assert magics.cash_status("dict")["cache_stats"]["late_writes_refused"] == 1


def test_explain_says_why_a_shared_value_stayed_in_ram():
    from cash.decorator.explain import not_persisted_reason

    reason = not_persisted_reason({"storage": ["RAM"], "persist_skipped": "shared"})
    assert reason is not None and "shares a part" in reason
