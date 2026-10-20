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
is its own copy, and goes to the writer as one.
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


def _tagged_table(n: int = 5) -> pd.DataFrame:
    tag = Tag("x")
    tag.note = "fresh"
    # The categorical column keeps the table from being frozen: copied deep.
    return pd.DataFrame({"c": pd.Categorical(["a"] * n), "t": pd.Series([tag] * n, dtype=object)})


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


NOT_OWN = {
    "table copied deep, with shared cells": _tagged_table,
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
    table["t"].iloc[0].note = "EDITED"


LATER = {
    "a helper's closure": (_make_helper, lambda h: h()),
    "copied by deepcopy": (TrustsItself, lambda v: v.steps.append(1)),
    "table copied deep, with shared cells": (_tagged_table, _edit_a_cell),
}


@pytest.mark.parametrize("name", LATER)
def test_a_later_persist_of_a_value_that_may_share_a_part_is_not_written(name):
    """``h, peek = make_helper()``: deepcopy hands back the helpers
    themselves, closure and generator with them. Written once a later cell
    made it worth its disk, the entry held the generator as the draws below
    had left it, and after a restart the next draw was not plain Python's.
    A table pandas copied deep shares its object cells the same way."""
    make, edit = LATER[name]
    tiers, disk = _tiers()
    made = make()
    tiers.set("k", {"variables": {"v": made}}, dict(_NOTEBOOK))
    edit(made)
    assert not tiers.persist_from_memory("k", rebuild_seconds=5.0)
    assert disk.calls == []
    assert tiers.backends[0].peek_entry("k", value=False)[0]["persist_skipped"] == "shared"
    assert tiers.late_writes_refused == 1
