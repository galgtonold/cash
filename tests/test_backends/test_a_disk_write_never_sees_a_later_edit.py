"""The disk entry is what the call returned, whatever the caller does after.

A store hands the persistent tiers the RAM tier's own copy, which they
serialize later, on the writer thread (``private``). That is safe only for a
copy that keeps nothing of the caller's. A table pandas copies deep (one
that cannot be frozen: a categorical or nullable column, a subclass) keeps
the Python objects of its object cells, and ``deepcopy`` trusts a class's
``__deepcopy__``: a part the caller still holds was pickled with whatever the
caller did to it while the write waited in the queue. Such a value is
serialized on the calling thread, as the caller's value always was.
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


NOT_OWN = {
    "table copied deep, with shared cells": _tagged_table,
    "subclass": lambda: Subclass({"a": np.arange(3.0)}),
    "nullable column": lambda: pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")}),
    "copied by deepcopy": TrustsItself,
}


@pytest.mark.parametrize("name", NOT_OWN)
def test_a_value_that_may_share_a_part_goes_to_disk_as_the_caller_gave_it(name):
    tiers, disk = _tiers()
    value = NOT_OWN[name]()
    tiers.set("k", value, dict(_KEEP))
    (_key, written, private), = disk.calls
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
    (_key, written, private), = disk.calls
    assert private and written is not value


def test_a_later_persist_of_a_value_that_may_share_a_part_is_not_private():
    tiers, disk = _tiers()
    value = {"t": _tagged_table()}
    tiers.set("k", value, dict(_NOTEBOOK))
    assert disk.calls == []
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    (_key, _written, private), = disk.calls
    assert not private


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
