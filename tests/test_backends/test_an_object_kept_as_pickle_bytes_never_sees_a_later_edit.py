"""An object the RAM tier keeps as pickle bytes comes back as it was stored.

An object of a class of its own -- a fitted model, a graph -- was copied
by a pickle round trip when it was stored, and the unpickling was most of
the store: a statsmodels result rebuilds its design matrices, 7 s of a
10 s fit, paid for values never read again. The tier now keeps the pickle
bytes and reads a copy out of them when the value is served.

The bytes are cash's own, made when the value is stored: the stream is
new, and every large buffer pickle hands out of band (an array's memory)
is copied out of the value. No edit of the caller's object afterwards --
in Python, or by numpy writing into an array's memory in C -- reaches what
a hit hands out, and each hit hands out a copy of its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

ROWS = 1 << 18  # 2 MiB of floats: big enough to be kept as bytes, and out of band


class Fit:
    """A results object: arrays, a frame, lists and dicts, and attributes."""

    def __init__(self) -> None:
        self.design = np.arange(ROWS, dtype=float)
        self.frozen = np.arange(ROWS, dtype=float)
        self.frozen.flags.writeable = False
        self.labels = ["a", "b"]
        self.nested = {"rows": [[1, 2], [3, 4]]}
        self.frame = pd.DataFrame({"x": np.arange(1000.0), "s": ["a"] * 1000})
        self.name = "fit"


class Graph:
    """A graph as networkx keeps one: dicts of dicts, an array on a node."""

    def __init__(self) -> None:
        self._adj = {n: {n + 1: {"w": 1.0}} for n in range(100)}
        self._node = {0: {"weights": np.ones(ROWS)}}


def _append_to_a_list(fit: Fit) -> None:
    fit.labels.append("z")


def _set_a_dict_item(fit: Fit) -> None:
    fit.nested["new"] = 1


def _append_to_a_list_inside_a_dict(fit: Fit) -> None:
    fit.nested["rows"][0].append(9)


def _set_an_attribute(fit: Fit) -> None:
    fit.name = "changed"


def _set_an_array_item(fit: Fit) -> None:
    fit.design[0] = -1.0


def _write_an_array_in_c(fit: Fit) -> None:
    np.add(fit.design, 1.0, out=fit.design)


def _set_a_frame_cell(fit: Fit) -> None:
    frame = fit.frame
    frame.loc[0, "x"] = -1.0


def _assign_a_frame_column(fit: Fit) -> None:
    frame = fit.frame
    frame["s"] = "b"


EDITS = {
    "a list appended to": _append_to_a_list,
    "a dict item set": _set_a_dict_item,
    "a list inside a dict appended to": _append_to_a_list_inside_a_dict,
    "an attribute set": _set_an_attribute,
    "an array item set": _set_an_array_item,
    "an array written by numpy in C": _write_an_array_in_c,
    "a frame cell set with loc": _set_a_frame_cell,
    "a frame column assigned": _assign_a_frame_column,
}


def _edits(fit: Fit) -> None:
    """Every kind of change a caller can make to its own object."""
    for edit in EDITS.values():
        edit(fit)


def _changed(fit: Fit) -> bool:
    try:
        _as_built(fit)
    except AssertionError:
        return True
    return False


def _as_built(fit: Fit) -> None:
    """Fails unless *fit* is what `Fit()` builds."""
    fresh = Fit()
    np.testing.assert_array_equal(fit.design, fresh.design)
    np.testing.assert_array_equal(fit.frozen, fresh.frozen)
    assert fit.labels == fresh.labels
    assert fit.nested == fresh.nested
    assert fit.name == fresh.name
    pd.testing.assert_frame_equal(fit.frame, fresh.frame)


def _ram_with(value, key: str = "k") -> InMemoryBackend:
    ram = InMemoryBackend()
    assert ram.set(key, value, {"execution_time": 5.0}) is not False
    assert ram.stored_payload(key) is not None, "not kept as pickle bytes: the test checks nothing new"
    return ram


@pytest.mark.parametrize("edit", list(EDITS.values()), ids=list(EDITS))
def test_an_edit_of_the_callers_object_after_the_store_is_not_served(edit):
    fit = Fit()
    ram = _ram_with(fit)
    edit(fit)
    assert _changed(fit), "the edit changed nothing: the test checks nothing"
    _as_built(ram.get("k")[1])


@pytest.mark.parametrize("edit", list(EDITS.values()), ids=list(EDITS))
def test_an_edit_of_one_hit_is_not_in_the_next(edit):
    ram = _ram_with(Fit())
    first = ram.get("k")[1]
    edit(first)
    assert _changed(first), "the edit changed nothing: the test checks nothing"
    _as_built(ram.get("k")[1])


def test_every_edit_at_once_reaches_neither_the_entry_nor_the_disk(tmp_path):
    """The disk tier writes the bytes the RAM tier keeps: what it holds is
    the value as stored too, read in by a new backend over the folder."""
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    fit = Fit()
    tiers.set("k", fit, {"execution_time": 60.0, "force_persist": True})
    _edits(fit)
    disk.shutdown()
    _as_built(tiers.backends[0].get("k")[1])
    _as_built(FileBackend(str(tmp_path / "c")).get("k")[1])


def test_an_array_inside_a_graph_written_in_c_is_not_served():
    graph = Graph()
    ram = _ram_with(graph)
    np.multiply(graph._node[0]["weights"], 0.0, out=graph._node[0]["weights"])
    graph._adj[0][1]["w"] = 5.0
    served = ram.get("k")[1]
    assert served._node[0]["weights"].sum() == ROWS
    assert served._adj[0][1]["w"] == 1.0


def test_a_read_only_array_stays_read_only_and_a_writable_one_writable():
    served = _ram_with(Fit()).get("k")[1]
    assert not served.frozen.flags.writeable
    assert served.design.flags.writeable
    served.design[0] = 7.0  # its own memory: no other hit sees it


def test_a_decorated_function_hands_each_caller_the_result_as_computed(cash_instance):
    @cash_instance.cache
    def fit_model() -> Fit:
        return Fit()

    first = fit_model()
    _edits(first)
    second = fit_model()
    _as_built(second)
    _edits(second)
    _as_built(fit_model())
