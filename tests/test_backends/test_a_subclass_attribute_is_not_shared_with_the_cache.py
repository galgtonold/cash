"""A table subclass's own attributes are copied with the table.

pandas copies the attributes a DataFrame or Series subclass names in
``_metadata`` (a GeoDataFrame's ``crs``, a user's ``info`` dict) by
reference into every copy, deep or not. The RAM tier deep-copies such a
table on the store and on every hit, so the stored table, the caller's
result and every hit held one ``info`` dict: what the caller did to its own
result's ``df.info`` reached every later call.
"""

from __future__ import annotations

import threading

import pytest

from cash.backends.memory_backend import InMemoryBackend
from cash.exceptions import CacheBackendError

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


class Tagged(pd.DataFrame):
    _metadata = ["info"]

    @property
    def _constructor(self):
        return Tagged


class TaggedSeries(pd.Series):
    _metadata = ["info"]

    @property
    def _constructor(self):
        return TaggedSeries


class NoDeepcopy:
    """Pickles, but refuses deepcopy."""

    def __init__(self):
        self.steps = []

    def __deepcopy__(self, memo):
        raise TypeError("no deepcopy")


class Holder:
    """Holds a table beside what pickle refuses: copied by deepcopy."""

    def __init__(self, table):
        self.table = table
        self.fn = lambda: None


def _tagged(n=3, cells=None):
    frame = Tagged({"a": np.arange(n, dtype=float)} if cells is None else {"a": cells})
    frame.info = {"source": "fresh", "steps": []}
    return frame


def _series():
    s = TaggedSeries(np.arange(3.0))
    s.info = {"source": "fresh", "steps": []}
    return s


TABLES = {
    "frame": _tagged,
    "series": _series,
    "frame with list cells": lambda: _tagged(cells=[[1], [2], [3]]),
    "frame with object cells": lambda: _tagged(cells=["x", 1, None]),
}

FRESH = {"source": "fresh", "steps": []}


def _store_and_hit(value, key="k"):
    backend = InMemoryBackend()
    backend.set(key, value, {"execution_time": 1.0, "copy_required": True})
    return backend


@pytest.mark.parametrize("name", TABLES)
def test_neither_the_caller_nor_a_hit_changes_what_the_next_hit_gets(name):
    table = TABLES[name]()
    backend = _store_and_hit(table)
    table.info["steps"].append("caller edit")
    first = backend.get("k")[1]
    assert first.info == FRESH
    first.info["steps"].append("edit of a hit")
    assert backend.get("k")[1].info == FRESH
    assert table.info["steps"] == ["caller edit"]


@pytest.mark.parametrize(
    "wrap",
    [
        lambda t: (t, 1),
        lambda t: {"variables": {"df": t}},
        lambda t: [t, {"x": [1]}],
        lambda t: Holder(t),
    ],
    ids=["tuple", "notebook entry", "list", "object deepcopy copies"],
)
def test_a_table_inside_a_value_gets_its_own_attributes(wrap):
    table = _tagged()
    backend = InMemoryBackend()
    backend.set("k", wrap(table), {"execution_time": 1.0})
    table.info["steps"].append("caller edit")

    def find(value):
        if isinstance(value, Holder):
            return value.table
        if isinstance(value, dict):
            return value["variables"]["df"]
        return value[0]

    hit = find(backend.get("k")[1])
    assert hit.info == FRESH
    hit.info["steps"].append("edit of a hit")
    assert find(backend.get("k")[1]).info == FRESH


def test_two_attributes_holding_one_object_still_share_it():
    class Two(pd.DataFrame):
        _metadata = ["a_info", "b_info"]

        @property
        def _constructor(self):
            return Two

    table = Two({"a": [1.0]})
    table.a_info = table.b_info = {"k": []}
    hit = _store_and_hit(table).get("k")[1]
    assert hit.a_info is hit.b_info
    assert hit.a_info is not table.a_info


def test_an_attribute_deepcopy_refuses_is_copied_through_pickle():
    table = _tagged()
    table.info = NoDeepcopy()
    backend = _store_and_hit(table)
    table.info.steps.append("caller edit")
    hit = backend.get("k")[1]
    assert hit.info.steps == []
    assert hit.info is not table.info


def test_an_attribute_no_copy_can_be_made_of_is_refused_not_shared():
    """A lock neither deep-copies nor pickles: a decorator store refuses the
    table, as it does a frame whose cells cannot be copied."""
    table = _tagged()
    table.info = {"lock": threading.Lock()}
    with pytest.raises(CacheBackendError):
        _store_and_hit(table)
    backend = InMemoryBackend()
    metadata = {"execution_time": 1.0}
    backend.set("k", table, metadata)
    assert metadata.get("by_reference") is True


def test_a_cached_function_returns_the_attributes_it_built(cash_instance):
    @cash_instance.cache
    def load(n):
        frame = Tagged({"a": np.arange(n, dtype=float)})
        frame.info = {"source": "fresh", "steps": []}  # @cash:assume-safe
        return frame

    load(3).info["steps"].append("caller edit")
    load(3).info["steps"].append("edit of hit 1")
    assert load(3).info == FRESH
