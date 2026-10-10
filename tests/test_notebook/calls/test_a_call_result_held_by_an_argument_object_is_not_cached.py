"""A call that returns an object its argument holds is never served as a copy.

``training_rows(ds)`` returning ``ds.rows``, ``ds`` an object of a class
defined in the user's own module (not in a cell): served from the call cache,
the result is a deserialised copy and a later ``rows.append(3)`` is not seen
through it. The call cache looks inside objects of the user's module classes
the way it looks inside a cell's own classes, and when an argument holds an
object it cannot look inside (``__slots__``, a library object) and something
else holds the result too, it does not store the result.

Every call sleeps above the persistence floor, and each test checks the call
log's ``cache_hit`` flags, not identity alone (see
``test_call_interception_identity_guard.py``).
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

import cash
from cash.notebook.call_interception import CallSite
from cash.notebook.shared_objects import holds_part_of
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

MODULE = f"""import time
class Dataset:
    def __init__(self, rows):
        self.rows = rows
class Slotted:
    __slots__ = ("rows",)
    def __init__(self, rows):
        self.rows = rows
def training_rows(ds):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return ds.rows
def row_count(ds):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return [len(ds.rows)]
"""


@pytest.fixture
def mylib(tmp_path, monkeypatch):
    name = f"_held_rows_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module(name)
    sys.modules.pop(name, None)


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _hits(call_cache, fn, *args):
    call_cache.set_sites([CallSite(source=f"{fn.__name__}(ds)", free_names=frozenset({"ds"}), occurrence_index=0)])
    cached = call_cache.resolve(fn)
    results = [cached(*args), cached(*args)]
    return results, [e["cache_hit"] for e in call_cache.drain_call_log()]


def test_an_object_of_a_module_class_is_looked_inside(mylib):
    rows = [1, 2]
    assert holds_part_of(rows, [mylib.Dataset(rows)])


def test_a_list_a_module_object_holds_is_handed_back_itself(call_cache, mylib):
    rows = [1, 2]
    ds = mylib.Dataset(rows)
    results, hits = _hits(call_cache, mylib.training_rows, ds)
    assert hits == [False, False]
    assert all(result is rows for result in results)


def test_a_list_an_object_with_slots_holds_is_handed_back_itself(call_cache, mylib):
    """The cache cannot look inside an object with ``__slots__``; the result is
    held elsewhere, so it is not stored."""
    rows = [1, 2]
    ds = mylib.Slotted(rows)
    results, hits = _hits(call_cache, mylib.training_rows, ds)
    assert hits == [False, False]
    assert all(result is rows for result in results)


def test_a_new_result_from_an_object_with_slots_is_still_served(call_cache, mylib):
    """Control: a result nothing else holds is stored and served."""
    ds = mylib.Slotted([1, 2])
    results, hits = _hits(call_cache, mylib.row_count, ds)
    assert hits == [False, True]
    assert results == [[2], [2]]
