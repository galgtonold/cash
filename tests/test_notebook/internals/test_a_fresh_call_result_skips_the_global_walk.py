"""The check that a call result holds no global the function reaches walks
those globals only when something may hold part of the result.

A function reading a large global (a list of 200,000 records) and returning
a new object would otherwise pay a walk over the whole global on every
stored call. Pins the work, not the behaviour: a refactor may expect this to
fail.
"""

from __future__ import annotations

import time

import pytest

import cash
from cash.notebook import call_entries
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

RECORDS = [{"a": i} for i in range(1000)]


def fresh(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"row": dict(RECORDS[i]), "n": len(RECORDS)}


CONFIG = {"rate": 3}


def config(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return CONFIG


def shared(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"row": RECORDS[i]}


def _walks(tmp_path, monkeypatch, fn):
    walked = []
    real = call_entries.reached_objects
    monkeypatch.setattr(call_entries, "reached_objects", lambda f: walked.append(f) or real(f))
    call_cache = make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    call_cache.set_sites(
        [CallSite(source=f"{fn.__name__}(i)", free_names=frozenset({fn.__name__}), occurrence_index=0)]
    )
    call_cache.resolve(fn)(1)
    return walked


def test_a_fresh_result_does_not_walk_the_globals(tmp_path, monkeypatch):
    assert _walks(tmp_path, monkeypatch, fresh) == []


def test_a_result_holding_a_global_item_walks_them(tmp_path, monkeypatch):
    assert _walks(tmp_path, monkeypatch, shared) == [shared]


def test_a_result_that_is_a_global_walks_them(tmp_path, monkeypatch):
    assert _walks(tmp_path, monkeypatch, config) == [config]


def _after_try(make):
    # A name assigned inside ``try`` may be unbound after it: Python 3.14
    # loads it as a new reference there, not a borrowed one.
    try:
        value = make()
    finally:
        pass
    return call_entries.refs_beyond([value])


def _plain(make):
    value = make()
    return call_entries.refs_beyond([value])


@pytest.mark.parametrize("read", [_after_try, _plain], ids=["after_try", "plain"])
def test_the_reference_count_does_not_depend_on_how_the_caller_loads_its_local(read):
    assert read(object) == call_entries.ONE_LOCAL
    assert read(lambda: CONFIG) > call_entries.ONE_LOCAL
