"""The check that a call result holds no global the function reaches walks
those globals only when something may hold part of the result.

A function reading a large global (a list of 200,000 records) and returning
a new object would otherwise pay a walk over the whole global on every
stored call. Pins the work, not the behaviour: a refactor may expect this to
fail.
"""

from __future__ import annotations

import time

import cash
from cash.notebook import call_entries
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

RECORDS = [{"a": i} for i in range(1000)]


def fresh(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"row": dict(RECORDS[i]), "n": len(RECORDS)}


def shared(i):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"row": RECORDS[i]}


def _walks(tmp_path, monkeypatch, fn):
    walked = []
    real = call_entries.reached_objects
    monkeypatch.setattr(call_entries, "reached_objects", lambda f: walked.append(f) or real(f))
    call_cache = make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    call_cache.set_sites([CallSite(source=f"{fn.__name__}(i)", free_names=frozenset({fn.__name__}), occurrence_index=0)])
    call_cache.resolve(fn)(1)
    return walked


def test_a_fresh_result_does_not_walk_the_globals(tmp_path, monkeypatch):
    assert _walks(tmp_path, monkeypatch, fresh) == []


def test_a_result_holding_a_global_item_walks_them(tmp_path, monkeypatch):
    assert _walks(tmp_path, monkeypatch, shared) == [shared]
