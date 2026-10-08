"""An intercepted call whose result holds an object the function reaches
outside its arguments -- a global, an item of one, a class attribute, its own
memo, a bound method of a global -- is not served from the call cache.

A hit hands back a deserialised copy. ``m = best()`` returning ``MODELS[1]``
then binds ``m`` to a copy, and ``m['w'] = 9`` leaves ``MODELS`` as it was;
``b = build('x')`` returning ``{'model': MODEL}`` holds a copy, and a later
in-place change of ``MODEL`` is not seen through ``b['model']``. Plain Python
keeps the very objects, so the call runs every time instead (bug-hunt-5 AL-02,
CC-6).
"""

from __future__ import annotations

import functools
import time

import pytest

import cash
from cash.notebook.call_entries import reached_objects
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


MODELS = [{"w": 0}, {"w": 0}]
MODEL = {"w": 1}
LOG: list = []
_DATA = None


class Config:
    items: list = []


def best():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return MODELS[1]


def build(tag):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"tag": tag, "model": MODEL}


def get_data():
    global _DATA
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    if _DATA is None:
        _DATA = {"rows": [1, 2]}
    return _DATA


def items():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return Config.items


def logger():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return LOG.append


def _pick_best():
    return MODELS[0]


def through_a_helper():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return [_pick_best()]


def summary():
    """Reads a global, returns a new object: nothing of the global in it."""
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"n": len(MODELS), "w": [m["w"] for m in MODELS]}


def _hits(call_cache, fn, *args):
    call_cache.set_sites([CallSite(source=f"{fn.__name__}()", free_names=frozenset({fn.__name__}), occurrence_index=0)])
    cached = call_cache.resolve(fn)
    results = [cached(*args), cached(*args)]
    return results, [e["cache_hit"] for e in call_cache.drain_call_log()]


@pytest.mark.parametrize(
    ("fn", "args", "held"),
    [
        (best, (), lambda r: r is MODELS[1]),
        (build, ("x",), lambda r: r["model"] is MODEL),
        (get_data, (), lambda r: r is _DATA),
        (items, (), lambda r: r is Config.items),
        (logger, (), lambda r: r.__self__ is LOG),
        (through_a_helper, (), lambda r: r[0] is MODELS[0]),
    ],
    ids=["item-of-a-global", "dict-holding-a-global", "lazy-singleton", "class-attribute", "bound-method", "helper"],
)
def test_a_result_holding_a_global_is_the_global_again(call_cache, fn, args, held):
    results, hits = _hits(call_cache, fn, *args)
    assert hits == [False, False], "a result holding a global was served as a copy"
    assert all(held(r) for r in results)


def test_a_result_holding_what_a_closure_keeps_is_not_cached(call_cache):
    registry = {"k": 1}

    def get():
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return registry

    results, hits = _hits(call_cache, get)
    assert hits == [False, False]
    assert all(r is registry for r in results)


def test_a_result_built_from_a_global_is_still_cached(call_cache):
    """Control: reading a global and returning a new object is cached."""
    results, hits = _hits(call_cache, summary)
    assert hits == [False, True]
    assert results[0] == results[1] == {"n": 2, "w": [0, 0]}


def test_reached_objects_follows_partials_and_bound_methods():
    shared = {"k": 1}

    class Holder:
        def get(self):
            return self.table

    holder = Holder()
    holder.table = shared
    assert any(obj is shared for obj in reached_objects(functools.partial(dict.get, shared)))
    assert any(obj is holder for obj in reached_objects(holder.get))
