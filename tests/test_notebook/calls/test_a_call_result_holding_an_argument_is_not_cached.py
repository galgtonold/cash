"""An intercepted call whose result holds one of its arguments, or an object
inside one, is not served from the call cache.

A hit hands back a deserialised copy. ``b = bundle(model, data)`` returning
``{'model': model, ...}`` then holds a copy of ``model``, and a later
``model.fit()`` is not seen through ``b['model']``; ``c = pick(cfg, 'a')``
returning ``cfg['a']`` binds ``c`` to a copy, and ``c['n'] = 5`` leaves
``cfg`` as it was. Plain Python keeps the very objects, so the call runs
every time instead.
"""

from __future__ import annotations

import time

import pytest

import cash
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


class Model:
    def __init__(self):
        self.w = 0


def bundle(model, data):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"model": model, "data": data, "n": len(data)}


def pick(cfg, key):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return cfg[key]


def wrap(data):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return [data, len(data)]


def count(data):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return {"n": len(data), "first": list(data[:1])}


def _hits(call_cache, fn, source, *args):
    call_cache.set_sites([CallSite(source=source, free_names=frozenset({fn.__name__}), occurrence_index=0)])
    cached = call_cache.resolve(fn)
    results = [cached(*args), cached(*args)]
    return results, [e["cache_hit"] for e in call_cache.drain_call_log()]


@pytest.mark.parametrize(
    ("fn", "args", "held"),
    [
        (bundle, (Model(), [1, 2]), lambda r, args: r["model"] is args[0] and r["data"] is args[1]),
        (pick, ({"a": {"n": 1}}, "a"), lambda r, args: r is args[0]["a"]),
        (wrap, ([1, 2],), lambda r, args: r[0] is args[0]),
    ],
    ids=["bundle", "pick", "wrap"],
)
def test_a_result_holding_an_argument_is_the_argument_again(call_cache, fn, args, held):
    results, hits = _hits(call_cache, fn, f"{fn.__name__}(x)", *args)
    assert hits == [False, False], "a result holding an argument was served as a copy"
    assert all(held(r, args) for r in results)


def test_a_result_built_from_an_argument_is_still_cached(call_cache):
    """Control: a new object made from the argument holds nothing of it."""
    results, hits = _hits(call_cache, count, "count(x)", [1, 2])
    assert hits == [False, True]
    assert results == [{"n": 2, "first": [1]}] * 2
