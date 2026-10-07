"""Storing a notebook entry in the RAM tier walks its leaves once.

``d = [f(i) for i in range(200_000)]`` stored its payload with three passes
over the 200,000 ints: one to size it (``tree_size``), one to copy it
(``spine_copy``) and one to plan the copy a hit would make (``copy_plan``),
15 ms of a cell the plain kernel ran in 30 ms. The size and the copy now
share one walk, and the plan is made by the first hit: most entries are
never read.
"""

from __future__ import annotations

import random

from cash import _plain_data
from cash.backends.memory_backend import InMemoryBackend


def _entry():
    return {
        "variables": {"d": [i * 2 for i in range(200_000)], "rows": [{"a": i} for i in range(3)]},
        "stdout": "",
        "rng_state": random.getstate(),
    }


def _counting(monkeypatch, name):
    calls = []
    real = getattr(_plain_data, name)

    def counted(*args, **kwargs):
        if name != "copy_plan" or len(args) == 1:  # not its recursion into the parts
            calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(_plain_data, name, counted)
    return calls


def test_a_store_walks_the_leaves_once_and_makes_no_copy_plan(monkeypatch):
    walks = _counting(monkeypatch, "_tree_walk")
    plans = _counting(monkeypatch, "copy_plan")
    entry = _entry()
    backend = InMemoryBackend()
    backend.set("k", entry, {})
    assert len(walks) == 1
    assert plans == []
    stored = backend.get("k")[1]
    assert stored == entry and stored["variables"]["d"] is not entry["variables"]["d"]
    assert stored["variables"]["rows"][0] is not entry["variables"]["rows"][0]


def test_the_first_hit_plans_the_copy_and_later_hits_reuse_it(monkeypatch):
    entry = _entry()
    backend = InMemoryBackend()
    backend.set("k", entry, {})
    plans = _counting(monkeypatch, "copy_plan")
    first, second = backend.get("k")[1], backend.get("k")[1]
    assert first == entry == second and first is not second
    assert first["variables"]["d"] is not second["variables"]["d"]
    assert len(plans) == 1


def test_a_replaced_entry_is_not_copied_by_the_plan_of_the_one_before(monkeypatch):
    backend = InMemoryBackend()
    backend.set("k", _entry(), {})
    replaced = {"variables": {"d": [[1], [2]]}, "stdout": ""}
    real = _plain_data.copy_plan

    def plan_then_replace(value, *args):
        plan = real(value, *args)
        if value is not replaced:
            backend.set("k", replaced, {})
        return plan

    monkeypatch.setattr(_plain_data, "copy_plan", plan_then_replace)
    backend.get("k")
    assert backend.get("k")[1] == replaced


def test_a_level_is_sampled_at_the_same_positions_without_drawing_them_again(monkeypatch):
    level = list(range(_plain_data.SIZE_EXACT_UP_TO + 1000))
    size = _plain_data._level_size(level)
    seeded = []
    monkeypatch.setattr(_plain_data.random, "Random", lambda *a: seeded.append(a) or random.Random(*a))
    assert _plain_data._level_size(level) == size
    assert seeded == []
