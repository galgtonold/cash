"""A value holding a closure that keeps state is never served from the cache,
however deep in the value the closure sits.

A function is stored by reference, so a hit hands back the closure the last
run made, with everything it counted since. Only the value itself and the
containers directly in it were looked at: ``handlers = make_handlers()``
returning ``{"on": {"click": [counter]}}`` was cached, and running the
notebook again handed back a counter that went on from where it was.
"""

from __future__ import annotations

import time

from cash.notebook.call_interception import CallSite
from cash.notebook.call_key import holds_a_closure_with_state


def make_counter():
    n = 0

    def bump():
        nonlocal n
        n += 1
        return n

    return bump


def make_handlers():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return {"on": {"click": [make_counter()]}}


def test_a_closure_three_containers_down_is_seen():
    assert holds_a_closure_with_state({"on": {"click": [make_counter()]}})
    assert holds_a_closure_with_state([[[[[(make_counter(),)]]]]])


def test_a_value_that_holds_no_such_closure_is_still_storable():
    loop: list = [1, 2]
    loop.append(loop)
    assert not holds_a_closure_with_state({"a": [loop, {"b": (3, "x")}]}), "a cycle must end"
    assert not holds_a_closure_with_state([[[len]]])


def test_a_closure_among_many_tuples_of_plain_values_is_seen():
    records = [("user", [("action", 1, 2.5, None, b"x"), ("other", True)]) for _ in range(50)]

    assert not holds_a_closure_with_state(records)
    records[7][1].append((1, (make_counter(),)))
    assert holds_a_closure_with_state(records)


def test_a_factory_returning_nested_handlers_is_not_served_the_last_run_s(call_unit_harness):
    unit = call_unit_harness(lineage={"make_handlers": "hash-make-handlers"}, user_ns={})
    site = CallSite(source="make_handlers()", free_names=frozenset({"make_handlers"}), occurrence_index=0)

    unit.begin_statement()
    first = unit.wrap(make_handlers, site)()
    first["on"]["click"][0]()
    unit.begin_statement()
    handlers = unit.wrap(make_handlers, site)()

    assert handlers["on"]["click"][0]() == 1, "the counter the last run made was served again"


class Stepper:
    def __init__(self):
        self.n = 0
        self.size = 3

    def step(self):
        self.n += 1
        return self.n


def make_movers():
    import random

    import numpy as np

    g = np.random.default_rng(0)
    r = random.Random(0)
    it = iter(range(9))
    o = Stepper()

    def draw():
        return float(g.normal())

    def pick():
        return r.random()

    def take():
        return next(it)

    def step():
        return o.step()

    return draw, pick, take, step


def make_readers():
    import numpy as np

    cfg = {"a": 1}
    arr = np.zeros(3)
    o = Stepper()

    def read():
        return cfg["a"] + float(arr.sum()) + o.size

    return read


def test_a_closure_whose_generator_iterator_or_object_moves_when_called_is_seen():
    """A draw, a ``next`` or a method of the user's object moves what the
    closure holds without the source rule seeing a write: served by
    reference after a re-run, ``h, peek = make_helper()`` handed back a
    generator already past the draw ``h()`` stands for."""
    for mover in make_movers():
        assert holds_a_closure_with_state(mover), mover.__name__


def test_a_closure_only_reading_what_it_holds_is_still_storable():
    assert not holds_a_closure_with_state(make_readers())
