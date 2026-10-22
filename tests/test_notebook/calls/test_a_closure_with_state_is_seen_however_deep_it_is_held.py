"""A value holding a closure that keeps state is never served from the cache,
however deep in the value the closure sits.

A function is stored by reference, so a hit hands back the closure the last
run made, with everything it counted since. Only the value itself and the
containers directly in it were looked at: ``handlers = make_handlers()``
returning ``{"on": {"click": [counter]}}`` was cached, and running the
notebook again handed back a counter that went on from where it was.
"""

from __future__ import annotations

import functools
import time
import types

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


def _double(n):
    return n * 2


def make_memo():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return functools.cache(_double)


def test_a_memo_is_a_function_that_keeps_state():
    assert holds_a_closure_with_state(functools.cache(_double))
    assert holds_a_closure_with_state({"memo": [functools.lru_cache(maxsize=8)(_double)]})


def test_a_memo_a_call_made_is_not_served_already_filled(call_unit_harness):
    # `functools.cache(g)` was stored when it took over the cost floor (a busy
    # machine), then filled later in the statement: after a restart the hit
    # handed back the filled memo and `g` never ran where plain Python ran it.
    unit = call_unit_harness(lineage={"make_memo": "hash-make-memo"}, user_ns={})
    site = CallSite(source="make_memo()", free_names=frozenset({"make_memo"}), occurrence_index=0)

    unit.begin_statement()
    first = unit.wrap(make_memo, site)()
    first(1)
    unit.begin_statement()
    memo = unit.wrap(make_memo, site)()

    assert memo.cache_info().currsize == 0, "the memo the last run filled was served again"


class _Box:
    pass


class _Holder:
    def __init__(self):
        self.memo = functools.cache(_double)

    def read(self, n):
        return self.memo(n)


def _boxed_memo():
    box = _Box()
    box.memo = functools.cache(_double)
    return box


def _wrapped_memo():
    memo = functools.cache(_double)

    def fast(n):
        return memo(n)

    return fast


def _class_with_a_memo():
    class Scores:
        @classmethod
        @functools.cache
        def of(cls, n):
            return n * 2

    return Scores


def test_a_memo_or_counter_held_past_a_container_is_seen():
    # A deep copy of these still holds the same memo: a function, a partial's
    # function and a class copy as themselves.
    assert holds_a_closure_with_state(types.SimpleNamespace(f=functools.cache(_double)))
    assert holds_a_closure_with_state(_boxed_memo())
    assert holds_a_closure_with_state(_wrapped_memo())
    assert holds_a_closure_with_state(functools.partial(functools.cache(_double)))
    assert holds_a_closure_with_state(_class_with_a_memo())
    assert holds_a_closure_with_state(types.SimpleNamespace(c=make_counter()))
    assert holds_a_closure_with_state(_Holder().read)  # the memo is on the method's object
    assert not holds_a_closure_with_state(types.SimpleNamespace(f=_double, box=_Box()))


def test_a_memo_on_an_object_a_call_made_is_not_served_already_filled(call_unit_harness):
    def make_box():
        time.sleep(0.05)  # above the call cost floor, so a miss is stored
        return _boxed_memo()

    unit = call_unit_harness(lineage={"make_box": "hash-make-box"}, user_ns={})
    site = CallSite(source="make_box()", free_names=frozenset({"make_box"}), occurrence_index=0)

    unit.begin_statement()
    unit.wrap(make_box, site)().memo(1)
    unit.begin_statement()
    box = unit.wrap(make_box, site)()

    assert box.memo.cache_info().currsize == 0, "the memo the last run filled was served again"
