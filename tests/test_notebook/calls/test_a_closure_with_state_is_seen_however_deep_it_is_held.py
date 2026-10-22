"""A value holding a closure that keeps state is never served from the cache,
however deep in the value the closure sits.

A function is stored by reference, so a hit hands back the closure the last
run made, with everything it counted since. Only the value itself and the
containers directly in it were looked at: ``handlers = make_handlers()``
returning ``{"on": {"click": [counter]}}`` was cached, and running the
notebook again handed back a counter that went on from where it was.
"""

from __future__ import annotations

import argparse
import collections
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


class _Row:
    def __init__(self, i):
        self.i = i


def _defaulted():
    def fast(n, _memo=functools.cache(_double)):
        return _memo(n)

    return fast


def _kw_defaulted():
    def fast(n, *, _memo=functools.cache(_double)):
        return _memo(n)

    return fast


def _instance_of_a_made_class():
    class Scores:
        @classmethod
        @functools.cache
        def of(cls, n):
            return n * 2

    return Scores()


def _instance_with_a_class_memo():
    class Tool:
        memo = staticmethod(functools.cache(_double))

    return Tool()


def _subclass_of_a_class_with_a_memo():
    class Base:
        @staticmethod
        @functools.cache
        def of(n):
            return n * 2

    class Sub(Base):
        pass

    return Sub


def test_a_memo_in_a_default_on_a_class_or_a_base_is_seen():
    assert holds_a_closure_with_state(_defaulted())
    assert holds_a_closure_with_state(_kw_defaulted())
    assert holds_a_closure_with_state(_instance_of_a_made_class())
    assert holds_a_closure_with_state(_instance_with_a_class_memo())
    assert holds_a_closure_with_state(_subclass_of_a_class_with_a_memo())
    assert holds_a_closure_with_state(collections.deque([functools.cache(_double)]))
    assert holds_a_closure_with_state(argparse.Namespace(memo=functools.cache(_double)))


def test_a_value_with_more_holders_than_are_looked_into_is_not_stored(monkeypatch):
    # Past the cap what the rest holds is unknown: not stored (one rerun),
    # never served unchecked. The memo sits where the walk comes last.
    from cash.notebook import call_key

    monkeypatch.setattr(call_key, "_HOLDERS_LOOKED_INTO", 50)
    assert holds_a_closure_with_state(
        (types.SimpleNamespace(memo=functools.cache(_double)), [_Row(i) for i in range(51)])
    )
    assert holds_a_closure_with_state([_Row(i) for i in range(51)])
    assert not holds_a_closure_with_state([_Row(i) for i in range(40)])


class _Hostile:
    __slots__ = ("a", "b")
    asked: list = []

    def __init__(self):
        self.a = functools.cache(_double)

    def __getattr__(self, name):
        _Hostile.asked.append(name)
        raise RuntimeError("a remote fetch")


def test_the_walk_asks_an_object_nothing():
    import gc

    _Hostile.asked.clear()
    assert holds_a_closure_with_state(_Hostile())
    assert _Hostile.asked == []

    # Made here, not at module level: a live class whose metaclass answers
    # lookups itself changes how later tests in the same process are cached.
    asked = []

    class AsksBack(type):
        def __getattribute__(cls, name):
            asked.append(name)
            return super().__getattribute__(name)

    class Guarded(metaclass=AsksBack):
        memo = staticmethod(functools.cache(_double))

    assert holds_a_closure_with_state(Guarded)
    assert holds_a_closure_with_state(Guarded())
    assert asked == []
    del Guarded, AsksBack
    gc.collect()


def test_a_class_a_call_made_is_not_kept_alive_by_the_walk():
    import gc
    import weakref

    made = _subclass_of_a_class_with_a_memo()
    holds_a_closure_with_state(made)
    gone = weakref.ref(made)
    del made
    gc.collect()
    assert gone() is None
