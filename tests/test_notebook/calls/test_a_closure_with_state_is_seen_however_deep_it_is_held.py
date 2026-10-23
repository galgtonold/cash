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
import dataclasses
import decimal
import enum
import functools
import reprlib
import time
import types

import numpy as np
import pytest

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


class _Parts:
    def __init__(self, i):
        self.parts = [i, (i,)]


def test_a_value_with_more_holders_than_are_looked_into_is_not_stored(monkeypatch):
    # Past the cap what the rest holds is unknown: not stored (one rerun),
    # never served unchecked, and said so apart from a memo found.
    from cash.notebook import call_key
    from cash.notebook.statement.store import _holds_state

    monkeypatch.setattr(call_key, "_HOLDERS_LOOKED_INTO", 50)
    with pytest.raises(call_key.TooMuchToLookThrough):
        holds_a_closure_with_state(
            (types.SimpleNamespace(memo=functools.cache(_double)), [_Parts(i) for i in range(51)])
        )
    assert _holds_state([_Parts(i) for i in range(51)]) == "too_many"
    assert not holds_a_closure_with_state([_Parts(i) for i in range(40)])
    # Records of plain values have nothing inside to look at: not counted.
    assert not holds_a_closure_with_state([_Row(i) for i in range(1000)])
    assert _holds_state([_Row(i) for i in range(1000)]) is None
    assert _holds_state([functools.cache(_double)]) == "state"


@dataclasses.dataclass
class _Rec:
    a: int
    b: str


@dataclasses.dataclass(frozen=True)
class _FrozenRec:
    a: int


class _Pretty:
    @reprlib.recursive_repr()
    def __repr__(self):
        return "pretty"


def _made_dataclass():
    @dataclasses.dataclass
    class Made:
        a: int

    return Made(1)


def test_a_record_or_a_named_class_with_a_memo_method_is_storable(monkeypatch):
    # A class its module names comes back as that very class: its memo is
    # the live one, as in plain Python. A dataclass's ``__repr__`` guard
    # keeps nothing between calls.
    class _Model:
        def __init__(self, n):
            self.n = n

        @functools.lru_cache(maxsize=None)
        def score(self, x):
            return self.n * x

    _named_here(monkeypatch, _Model)
    assert not holds_a_closure_with_state([_Rec(i, str(i)) for i in range(100)])
    assert not holds_a_closure_with_state(_FrozenRec(1))
    assert not holds_a_closure_with_state(_Model(3))
    assert not holds_a_closure_with_state(_Pretty())
    assert not holds_a_closure_with_state(_made_dataclass())


def load_recs():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    return [_Rec(i, str(i)) for i in range(100)]


def test_a_call_returning_records_is_served(call_unit_harness):
    # Refused on every run while the walk counted a dataclass's ``__repr__``
    # guard as a closure that keeps state.
    unit = call_unit_harness(lineage={"load_recs": "hash-load-recs"}, user_ns={})
    site = CallSite(source="load_recs()", free_names=frozenset({"load_recs"}), occurrence_index=0)
    for _ in range(3):
        unit.begin_statement()
        unit.wrap(load_recs, site)()

    assert [entry["cache_hit"] for entry in unit.call_log] == [False, True, True]


def test_a_statement_holding_records_is_restored(cash_magics):
    from tests._cell_driver import run_cash_cell

    setup = "import dataclasses, time\n@dataclasses.dataclass\nclass Rec:\n    a: int\n    b: str\n"
    statement = "recs = (time.sleep(0.05), [Rec(i, str(i)) for i in range(100)])[1]"
    statuses = []
    for _ in range(3):
        run_cash_cell(cash_magics, setup)
        run_cash_cell(cash_magics, statement)
        statuses.append(str(cash_magics.cash_status("dict")["last_cell"]["statements"][-1]["status"]))

    assert statuses[1:] == ["RESTORED", "RESTORED"], statuses


class _RowsList(list):
    pass


class _RowsDict(dict):
    pass


def _property_over_a_memo():
    memo = functools.cache(_double)

    class Tool:
        @property
        def m(self):
            return memo

    return Tool()


def test_a_memo_on_a_container_subclass_a_property_or_a_library_holder_is_seen():
    rows = _RowsList([1])
    rows.memo = functools.cache(_double)
    assert holds_a_closure_with_state(rows)
    table = _RowsDict(a=1)
    table.memo = functools.cache(_double)
    assert holds_a_closure_with_state(table)
    assert holds_a_closure_with_state(_property_over_a_memo())
    assert holds_a_closure_with_state(collections.UserDict(m=functools.cache(_double)))
    assert holds_a_closure_with_state(collections.UserList([functools.cache(_double)]))
    assert holds_a_closure_with_state(collections.ChainMap({"m": functools.cache(_double)}))
    assert not holds_a_closure_with_state(_RowsList([1, 2]))


class _Iterates(list):
    asked: list = []

    def __iter__(self):
        _Iterates.asked.append("__iter__")
        return super().__iter__()


class _Values(dict):
    asked: list = []

    def values(self):
        _Values.asked.append("values")
        raise RuntimeError("values raised")


def test_a_users_container_subclass_is_read_without_asking_it():
    _Iterates.asked.clear()
    _Values.asked.clear()
    assert holds_a_closure_with_state(_Iterates([functools.cache(_double)]))
    assert holds_a_closure_with_state(_Values(m=functools.cache(_double)))
    assert _Iterates.asked == [] and _Values.asked == []


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


def _named_here(monkeypatch, cls):
    """*cls*, named by this module for one test: a memo on a class at the
    top of a test module is one of the user's to every other test."""
    cls.__qualname__ = cls.__name__
    for member in vars(cls).values():
        inner = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
        wrapped = getattr(inner, "__wrapped__", None)
        if wrapped is not None:
            wrapped.__qualname__ = f"{cls.__name__}.{wrapped.__name__}"
    monkeypatch.setitem(globals(), cls.__name__, cls)
    return cls


def put_a_memo_on_a_named_class():
    time.sleep(0.05)  # above the call cost floor, so a miss is stored
    holder = globals()["_MemoHolder"]  # named by the test that runs it
    holder.memo = staticmethod(functools.cache(_double))
    return holder()


def test_a_memo_the_call_puts_on_a_named_class_is_not_served(call_unit_harness, monkeypatch):
    # Served, the class kept the first run's memo, filled since; plain
    # Python puts a new one there on every run.
    class _MemoHolder:
        pass

    _named_here(monkeypatch, _MemoHolder)
    unit = call_unit_harness(lineage={"put_a_memo_on_a_named_class": "hash-put-memo"}, user_ns={})
    site = CallSite(
        source="put_a_memo_on_a_named_class()",
        free_names=frozenset({"put_a_memo_on_a_named_class"}),
        occurrence_index=0,
    )
    for _ in range(2):
        unit.begin_statement()
        held = unit.wrap(put_a_memo_on_a_named_class, site)()
        held.memo(3)

    assert [entry["cache_hit"] for entry in unit.call_log] == [False, False]
    assert _MemoHolder.memo.cache_info().hits == 0


def test_a_memo_on_a_named_class_counts_when_the_call_filled_it(monkeypatch):
    # After a restart the class comes back with an empty memo, which the
    # call filled in plain Python: ``y = (lookup(1), hits)`` read (2, 0)
    # served where it is (2, 1).
    class _Looks:
        @staticmethod
        @functools.cache
        def lookup(n):
            return n * 2

    _named_here(monkeypatch, _Looks)
    assert not holds_a_closure_with_state(_Looks())
    _Looks.lookup(1)
    ran_here = {_Looks.lookup.__wrapped__.__code__: None}

    assert holds_a_closure_with_state(_Looks(), ran_here, []), "filled in the call"
    assert holds_a_closure_with_state(type("Sub", (_Looks,), {})(), ran_here, []), "on a named base"
    assert holds_a_closure_with_state(_Looks(), {}), "nowhere to record what it holds"


def test_a_memo_filled_before_the_call_is_recorded_for_the_entry(monkeypatch):
    # Refused, the second call of ``get_model("bert")`` in another cell
    # was never stored (3 of 17 statements of a notebook without a trace).
    # Recorded, the entry is served only while the memo holds these
    # answers again: after a restart it is empty, and the call runs.
    from cash.notebook import helper_memos

    class _Kept:
        @staticmethod
        @functools.cache
        def lookup(n):
            return n * 2

    _named_here(monkeypatch, _Kept)
    _Kept.lookup(1)
    memos: list[dict] = []
    assert not holds_a_closure_with_state(_Kept(), {}, memos)
    assert [m["where"][2:] for m in memos] == [["_Kept", ".lookup"]]
    assert helper_memos.unfilled(memos, {}) is None
    _Kept.lookup.cache_clear()  # what a restart leaves
    assert helper_memos.unfilled(memos, {}) is not None


def test_without_a_trace_the_miss_count_tells_who_filled_the_memo(monkeypatch):
    class _Counted:
        @staticmethod
        @functools.cache
        def lookup(n):
            return n * 2

    _named_here(monkeypatch, _Counted)
    _Counted.lookup(1)
    before = {id(_Counted.lookup): _Counted.lookup.cache_info().misses}
    assert not holds_a_closure_with_state(_Counted(), None, [], before), "only read"
    _Counted.lookup(2)
    assert holds_a_closure_with_state(_Counted(), None, [], before), "filled in the call"
    assert holds_a_closure_with_state(_Counted(), None, [], {}), "not known before"


class _Tagged:
    def __init__(self, i):
        self.i = i
        self.tags = [i, "x"]


def test_records_with_lists_of_plain_values_are_not_counted(monkeypatch):
    # On 3.12+ an instance hands over its attributes themselves, not a dict:
    # a record with a list was counted as a holder, and 200k of them were
    # not stored.
    from cash.notebook import call_key

    monkeypatch.setattr(call_key, "_HOLDERS_LOOKED_INTO", 50)
    assert not holds_a_closure_with_state([_Tagged(i) for i in range(1000)])


def test_the_walk_takes_a_bounded_number_of_values(monkeypatch):
    # With plain records not counted the walk had no end: a million took
    # 2.7 s. Past the bound it is not known, so not stored, and said so.
    from cash.notebook import call_key
    from cash.notebook.statement.store import _holds_state

    monkeypatch.setattr(call_key, "_STEPS_AT", 1000)
    rows = [_Tagged(_Tagged(_Tagged(i))) for i in range(2000)]
    with pytest.raises(call_key.TooMuchToLookThrough):
        holds_a_closure_with_state(rows)
    assert _holds_state(rows) == "too_many"
    assert not holds_a_closure_with_state(rows[:100])


class _Color(enum.Enum):
    RED = 1


class _Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b


ROWS_OF = {
    "numpy_scalar": lambda i: _Pair(i, np.float64(i)),
    "enum_member": lambda i: _Pair(i, _Color.RED),
    "decimal": lambda i: _Pair(i, decimal.Decimal(i)),
    "record_of_records": lambda i: _Pair(_Pair(i, 0.5), _Pair(-i, 0.5)),
}


@pytest.mark.parametrize("shape", list(ROWS_OF))
def test_records_of_scalars_enum_members_or_flat_records_are_not_counted(monkeypatch, shape):
    # A row holding a numpy scalar, an Enum member, a Decimal or a record of
    # plain values took a step of its own: 300,000 of them went past the
    # bound and were not stored, where the base stored them.
    from cash.notebook import call_key

    monkeypatch.setattr(call_key, "_HOLDERS_LOOKED_INTO", 50)
    monkeypatch.setattr(call_key, "_STEPS_AT", 1000)
    assert not holds_a_closure_with_state([ROWS_OF[shape](i) for i in range(2000)])


def test_a_memo_on_the_class_of_a_record_a_record_holds_is_seen():
    held = _instance_with_a_class_memo()
    assert holds_a_closure_with_state([_Pair(i, 0.5) for i in range(100)] + [_Pair(1, held)])
    assert holds_a_closure_with_state([_Pair(_Pair(i, 0.5), held) for i in range(100)])


def test_a_million_plain_records_are_looked_through_within_the_bound():
    # Each record pushed its class again: 300,000 dataclass rows went past
    # the bound and were not stored.
    @dataclasses.dataclass
    class _Row:
        a: int
        b: str
        c: float

    rows = [_Row(i, str(i), i * 0.5) for i in range(1_000_000)]
    assert not holds_a_closure_with_state(rows)


def test_a_module_is_not_looked_into():
    # ``module`` says it is from ``builtins``, which does not name it: it was
    # taken for a class made by ``exec`` and looked into, so ``import numpy
    # as np; arr = np.loadtxt(p)`` held numpy's memos and was never stored.
    import fnmatch

    assert not holds_a_closure_with_state(fnmatch)
    assert not holds_a_closure_with_state({"m": [fnmatch]})


@pytest.mark.parametrize(
    "shape",
    [
        lambda m: m,
        lambda m: (m, 1),
        lambda m: {"lookup": [m]},
    ],
    ids=["itself", "in_a_tuple", "in_a_dict_of_lists"],
)
def test_a_memo_is_state(shape):
    # Served, ``f = functools.cache(g)`` handed back the memo the last run
    # filled: ``f(1)`` then did not run ``g`` where plain Python runs it.
    assert holds_a_closure_with_state(shape(functools.cache(_double)))
    assert holds_a_closure_with_state(shape(functools.lru_cache(maxsize=4)(_double)))


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
