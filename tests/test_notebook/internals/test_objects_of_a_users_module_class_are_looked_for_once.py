"""Whether an object of a class of the user's modules exists is asked by
scanning every object the collector tracks (``gc.get_referrers``), about
17 ms per million objects. A statement's keys asked it many times: late in
a long test run, with a big heap on a loaded machine, one statement took
over 30 s. A yes is kept until code runs from an ``exec`` again: kept
stale (the object collected, a class id reused), it can only add to a key,
since it lets a walk run, which finds what is there. A no is asked afresh.
"""

from __future__ import annotations

import gc
import sys
import types

from cash.notebook import held_functions


class _OfAUsersModule:
    pass


def _asked_about(monkeypatch, cls):
    """Ask about *cls* as a class of a user's module, kept by that module
    alone (a closure over it would hold it as a value does)."""
    module = types.ModuleType("_a_users_module")
    name = cls.__name__
    setattr(module, name, cls)
    monkeypatch.setitem(sys.modules, "_a_users_module", module)
    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [getattr(module, name)])
    monkeypatch.setattr(held_functions, "_users_module_functions", list)


def test_objects_of_a_users_module_class_are_looked_for_once_until_code_runs(monkeypatch):
    # Each statement's keys asked this many times, and each asked was a scan
    # of every object the collector tracks (~17 ms per million): late in a
    # long test run one statement took over 30 s. A yes makes a walk run,
    # which finds what is there, so it is kept until code runs again.
    scans = []
    real = gc.get_referrers

    def counting(*objects):
        scans.append(len(objects))
        # Not the arguments tuple this wrapper made, which holds the classes.
        return [referrer for referrer in real(*objects) if referrer is not objects]

    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [_OfAUsersModule])
    monkeypatch.setattr(held_functions, "_users_module_functions", list)
    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    kept = _OfAUsersModule()
    assert all(held_functions.module_class_objects_exist() for _ in range(5))
    assert len(scans) == 1, scans
    exec("pass", {})  # the next statement
    assert held_functions.module_class_objects_exist()
    assert len(scans) == 2, scans
    del kept


def test_no_object_is_looked_for_again_each_time(monkeypatch):
    # A no is not kept: running code can make an object with no new code.
    scans = []
    real = gc.get_referrers

    def counting(*objects):
        scans.append(len(objects))
        # Not the arguments tuple this wrapper made, which holds the classes.
        return [referrer for referrer in real(*objects) if referrer is not objects]

    class Fresh:
        pass

    _asked_about(monkeypatch, Fresh)
    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    assert not held_functions.module_class_objects_exist()
    made = Fresh()
    assert held_functions.module_class_objects_exist()
    assert len(scans) == 2, scans
    del made


def test_a_class_held_without_an_object_of_it_counts(monkeypatch):
    # ``[mylib.Order] + rows`` holds the class and no object of it: answered
    # no, a walk of a big value stopped before it met the class, and the key
    # left the module out.
    class Held:
        pass

    _asked_about(monkeypatch, Held)
    rows = [Held, 1, 2]
    assert held_functions.module_class_objects_exist()
    del rows
    by_name = {"cls": Held}
    assert held_functions.module_class_objects_exist()
    del by_name


def test_what_every_class_has_does_not_count(monkeypatch):
    # Its module's globals, a subclass, its ``__mro__`` and ``__bases__``, its
    # slot and ``__dict__`` descriptors, and the ``__class__`` cell of a
    # method calling ``super()``.
    class Base:
        __slots__ = ("a",)

        def m(self):
            return super().__repr__()

    class Derived(Base):
        def m(self):
            return super().m()

    _asked_about(monkeypatch, Base)
    from cash.analysis import handed_callables

    handed_callables.is_users_class(Base)  # cash's own memo of it does not count
    assert not held_functions.module_class_objects_exist()
    assert Derived.__mro__[1] is Base


def test_with_the_trace_on_a_class_held_without_an_object_still_counts(monkeypatch):
    # The trace checks only its own statement's hit: not a cell built from
    # it, a call cached inside one, or a restore after a restart. A walk
    # it stopped short left the module out of all of those keys.
    from cash.notebook import shadow

    class Held:
        pass

    monkeypatch.setattr(shadow, "requested", lambda: True)
    _asked_about(monkeypatch, Held)
    rows = [Held, 1, 2]
    assert held_functions.module_class_objects_exist()
    del rows


def test_a_closure_over_the_class_counts_and_an_odd_name_does_not_raise(monkeypatch):
    class Held:
        pass

    _asked_about(monkeypatch, Held)

    def factory():
        k = Held
        return lambda: k

    made = factory()
    assert held_functions.module_class_objects_exist()
    del made
    held_functions._OBJECTS_SEEN[0] = None
    record = {"__name__": ["x"], "kind": Held}  # a parsed JSON record
    assert held_functions.module_class_objects_exist()
    del record


def test_a_module_function_held_in_a_value_counts(monkeypatch):
    # ``{'fn': mylib.helper, 'rows': rows}``: the function runs the module's
    # code, as a class held there does. Its module holding it does not count.
    module = types.ModuleType("_a_users_module")
    exec("def helper(v):\n    return v", vars(module))
    monkeypatch.setitem(sys.modules, "_a_users_module", module)
    monkeypatch.setattr(held_functions, "_users_module_classes", list)
    monkeypatch.setattr(held_functions, "_users_module_functions", lambda: [module.helper])
    assert not held_functions.module_class_objects_exist()
    big = {"fn": module.helper, "rows": [1, 2]}
    assert held_functions.module_class_objects_exist()
    del big


def test_only_a_module_function_held_elsewhere_is_looked_for(monkeypatch):
    # Every top-level function of the user's modules was asked about, and
    # ``gc.get_referrers`` costs each tracked object a look at each one: a
    # ``utils.py`` of 300 functions made a big value's walk 3-5x slower.
    # One its module alone holds is in no value.
    module = types.ModuleType("_a_users_module")
    exec("".join(f"def f{i}(v):\n    return v + {i}\n" for i in range(300)), vars(module))
    monkeypatch.setitem(sys.modules, "_a_users_module", module)
    monkeypatch.setattr(held_functions, "_users_module_classes", list)
    monkeypatch.setattr(
        held_functions,
        "_users_module_functions",
        lambda: [v for v in vars(module).values() if type(v) is types.FunctionType],
    )
    asked = []
    real = gc.get_referrers

    def counting(*objects):
        asked.append(len(objects))
        return [referrer for referrer in real(*objects) if referrer is not objects]

    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    assert not held_functions.module_class_objects_exist()
    assert asked == [], asked
    big = {"fn": module.f7, "rows": [1, 2]}
    assert held_functions.module_class_objects_exist()
    assert asked == [1], asked
    del big
