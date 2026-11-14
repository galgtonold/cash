"""A helper's guard in a stored purity report is worked out once per process
and shared by every report that visits it, and worked out again as soon as
anything it was worked out from is not the same.

Forty cached functions of one library visit the same few hundred helpers:
working out each one's guard (`cash.analysis.report_store`) for every report
cost more than the walks the store spares. The guard is kept with what it
read -- the function's code, its file's text, what its names, cells and
defaults hold -- and served only while all of that holds: a rebind, a new
code object, a patched global, a mark or an edited file is seen.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap

import pytest

from cash.analysis import report_store

_SOURCE = textwrap.dedent(
    """
    import math

    LIMIT = 3


    def other(x):
        return -x


    def leaf(x):
        return x + 1


    def helper(x, k=2):
        return leaf(x) * k + LIMIT + math.floor(x)


    def make():
        inner = leaf

        def closure(x):
            return inner(x)

        return closure
    """
)


@pytest.fixture
def mod(tmp_path):
    path = tmp_path / "guarded_mod.py"
    path.write_text(_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("guarded_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["guarded_mod"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("guarded_mod", None)


@pytest.fixture
def worked_out(monkeypatch):
    """The objects whose guard was worked out (not served)."""
    seen = []
    plain = report_store._object_guard

    def counted(obj):
        seen.append(obj)
        return plain(obj)

    monkeypatch.setattr(report_store, "_object_guard", counted)
    return seen


def _digest(fn, file_digest="f1"):
    return report_store._object_digest(fn, file_digest)


def test_an_unchanged_helper_is_worked_out_once(mod, worked_out):
    first = _digest(mod.helper)
    assert _digest(mod.helper) == first
    assert worked_out == [mod.helper]


@pytest.mark.parametrize(
    "change",
    [
        "rebind-called-name",
        "rebind-module-name",
        "global-of-another-type",
        "default-of-another-type",
        "new-code",
        "marked-pure",
        "file-edited",
    ],
)
def test_a_change_to_what_the_guard_read_works_it_out_again(mod, worked_out, change):
    before = _digest(mod.helper)
    file_digest = "f1"
    if change == "rebind-called-name":
        mod.leaf = mod.other
    elif change == "rebind-module-name":
        mod.math = sys
    elif change == "global-of-another-type":
        mod.LIMIT = "3"
    elif change == "default-of-another-type":
        mod.helper.__defaults__ = (2.0,)
    elif change == "new-code":
        mod.helper.__code__ = mod.helper.__code__.replace(co_consts=mod.helper.__code__.co_consts)
    elif change == "marked-pure":
        mod.helper._cash_pure = True
    elif change == "file-edited":
        file_digest = "f2"
    after = _digest(mod.helper, file_digest)
    assert worked_out == [mod.helper, mod.helper], "the kept guard was served after a change"
    if change in ("rebind-called-name", "rebind-module-name", "global-of-another-type", "default-of-another-type"):
        assert after != before
    assert _digest(mod.helper, file_digest) == after
    assert len(worked_out) == 2


def test_a_value_of_the_same_plain_kind_keeps_the_guard(mod, worked_out):
    """The guard holds a plain value's kind, not the value: a new value of
    the same kind is the same guard (the global's value is keyed elsewhere)."""
    before = _digest(mod.helper)
    mod.LIMIT = 4
    assert _digest(mod.helper) == before
    assert worked_out == [mod.helper]


def test_a_closure_cell_rebound_works_it_out_again(mod, worked_out):
    closure = mod.make()
    before = _digest(closure)
    closure.__closure__[0].cell_contents = mod.other
    assert _digest(closure) != before
    assert worked_out == [closure, closure]


def test_the_guard_holds_no_data_value_alive(mod):
    import gc
    import weakref

    class Big:
        pass

    value = Big()
    mod.LIMIT = value
    ref = weakref.ref(value)
    _digest(mod.helper)
    mod.LIMIT = 3
    del value
    gc.collect()
    assert ref() is None


def test_a_function_or_class_a_kept_guard_read_can_be_dropped(mod):
    """The memo holds what a guard read by weak references only: a class
    or function the user drops (a notebook cell run again, a module
    unloaded) goes, and a new object given its id is worked out anew."""
    import gc
    import weakref

    class Dropped:
        pass

    def dropped_helper(x):
        return x

    mod.leaf = dropped_helper
    mod.LIMIT = Dropped
    _digest(mod.helper)
    _digest(dropped_helper)
    _digest(Dropped)
    report_store._identity(dropped_helper, {})
    refs = [weakref.ref(dropped_helper), weakref.ref(Dropped)]
    mod.leaf = mod.other
    mod.LIMIT = 3
    del dropped_helper, Dropped
    gc.collect()
    assert [r() for r in refs] == [None, None]


def test_a_dropped_helpers_entry_is_not_served_to_one_given_its_id(worked_out):
    """An entry is found by id; one whose object has gone matches nothing,
    so a new function that takes the dead one's id is worked out anew."""
    import gc

    namespace = {}
    source = "def f(x):\n    return x\n"
    exec(source, namespace)
    first = namespace.pop("f")
    _digest(first)
    entry = report_store._OBJECT_GUARDS.get(id(first))
    assert entry is not None
    worked_out.clear()  # it holds the function
    del first
    gc.collect()
    assert entry[0]() is None
    exec(source, namespace)
    again = namespace.pop("f")
    report_store._OBJECT_GUARDS[id(again)] = entry  # as if it took the dead one's id
    _digest(again)
    assert worked_out == [again]
