"""A state segment that reads a library's values the world watch refused
is reused while what the key read of them holds.

Real walks met them on every call of imbalanced-learn's and pvlib's code,
which folded the whole segment again (`state_memo`) each time:

* a list of what is watched otherwise (sklearn's parameter constraints,
  ``[Interval(...), StrOptions(...)]``): which items it holds is read again
  before each reuse, by identity, and each item is looked into;
* a named tuple (a ``tuple`` subclass without attributes of its own): its
  items looked into as any value;
* values fixed once made: a numpy scalar, pandas' ``NaT`` and axis
  descriptors, a dataclass's ``__dataclass_params__``;
* a ``ContextVar``: what it holds in the call's context is read again;
* an object read through a library's module (``logging._handlers``),
  which a key never folds: not looked into at all.

A weak mapping is still refused: the collector changes it, and watching it
moved the world for every memo.

Each must-miss scenario checks the segment IS reused first, then changes
one thing the fold read, and checks the segment is a fresh fold's and the
call returns what plain Python does; then again against a sabotaged memo
that takes every change for none, which must serve the stale segment.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import weakref

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo
from cash.decorator.user_code import is_user_code_object

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

LIBRARY = textwrap.dedent("""
    import collections


    class Interval:
        def __init__(self, left, right):
            self.left = left
            self.right = right

        def __repr__(self):
            return f"Interval({self.left}, {self.right})"


    Pair = collections.namedtuple("Pair", "items scale")
""")

USER = textwrap.dedent("""
    import contextvars
    import dataclasses
    import logging

    import numpy as np
    import pandas as pd

    from {lib} import Interval, Pair

    CONSTRAINTS = [Interval(0, 1), "auto"]
    PAIR = Pair([1, 2], 3)
    SCALE = np.float64(1.5)
    FLAG = np.bool_(True)
    MISSING = pd.NaT
    INDEX = vars(pd.DataFrame)["index"]
    VAR = contextvars.ContextVar("var", default=1)

    @dataclasses.dataclass(frozen=True)
    class Point:
        x: int = 1

    PARAMS = Point.__dataclass_params__

    def constrained(x):
        return (x, repr(CONSTRAINTS))

    def paired(x):
        return x * sum(PAIR.items) * PAIR.scale

    def fixed(x):
        return (x * float(SCALE), bool(FLAG), MISSING is pd.NaT, INDEX is not None, PARAMS.frozen)

    def contextual(x):
        return x + VAR.get()

    def logged(x):
        return x + len(logging._handlers) * 0
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    tag = tmp_path.name.replace("-", "_")
    lib = f"vallib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"libvals_{tag}"
    (tmp_path / f"{name}.py").write_text(USER.format(lib=lib), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)
    sys.modules.pop(lib, None)


@pytest.fixture(params=["honest", "sabotaged"])
def memo_kind(request, monkeypatch):
    """``sabotaged``: the memo takes every change for none (and the verify
    mode, which would catch that, is off)."""
    if request.param == "sabotaged":
        monkeypatch.setattr(state_memo, "VERIFY", False)
        monkeypatch.setattr(state_memo, "_world_epoch", lambda: 1)
        monkeypatch.setattr(state_memo, "moved", lambda: None)
        monkeypatch.setattr(state_memo.StateMemo, "_holds", lambda self: True)
        monkeypatch.setattr(hit_memo.ContentProbe, "__call__", lambda self: self.digest)
    return request.param


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]
    return c._keys._code_state(spec, [], note=True), c._keys._code_state_now(spec, [], note=True)


def _warm(c, f, *args):
    """Call *f*, then ask for its segment until it is reused; return it."""
    f(*args)
    for _ in range(5):
        before = state_memo.STATS["reused"]
        got, fresh = _state(c, f)
        assert got == fresh
        if state_memo.STATS["reused"] > before:
            return got
    raise AssertionError("the segment is not being reused: the test proves nothing")


def _check(c, f, kind, old_state, *args):
    got, fresh = _state(c, f)
    assert fresh != old_state, "the change did not move the segment: the scenario proves nothing"
    if kind == "honest":
        assert got == fresh
        assert f(*args) == f.__wrapped__(*args)
        assert f(*args) == f.__wrapped__(*args)
    else:
        assert got == old_state  # stale: the sabotage is what the honest run guards against


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.CONSTRAINTS.__setitem__(0, m.Interval(0, 2)),
        lambda m: m.CONSTRAINTS.append(m.Interval(5, 6)),
        lambda m: setattr(m.CONSTRAINTS[0], "right", 9),
    ],
    ids=["item_replaced", "item_added", "item_changed"],
)
def test_a_list_of_library_objects_changed(mod, c, memo_kind, change):
    f = c.cache(mod.constrained)
    old = _warm(c, f, 1)
    change(mod)
    _check(c, f, memo_kind, old, 1)


def test_a_named_tuple_holding_a_list_changed(mod, c, memo_kind):
    f = c.cache(mod.paired)
    old = _warm(c, f, 2)
    mod.PAIR.items.append(10)
    _check(c, f, memo_kind, old, 2)


@pytest.mark.parametrize("name", ["fixed", "contextual", "logged"])
def test_fixed_values_a_context_variable_and_a_library_object_are_reused_and_verified(mod, c, monkeypatch, name):
    """Reused, every reuse checked against a fresh fold; a context
    variable set and a logging handler added leave it agreeing with one."""
    import logging

    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(getattr(mod, name))
    _warm(c, f, 3)
    token = mod.VAR.set(7)
    handler = logging.NullHandler()
    try:
        logging.getLogger("libvals-test").addHandler(handler)
        got, fresh = _state(c, f)
        assert got == fresh
        assert f(4) == f.__wrapped__(4)
    finally:
        logging.getLogger("libvals-test").removeHandler(handler)
        mod.VAR.reset(token)
    assert state_memo.STATS["verified"] > 0


def test_a_weak_mapping_is_not_watched():
    scan = hit_memo._Scan("anywhere")
    scan.content = True
    with pytest.raises(hit_memo.Ineligible):
        scan.value(weakref.WeakValueDictionary(), frozenset())


def test_a_pure_python_twin_of_a_c_class_is_library_code():
    """``zoneinfo._zoneinfo.ZoneInfo`` calls itself ``zoneinfo.ZoneInfo``,
    the C class's name: judged by where its methods' code lives, it is the
    standard library's, not exec'd user code whose data a key folds."""
    twin = pytest.importorskip("zoneinfo._zoneinfo").ZoneInfo
    assert is_user_code_object(twin) is False
