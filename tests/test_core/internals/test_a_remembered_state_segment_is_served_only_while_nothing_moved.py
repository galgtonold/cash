"""A function's state segment is reused only while nothing it reads moved.

The state segment of a key (the function's code, its helpers' and cached
callees', the globals they read, captures, defaults) is folded on every call.
With the extension it is remembered per function and reused while the
watchers show nothing it reads changed and cash's own state that it reads
(registrations, analyses, learned mutations, pins, settings) is as it was
(``cash.decorator.state_memo``).

Each scenario below first checks the segment IS being reused (else it would
pass on a fresh fold and prove nothing), then moves one thing it reads -- a
global rebound, a dict changed in place, a module's attribute, a class
attribute, a cached callee's code, a registration, a setting, a mutation
learned on a miss -- and checks that the segment and the call's result are
what a fresh fold gives. Each runs a second time against a sabotaged memo
that never notices a change, which must serve the stale segment: that is
what shows the scenario exercises the memo at all.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import types

import pytest

from cash import Cash
from cash.decorator import state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RATE = 2
TABLE = {"a": 1, "b": 2}
COUNTS: dict = {}


class Settings:
    scale = 3


def times_rate(x):
    return x * RATE


def via_table(name):
    return TABLE[name]


def via_class(x):
    return x * Settings.scale


def inner(x, k=1):
    return x + k


def inner_other(x, k=1):
    return x - k


def outer(x):
    return inner(x) * 10


def tally(x):
    COUNTS.setdefault(x, x)  # not a write the source shows: learned when seen
    return x


@pytest.fixture(autouse=True)
def _put_back():
    global RATE, TABLE
    saved = (RATE, TABLE, dict(TABLE), Settings.scale, inner.__code__)
    COUNTS.clear()
    yield
    RATE, TABLE, content, Settings.scale, code = saved
    TABLE.clear()
    TABLE.update(content)
    inner.__code__ = code
    COUNTS.clear()


@pytest.fixture(params=["honest", "sabotaged"])
def memo_kind(request, monkeypatch):
    """``sabotaged``: the memo takes every change for none (and the verify
    mode, which would catch that, is off)."""
    if request.param == "sabotaged":
        monkeypatch.setattr(state_memo, "VERIFY", False)
        monkeypatch.setattr(state_memo, "_world_epoch", lambda: 1)
        monkeypatch.setattr(state_memo, "moved", lambda: None)
        monkeypatch.setattr(state_memo.StateMemo, "_holds", lambda self: True)
    return request.param


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]
    return c._keys._code_state(spec, [], note=True), c._keys._code_state_now(spec, [], note=True)


def _reused(c, f):
    """Ask for the segment until the memo serves it (the first builds move
    cash's own state -- pins, analyses -- so they are not remembered)."""
    for _ in range(5):
        before = state_memo.STATS["reused"]
        got, fresh = _state(c, f)
        assert got == fresh
        if state_memo.STATS["reused"] > before:
            return got
    raise AssertionError("the segment is not being reused: the test proves nothing")


def _warm(c, f, *args):
    """Call *f*, then ask for its segment until it is reused; return it."""
    f(*args)
    return _reused(c, f)


def _check(c, f, kind, old_state):
    got, fresh = _state(c, f)
    assert fresh != old_state, "the change did not move the segment: the scenario proves nothing"
    if kind == "honest":
        assert got == fresh
    else:
        assert got == old_state  # stale: the sabotage is what the honest run guards against


def test_a_global_rebound(c, memo_kind):
    global RATE
    f = c.cache(times_rate)
    old = _warm(c, f, 1)
    RATE = 5
    _check(c, f, memo_kind, old)
    if memo_kind == "honest":
        assert f(1) == 5


def test_a_dict_changed_in_place(c, memo_kind):
    f = c.cache(via_table)
    old = _warm(c, f, "a")
    TABLE["a"] = 10
    _check(c, f, memo_kind, old)
    if memo_kind == "honest":
        assert f("a") == 10


def test_a_class_attribute(c, memo_kind):
    f = c.cache(via_class)
    old = _warm(c, f, 1)
    Settings.scale = 7
    _check(c, f, memo_kind, old)
    if memo_kind == "honest":
        assert f(1) == 7


MODULE = textwrap.dedent("""
    LIMIT = 4

    def capped(x):
        return min(x, LIMIT)
""")

READER = textwrap.dedent("""
    import {conf}

    def read_limit(x):
        return x + {conf}.LIMIT
""")


@pytest.fixture
def conf_modules(tmp_path, monkeypatch):
    stem = tmp_path.name.replace("-", "_")
    conf, reader = f"conf_{stem}", f"reader_{stem}"
    (tmp_path / f"{conf}.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / f"{reader}.py").write_text(READER.format(conf=conf), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(conf), importlib.import_module(reader)
    sys.modules.pop(conf, None)
    sys.modules.pop(reader, None)


def test_a_module_attribute(c, memo_kind, conf_modules):
    conf, reader = conf_modules
    f = c.cache(reader.read_limit)
    old = _warm(c, f, 1)
    conf.LIMIT = 40
    _check(c, f, memo_kind, old)
    if memo_kind == "honest":
        assert f(1) == 41


def test_a_cached_callees_code(c, memo_kind):
    c.cache(inner)
    f = c.cache(outer)
    old = _warm(c, f, 1)
    inner.__code__ = inner_other.__code__
    _check(c, f, memo_kind, old)


def test_a_registration(c, memo_kind):
    """Another function registered under the callee's name, with another
    default: no global moved, only cash's registry."""
    c.cache(inner)
    f = c.cache(outer)
    old = _warm(c, f, 1)
    again = types.FunctionType(inner.__code__, inner.__globals__, "inner", (50,))
    again.__qualname__ = inner.__qualname__
    again.__module__ = inner.__module__
    c.cache(again)
    _check(c, f, memo_kind, old)


def test_a_setting(c, memo_kind):
    """A setting moves cash's state: nothing remembered before is reused."""
    f = c.cache(times_rate)
    _warm(c, f, 1)
    before = state_memo.STATS["reused"]
    c.config.disable = False
    got, fresh = _state(c, f)
    assert got == fresh
    if memo_kind == "honest":
        assert state_memo.STATS["reused"] == before
    else:
        assert state_memo.STATS["reused"] > before


def _analyzed(c, f):
    spec = c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]
    c._registry.ensure_closure_analyzed(spec.func)


def _warm_without_calling(c, f):
    _analyzed(c, f)
    return _reused(c, f)


@pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheStoreFailedWarning")
def test_a_mutation_learned_on_a_miss(c, memo_kind):
    """The call changes a dict it reads: its miss learns that, and the
    segment stops folding the dict."""
    f = c.cache(tally)
    old = _warm_without_calling(c, f)
    f(1)
    assert "COUNTS" in c._mutations.of(tally.__code__, "global")
    _check(c, f, memo_kind, old)


def test_a_learned_mutation_alone(c, memo_kind):
    """Learned with nothing in the world moving: only cash's own state."""
    f = c.cache(tally)
    old = _warm_without_calling(c, f)
    c._mutations.learn(tally.__code__, "global", "COUNTS")
    _check(c, f, memo_kind, old)


def test_verify_mode_rebuilds_behind_every_reuse(c, monkeypatch):
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(times_rate)
    before = state_memo.STATS["verified"]
    for x in range(5):
        f(x)
    _state(c, f)
    _state(c, f)
    assert state_memo.STATS["verified"] > before


def test_verify_mode_raises_on_a_stale_reuse(c, monkeypatch):
    global RATE
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(times_rate)
    _warm(c, f, 1)
    monkeypatch.setattr(state_memo, "_world_epoch", lambda: 1)
    monkeypatch.setattr(state_memo.StateMemo, "_holds", lambda self: True)
    spec = c._registry.cached[times_rate.__module__ + "." + times_rate.__qualname__]
    spec.state_memo.world = 1
    spec.state_memo.gen = state_memo.GEN[0]
    RATE = 9
    with pytest.raises(state_memo.StateMemoMismatch):
        c._keys._code_state(spec, [], note=True)
