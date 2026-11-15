"""A state segment that reads a numba dispatcher or a library's settings
object is reused while what the key read of it holds.

The world watch (`cash.decorator.hit_memo._Scan`) refused both, so every
call of such a function folded its state segment again (`state_memo`):

* a numba dispatcher (``@njit``), which a key reads as the Python function
  it compiles (``py_func``) and its compile options (``targetoptions``,
  ``locals``). Its ``__dict__``, which numba does not write after making
  it, is now watched whole, its options' dicts as any dict met, and
  ``py_func`` is scanned as any function;
* a library's settings object: an instance whose state is its
  ``__dict__`` (watched whole, a property's setter writing a private
  attribute in it), or a class whose metaclass has properties returning
  what the class holds (``settings.n_jobs`` reading ``cls._n_jobs``; the
  setter writes the class, a type event).

Each scenario checks the segment IS reused first, then changes one thing
the fold read between two calls, and checks the segment is a fresh fold's
and the call returns what plain Python does. Each runs again against a
sabotaged memo that takes every change for none, which must serve the
stale segment: that is what shows the scenario goes through the memo.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    # A ``__code__`` assigned in place reads as the file changing under the process.
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheStoreFailedWarning"),
]

LIBRARY = textwrap.dedent("""
    class Config:
        def __init__(self):
            self._n_jobs = 1
            self.rate = 2.0

        @property
        def n_jobs(self):
            return self._n_jobs

        @n_jobs.setter
        def n_jobs(self, value):
            self._n_jobs = int(value)


    settings = Config()


    class Meta(type):
        @property
        def n_jobs(cls):
            return cls._n_jobs

        @n_jobs.setter
        def n_jobs(cls, value):
            cls._n_jobs = int(value)


    class class_settings(metaclass=Meta):
        _n_jobs = 1


    class Computed:
        @property
        def n_jobs(self):
            return len(str(self))


    computed = Computed()


    class Slotted:
        __slots__ = ("rate",)

        def __init__(self):
            self.rate = 2.0


    slotted = Slotted()
""")

USER = textwrap.dedent("""
    import numba
    import numpy as np

    from {lib} import class_settings, computed, settings, slotted

    def _scaled(x):
        return x * 2.0

    def _shifted(x):
        return x + 100.0

    kernel = numba.njit(_scaled)

    def jitted(x):
        return float(kernel(np.float64(x)))

    def configured(x):
        return x * settings.rate + settings.n_jobs

    def class_configured(x):
        return x + class_settings.n_jobs

    def computed_configured(x):
        return x + computed.n_jobs

    def slotted_configured(x):
        return x * slotted.rate
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    pytest.importorskip("numba")
    tag = tmp_path.name.replace("-", "_")
    lib = f"cfglib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"jitcfg_{tag}"
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


def _code_changed(mod):
    mod.kernel.py_func.__code__ = mod._shifted.__code__


def _rebound(mod):
    import numba

    mod.kernel = numba.njit(mod._shifted)


def _py_func_rebound(mod):
    mod.kernel.py_func = mod._shifted


def _option_set(mod):
    mod.kernel.targetoptions["fastmath"] = True


def _options_rebound(mod):
    mod.kernel.targetoptions = {**mod.kernel.targetoptions, "error_model": "numpy"}


@pytest.mark.parametrize(
    "change",
    [_code_changed, _rebound, _py_func_rebound, _option_set, _options_rebound],
    ids=["code_changed", "rebound", "py_func_rebound", "option_set", "options_rebound"],
)
def test_a_jit_function_changed(mod, c, memo_kind, change):
    f = c.cache(mod.jitted)
    old = _warm(c, f, 3)
    change(mod)
    _check(c, f, memo_kind, old, 3)


def _attribute_set(mod):
    mod.settings.rate = 3.0


def _setter_used(mod):
    mod.settings.n_jobs = 4


@pytest.mark.parametrize("change", [_attribute_set, _setter_used], ids=["attribute_set", "setter_used"])
def test_a_settings_object_changed(mod, c, memo_kind, change):
    f = c.cache(mod.configured)
    old = _warm(c, f, 3)
    change(mod)
    _check(c, f, memo_kind, old, 3)


def test_a_settings_class_changed_through_its_property(mod, c, monkeypatch):
    """A settings class whose metaclass's property returns a class
    attribute: reused, and its setter (a write to the class) moves the
    world, the segment then agreeing with a fresh fold."""
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(mod.class_configured)
    _warm(c, f, 3)
    world = hit_memo.world_epochs()[0]
    mod.class_settings.n_jobs = 4
    assert hit_memo.world_epochs()[0] != world
    got, fresh = _state(c, f)
    assert got == fresh
    assert state_memo.STATS["verified"] > 0


@pytest.mark.parametrize("name", ["computed_configured", "slotted_configured"])
def test_a_settings_object_read_by_code_or_held_in_slots_is_not_reused(mod, c, name):
    """A property that computes (``len(str(self))``) or a slot no watcher
    sees keeps the segment from being remembered: it is folded again on
    every call."""
    f = c.cache(getattr(mod, name))
    f(3)
    before = state_memo.STATS["reused"]
    for _ in range(4):
        got, fresh = _state(c, f)
        assert got == fresh
    assert state_memo.STATS["reused"] == before


@pytest.mark.filterwarnings("error::cash.exceptions.CashCacheIneffectiveWarning")
def test_a_jit_function_and_a_settings_object_are_verified(mod, c, monkeypatch):
    """Under ``CASH_STATE_MEMO_VERIFY`` every reuse is checked against a
    fresh fold, across changes of both kinds."""
    monkeypatch.setattr(state_memo, "VERIFY", True)
    jitted, configured = c.cache(mod.jitted), c.cache(mod.configured)
    before = state_memo.STATS["verified"]
    for change in (None, _option_set, _setter_used, _rebound, _attribute_set):
        if change is not None:
            change(mod)
        for f in (jitted, configured):
            _warm(c, f, 3)
            assert f(3) == f.__wrapped__(3)
    assert state_memo.STATS["verified"] >= before + 10
