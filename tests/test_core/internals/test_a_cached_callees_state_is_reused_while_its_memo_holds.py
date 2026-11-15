"""A cached callee's state, folded into its caller's key, is reused while the callee's memo holds.

A caller whose own state segment cannot be remembered (a seed parameter, an
instance among its globals) builds it on every call, and with it the state
of each cached function it reaches without a graph edge of its own
registry: one on another `Cash`, reached through a module attribute, or one
held in a table (`KeyBuilder.callee_state`). That state is the callee's own
segment, which its memo (`state_memo`) remembers -- but it was built inside
the caller's key build, whose containers already held the caller's folds,
and the memo serves only into empty ones. So every call of the caller, and
the rebuild after its body when another thread is alive, folded the
callee's whole helper tree again. The callee's segment is now built apart
from the caller's containers (`state_memo.fresh_context`), where its memo
serves it.

Each scenario checks the callee's segment IS reused inside the caller's key
build, then changes one thing the callee's segment reads -- its helper
redefined, the helper's code swapped, a global, a list in place, the callee
itself replaced -- and checks the caller misses
and returns what plain Python does; with nothing changed, or only the
callee's stored entries cleared (which the caller's key does not read), it
hits. Each runs again against a sabotaged memo that takes every change for
none and must serve the stale result: that is what shows the scenario goes
through the memo at all.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheIneffectiveWarning"),
]

LIBRARY = textwrap.dedent("""
    SCALE = 2
    STEPS = [1, 2]

    def helper(x):
        return x * SCALE + sum(STEPS)

    def inner(x):
        return helper(x) + 1
""")

CALLER = textwrap.dedent("""
    import numpy as np

    import {lib} as lib

    RUNS = []
    TABLE = {{}}

    def outer(x, seed=0):
        RUNS.append(x)
        np.random.default_rng(seed)
        return lib.inner(x) * 10

    def outer_by_table(x, seed=0):
        RUNS.append(x)
        np.random.default_rng(seed)
        return TABLE["inner"](x) * 10
""")


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
def modules(tmp_path, monkeypatch):
    """A library module and a caller module importing it, fresh per test."""
    tag = tmp_path.name.replace("-", "_")
    lib, caller = f"calleelib_{tag}", f"calleecaller_{tag}"
    (tmp_path / f"{lib}.py").write_text(LIBRARY, encoding="utf-8", newline="\n")
    (tmp_path / f"{caller}.py").write_text(CALLER.format(lib=lib), encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(lib), importlib.import_module(caller)
    sys.modules.pop(lib, None)
    sys.modules.pop(caller, None)


def _other_instance(modules):
    """``outer`` reaches ``inner``, cached on another `Cash`, through a
    module attribute."""
    lib, mod = modules
    lib_cash = Cash(backend="memory", register_magic=False)
    lib.inner = lib_cash.cache(lib.inner)
    c = Cash(backend="memory", register_magic=False)
    return c, c.cache(mod.outer), lib.inner


def _table(modules):
    """``outer_by_table`` reaches ``inner``, cached on the same `Cash`,
    held in a dict."""
    lib, mod = modules
    c = Cash(backend="memory", register_magic=False)
    mod.TABLE["inner"] = c.cache(lib.inner)
    return c, c.cache(mod.outer_by_table), mod.TABLE["inner"]


SHAPES = [_other_instance, _table]


def _spec(c, f):
    return c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]


def _plain(modules, x):
    lib, _ = modules
    inner = getattr(lib.inner, "__wrapped__", lib.inner)
    return inner(x) * 10


def _warm(c, f, inner, modules):
    """Call *f* until the callee's segment is reused inside its key build."""
    assert state_memo.memo_for(_spec(c, f)) is None, "the caller's own segment is remembered: no callee build to test"
    f(1)
    for _ in range(6):
        before = state_memo.STATS["reused"]
        assert f(1) == _plain(modules, 1)
        if state_memo.STATS["reused"] > before:
            return
    raise AssertionError("the callee's segment is not being reused: the test proves nothing")


def _runs(modules):
    return len(modules[1].RUNS)


def _check_miss(f, modules, kind):
    """After a change the callee's segment reads: the caller misses and
    returns what plain Python does (honest), or serves the old result
    (sabotaged)."""
    runs = _runs(modules)
    got = f(1)
    if kind == "honest":
        assert got == _plain(modules, 1)
        assert _runs(modules) == runs + 1
        assert f(1) == _plain(modules, 1)
        assert _runs(modules) == runs + 1
    else:
        assert got != _plain(modules, 1)  # stale: what the honest run guards against


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_nothing_changed_hits(shape, modules):
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    runs = _runs(modules)
    for _ in range(3):
        assert f(1) == _plain(modules, 1)
    assert _runs(modules) == runs


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_clearing_the_callees_entries_keeps_the_callers(shape, modules):
    """The caller's key reads the callee's state segment, not what it
    stored: plain Python computes the same, and the caller hits."""
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    runs = _runs(modules)
    inner.cache_clear()
    assert f(1) == _plain(modules, 1)
    assert _runs(modules) == runs


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_a_global_the_callee_reads_rebound(shape, modules, memo_kind):
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    modules[0].SCALE = 7
    _check_miss(f, modules, memo_kind)


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_a_list_the_callee_reads_changed_in_place(shape, modules, memo_kind):
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    modules[0].STEPS.append(40)
    _check_miss(f, modules, memo_kind)


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_the_callees_helper_redefined(shape, modules, memo_kind):
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    exec("def helper(x):\n    return x * 100 + sum(STEPS)\n", modules[0].__dict__)
    _check_miss(f, modules, memo_kind)


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_the_callees_helpers_code_swapped(shape, modules, memo_kind):
    c, f, inner = shape(modules)
    _warm(c, f, inner, modules)
    scope: dict = {"SCALE": modules[0].SCALE, "STEPS": modules[0].STEPS}
    exec("def other(x):\n    return x - SCALE + sum(STEPS)\n", scope)
    modules[0].helper.__code__ = scope["other"].__code__
    _check_miss(f, modules, memo_kind)


def test_the_callee_redefined_and_decorated_again(modules):
    """A new function under the callee's name, cached again: the caller
    reaches another wrapper, and misses."""
    c, f, inner = _other_instance(modules)
    _warm(c, f, inner, modules)
    lib = modules[0]
    scope: dict = {"helper": lib.helper}
    exec("def inner(x):\n    return helper(x) - 3\n", scope)
    lib.inner = Cash(backend="memory", register_magic=False).cache(scope["inner"])
    _check_miss(f, modules, "honest")


@pytest.mark.parametrize("shape", SHAPES, ids=["other_instance", "table"])
def test_verify_mode_agrees_on_every_reuse(shape, modules, monkeypatch):
    monkeypatch.setattr(state_memo, "VERIFY", True)
    c, f, inner = shape(modules)
    before = state_memo.STATS["verified"]
    for x in range(4):
        assert f(x) == _plain(modules, x)
    modules[0].STEPS.append(3)
    for x in range(4):
        assert f(x) == _plain(modules, x)
    assert state_memo.STATS["verified"] > before
