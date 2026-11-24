"""A state segment whose world holds a lock is reused while it holds.

A lock (``threading.Lock()``, ``threading.RLock()``) carries no data: a
key reads it by its type alone. The world watch
(`cash.decorator.hit_memo._Scan`) refused one all the same, so a function
reaching a lock folded its state segment again on every call
(`state_memo`): the ``RLock`` of the module-global ``rich`` console that
nilearn's ``resample_img`` reaches failed every scan.

A lock the scan meets is now taken as it is: what a key reads of it, its
type, cannot change, and which lock a name or an attribute holds is
watched as the dict holding it. The segment is checked to be reused, then
to agree with a fresh fold after the data beside the lock changes (against
a sabotaged memo too, which must serve the stale segment: that shows the
scenario goes through the memo), after the lock itself is rebound, and
while it is held.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import threading

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

LIBRARY = textwrap.dedent("""
    import threading


    class Console:
        _class_lock = threading.RLock()

        def __init__(self):
            self.rate = 2.0

        def log(self, x):
            with self._class_lock:
                return x


    class LockedConsole:
        def __init__(self):
            self.rate = 2.0
            self._lock = threading.RLock()
            self._write_lock = threading.Lock()

        def log(self, x):
            with self._lock:
                return x


    console = Console()
    locked_console = LockedConsole()
""")

USER = textwrap.dedent("""
    import threading

    from {lib} import console, locked_console

    _LOCK = threading.Lock()
    _RLOCK = threading.RLock()
    OFFSET = 1.0

    def guarded(x):
        with _LOCK:
            with _RLOCK:
                return x + OFFSET

    def logged(x):
        return console.log(x) * console.rate

    def locked_logged(x):
        return locked_console.log(x) * locked_console.rate
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    tag = tmp_path.name.replace("-", "_")
    lib = f"locklib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"lockuser_{tag}"
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
    raise AssertionError("the segment is not being reused")


def _check(c, f, kind, old_state, *args):
    got, fresh = _state(c, f)
    assert fresh != old_state, "the change did not move the segment: the scenario proves nothing"
    if kind == "honest":
        assert got == fresh
        assert f(*args) == f.__wrapped__(*args)
        assert f(*args) == f.__wrapped__(*args)
    else:
        assert got == old_state  # stale: the sabotage is what the honest run guards against


def test_a_module_global_lock_is_reused(mod, c, memo_kind):
    """``with _LOCK:`` in the function: the lock is keyed by its type, and
    the data beside it still moves the segment."""
    f = c.cache(mod.guarded)
    old = _warm(c, f, 3)
    mod.OFFSET = 2.0
    _check(c, f, memo_kind, old, 3)


def test_a_library_object_holding_locks_is_reused_as_a_fresh_fold_keys_it(mod, c, monkeypatch):
    """A library's object holding an ``RLock`` and a ``Lock`` itself (the
    shape of ``rich``'s console): the scan takes it, and every reuse is the
    segment a fresh fold builds, before and after its data changes."""
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(mod.locked_logged)
    _warm(c, f, 3)
    world = hit_memo.world_epochs()[0]
    mod.locked_console.rate = 3.0
    assert hit_memo.world_epochs()[0] != world
    got, fresh = _state(c, f)
    assert got == fresh
    assert f(3) == f.__wrapped__(3) == 9.0


@pytest.mark.parametrize(
    "swap",
    [
        lambda mod: setattr(mod, "_LOCK", threading.Lock()),
        lambda mod: setattr(mod, "_RLOCK", threading.Lock()),
        lambda mod: setattr(mod.locked_console, "_lock", threading.RLock()),
        lambda mod: setattr(mod.locked_console, "_write_lock", 5),
        lambda mod: setattr(type(mod.console), "_class_lock", threading.Lock()),
    ],
    ids=[
        "global_lock_replaced",
        "rlock_turned_lock",
        "attribute_lock_replaced",
        "attribute_lock_dropped",
        "class_lock_replaced",
    ],
)
def test_which_lock_is_held_is_still_watched(mod, c, monkeypatch, swap):
    """Rebinding a lock is a write to the dict holding it: the world
    moves, and the segment then agrees with a fresh fold (by type: a lock
    swapped for another of its kind keys alike)."""
    monkeypatch.setattr(state_memo, "VERIFY", True)
    fs = [c.cache(mod.guarded), c.cache(mod.logged), c.cache(mod.locked_logged)]
    for f in fs:
        _warm(c, f, 3)
    world = hit_memo.world_epochs()[0]
    swap(mod)
    assert hit_memo.world_epochs()[0] != world
    for f in fs:
        got, fresh = _state(c, f)
        assert got == fresh
        assert f(3) == f.__wrapped__(3)


def test_a_held_lock_keys_as_a_free_one(mod, c):
    """Whether a lock is held is no part of the key: the segment taken
    while the lock is held is the one taken while it is free."""
    f = c.cache(mod.guarded)
    free = _warm(c, f, 3)
    with mod._LOCK, mod._RLOCK, mod.locked_console._lock:
        assert _state(c, f) == (free, free)
