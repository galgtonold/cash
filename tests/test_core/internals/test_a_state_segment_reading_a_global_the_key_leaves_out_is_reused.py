"""A state segment reading a global the key leaves out is reused.

A library's object that pickles as its ``__dict__`` and holds a lock there
(``rich``'s console, which nilearn's ``resample_img`` reaches through a
module global) cannot be hashed: the key leaves the global out with
KEY-UNHASHABLE-GLOBAL and the function is still cached. The world watch
(`cash.decorator.hit_memo._Scan`) looked into it all the same, met what
it holds per thread (a ``threading.local``) and refused it, so every call
folded the state segment again (`state_memo`).

The scan now watches such a global for what keeps it left out, its class
and its ``__dict__``, and looks into it only for the user code a key
finds there (`test_code_a_left_out_global_holds_where_no_watcher_sees_is_followed_by_the_state_memo`). Every reuse here is
checked against a fresh fold (``CASH_STATE_MEMO_VERIFY``): after a write
the watch sees, after one it does not (into the thread-local), after
the lock is dropped so the object is keyed again, and with a hasher
registered, which may read what the object holds.
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
        def __init__(self):
            self.rate = 2.0
            self._lock = threading.RLock()
            self._local = threading.local()

        def log(self, x):
            with self._lock:
                return x


    console = Console()
""")

USER = textwrap.dedent("""
    from {lib} import console

    OFFSET = 1.0

    def logged(x):
        return console.log(x) + OFFSET

    HOOK_ONLY = 1

    def _hook():
        return HOOK_ONLY
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    tag = tmp_path.name.replace("-", "_")
    lib = f"leftlib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"leftuser_{tag}"
    (tmp_path / f"{name}.py").write_text(USER.format(lib=lib), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)
    sys.modules.pop(lib, None)


@pytest.fixture
def c(monkeypatch):
    monkeypatch.setattr(state_memo, "VERIFY", True)
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


def _codes(f):
    return [w["code"] for w in f.cache_info()["warnings"]]


def test_the_segment_is_reused_and_the_global_still_warned_about(mod, c):
    f = c.cache(mod.logged)
    _warm(c, f, 3)
    verified = state_memo.STATS["verified"]
    assert f(3) == f(3) == 4.0
    assert state_memo.STATS["verified"] > verified
    assert "KEY-UNHASHABLE-GLOBAL" in _codes(f)


@pytest.mark.parametrize(
    "change",
    [
        lambda mod: setattr(mod.console, "rate", 3.0),
        lambda mod: setattr(mod.console._local, "level", 5),
        lambda mod: setattr(mod.console._local, "hook", mod._hook),
        lambda mod: setattr(mod.console, "_lock", threading.Lock()),
        lambda mod: setattr(mod, "console", type(mod.console)()),
    ],
    ids=["attribute_set", "thread_local_set", "thread_local_holds_code", "lock_replaced", "global_rebound"],
)
def test_a_change_inside_or_around_it_serves_what_a_fresh_fold_builds(mod, c, change):
    f = c.cache(mod.logged)
    old = _warm(c, f, 3)
    change(mod)
    got, fresh = _state(c, f)
    assert got == fresh == old  # left out before and after
    mod.OFFSET = 2.0
    got, fresh = _state(c, f)
    assert got == fresh != old
    assert f(3) == f.__wrapped__(3) == 5.0


def test_code_put_where_the_watch_does_not_look_changes_nothing_served(mod, c):
    """A function of the user's put into the thread-local, which no
    watcher sees: the key, which leaves the object out, reads nothing of
    it, so a global only that function reads moves neither segment."""
    f = c.cache(mod.logged)
    old = _warm(c, f, 3)
    mod.console._local.hook = mod._hook
    assert _state(c, f) == (old, old)
    mod.HOOK_ONLY = 2
    assert _state(c, f) == (old, old)
    assert f(3) == f.__wrapped__(3) == 4.0


def test_dropping_what_kept_it_out_keys_it_again(mod, c):
    """With the lock and the thread-local gone the object pickles: the
    write moves the world, and the segment is the fresh fold that keys
    it."""
    f = c.cache(mod.logged)
    old = _warm(c, f, 3)
    world = hit_memo.world_epochs()[0]
    del mod.console._lock, mod.console._local
    mod.console.log = lambda x: x  # what ``log`` would need of the lock
    assert hit_memo.world_epochs()[0] != world
    got, fresh = _state(c, f)
    assert got == fresh != old


def test_a_registered_hasher_reading_what_it_holds_is_followed(mod, c):
    """A hasher registered for its type keys the object whole, thread-local
    and all: the scan no longer takes it as left out, so a write into the
    thread-local is never served stale."""
    c.register_hasher(type(mod.console), lambda con: repr(sorted(vars(con._local).items())))
    f = c.cache(mod.logged)
    f(3)
    first = _state(c, f)
    mod.console._local.level = 7
    got, fresh = _state(c, f)
    assert got == fresh != first[1]
