"""Code a left-out global holds where no watcher sees is followed by the state memo.

A library's object that pickles as its ``__dict__`` and holds a lock there
(``rich``'s console) cannot be hashed: read whole (``c = console``), the
key leaves its value out with KEY-UNHASHABLE-GLOBAL but still keys the
user code it holds, and what that code reads (`GlobalsFold`'s held-code
walk), wherever it is held: in a list in its ``__dict__``, or in the
``threading.local`` it keeps. The state memo watched such a global only
for its class and its ``__dict__``, so code put into the thread-local or
the list moved no watcher and the memo served the segment built for the
code held before: a stale result, and StateMemoMismatch under
``CASH_STATE_MEMO_VERIFY``. The scan now finds that code again before
every reuse, on the thread reusing, with the key's own walk.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import threading

import pytest

from cash import Cash
from cash.decorator import state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

LIBRARY = textwrap.dedent("""
    import threading


    class Console:
        def __init__(self):
            self._lock = threading.RLock()
            self._local = threading.local()
            self.hooks = []


    console = Console()
""")

USER = textwrap.dedent("""
    from {lib} import console

    SCALE = 1

    def double(x):
        return 2 * x * SCALE

    def triple(x):
        return 3 * x

    def via_local(x):
        c = console
        return c._local.hook(x)

    def via_list(x):
        c = console
        return c.hooks[0](x)

    def plain(x):
        c = console
        return x + len(c.hooks)
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    tag = tmp_path.name.replace("-", "_")
    lib = f"heldlib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"helduser_{tag}"
    (tmp_path / f"{name}.py").write_text(USER.format(lib=lib), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    m = importlib.import_module(name)
    m.console._local.hook = m.double
    m.console.hooks.append(m.double)
    yield m
    sys.modules.pop(name, None)
    sys.modules.pop(lib, None)


@pytest.fixture(params=[False, True], ids=["served", "verified"])
def c(request, monkeypatch):
    """A Cash whose state memo is checked against a fresh fold on every
    reuse, or not: unchecked, a stale segment is served as it is."""
    monkeypatch.setattr(state_memo, "VERIFY", request.param)
    return Cash(backend="memory", register_magic=False)


def _warm(f, *args):
    """Call *f* until its state segment is reused."""
    for _ in range(6):
        before = state_memo.STATS["reused"]
        f(*args)
        if state_memo.STATS["reused"] > before:
            return
    raise AssertionError("the segment is not being reused")


@pytest.mark.parametrize(
    ("name", "swap"),
    [
        ("via_local", lambda mod: setattr(mod.console._local, "hook", mod.triple)),
        ("via_list", lambda mod: mod.console.hooks.__setitem__(0, mod.triple)),
    ],
    ids=["thread_local", "list"],
)
def test_code_swapped_inside_it_is_run_not_served_stale(mod, c, name, swap):
    f = c.cache(getattr(mod, name))
    _warm(f, 5)
    assert f(5) == 10
    swap(mod)
    assert f(5) == getattr(mod, name)(5) == 15
    assert "KEY-UNHASHABLE-GLOBAL" in [w["code"] for w in f.cache_info()["warnings"]]


def test_a_global_the_held_code_reads_is_followed(mod, c):
    f = c.cache(mod.via_local)
    _warm(f, 5)
    mod.SCALE = 4
    assert f(5) == mod.via_local(5) == 40


def test_another_thread_is_served_what_its_own_thread_local_holds(mod, c):
    f = c.cache(mod.via_local)
    _warm(f, 5)
    got: list = []

    def other():
        mod.console._local.hook = mod.triple
        got.append(f(5))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert got == [15]
    assert f(5) == 10


def test_nothing_moved_inside_it_is_still_reused(mod, c):
    """The positive control: with the same code held, a write the key does
    not read (data in the thread-local, an item of no code) reuses."""
    f = c.cache(mod.plain)
    _warm(f, 5)
    mod.console._local.level = 3
    before = state_memo.STATS["reused"]
    assert f(5) == 6
    assert state_memo.STATS["reused"] > before
