"""The decorated function itself changed in place: its code, defaults or closure.

``f = cache(g); f(1); g.__code__ = other.__code__; f(1)`` served the result
of the old code: the function's own identity is pinned once per function
object, from its source (`FunctionIdentity.pin_own_source`), and assigning
``__code__`` changes neither the object nor the file. Plain Python runs the
new code. The pin now holds the code object it was taken for, and a
function running other code is keyed by the code that runs.

Each change is made between calls in one process, after the call has been
served by the path under test -- the ordinary hit, the remembered state
segment (`state_memo`), the remembered fast hit (`hit_memo`) -- and the call
must then return what plain Python does. Assigning the old code back hits
the entry stored for it again. The state memo runs again sabotaged, taking
every change for none, to show the scenario goes through it.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytestmark = [pytest.mark.core, pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning")]

SOURCE = textwrap.dedent("""
    def plain(x):
        return x + 1

    def other(x):
        return x + 100

    def with_default(x, k=2):
        return x * k

    def with_kwdefault(x, *, k=2):
        return x * k

    def make(n):
        def closed(x):
                return x + n

        return closed

    closed = make(2)

    class Box:
        def __init__(self, v):
            self.v = v

        def __reduce__(self):
            return (Box, (self.v,))

        def __eq__(self, o):
            return isinstance(o, Box) and o.v == self.v

        def __hash__(self):
            return hash(self.v)

        def method(self, x):
                return x + self.v

    def other_method(self, x):
        return x * 1000 + self.v
""")

PATHS = ["slow", "state_memo", "fast", "sabotaged"]


@pytest.fixture(params=PATHS)
def path(request, monkeypatch):
    """Which path serves the warm calls: ``slow`` (no memo), ``state_memo``
    (no fast hit), ``fast`` (both on), ``sabotaged`` (the state memo takes
    every change for none)."""
    kind = request.param
    if kind in ("state_memo", "fast", "sabotaged") and not state_memo.ENABLED:
        pytest.skip("the memos need the C extension")
    if kind != "fast":
        monkeypatch.setattr(hit_memo, "ENABLED", False)
    if kind == "slow":
        monkeypatch.setattr(state_memo, "ENABLED", False)
    if kind == "sabotaged":
        monkeypatch.setattr(state_memo, "VERIFY", False)
        monkeypatch.setattr(state_memo, "_world_epoch", lambda: 1)
        monkeypatch.setattr(state_memo, "moved", lambda: None)
        monkeypatch.setattr(state_memo.StateMemo, "_holds", lambda self: True)
    return kind


@pytest.fixture
def mod(tmp_path):
    name = f"ownchange_{tmp_path.name.replace('-', '_')}"
    file = tmp_path / f"{name}.py"
    file.write_text(SOURCE, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _spec(c, fn):
    return c._registry.cached[fn.__module__ + "." + fn.__qualname__]


def _warm(c, f, fn, path, call):
    """Call until the path under test served a hit."""
    call()
    for _ in range(8):
        reused, fast = state_memo.STATS["reused"], hit_memo.status().get("fast_hits", 0)
        call()
        if path == "slow":
            return
        if path == "fast" and hit_memo.status().get("fast_hits", 0) > fast:
            return
        if path in ("state_memo", "sabotaged") and state_memo.STATS["reused"] > reused:
            return
        if path == "fast" and hit_memo.memo_for(_spec(c, fn), None) is None:
            break
    raise AssertionError(f"the {path} path did not serve a hit: the test proves nothing")


def _misses(f):
    return f.cache_info()["misses"]


def _check(f, path, call, plain, *, segment_only=True):
    """After the change: a miss returning what plain Python does, then a
    hit; sabotaged, the stale segment serves the old result, unless a fold
    outside the segment keys the change too (*segment_only* false)."""
    misses = _misses(f)
    got = call()
    if path == "sabotaged":
        # Stale; or, caught outside the segment, right (and, the stale
        # segment's folded code differing from what ran, not stored).
        assert got != plain() if segment_only else got == plain()
        return
    assert got == plain()
    assert _misses(f) == misses + 1
    assert call() == plain()
    assert _misses(f) == misses + 1


def test_code_assigned(c, mod, path):
    f = c.cache(mod.plain)
    _warm(c, f, mod.plain, path, lambda: f(1))
    old = mod.plain.__code__
    mod.plain.__code__ = mod.other.__code__
    _check(f, path, lambda: f(1), lambda: 101)
    mod.plain.__code__ = old
    if path != "sabotaged":
        misses = _misses(f)
        assert f(1) == 2
        assert _misses(f) == misses  # the old code's entry, hit again


def test_method_code_assigned(c, mod, path):
    if path == "fast":
        pytest.skip("a call with an instance argument is not served by the fast hit path")
    mod.Box.method = c.cache(mod.Box.method)
    fn = mod.Box.method.__wrapped__
    box = mod.Box(5)
    _warm(c, mod.Box.method, fn, path, lambda: box.method(1))
    fn.__code__ = mod.other_method.__code__
    # The class's methods are folded again with the instance's class, outside
    # the remembered segment, so the sabotaged memo is caught there too.
    _check(mod.Box.method, path, lambda: box.method(1), lambda: 1005, segment_only=False)


def test_defaults_assigned(c, mod, path):
    f = c.cache(mod.with_default)
    _warm(c, f, mod.with_default, path, lambda: f(3))
    mod.with_default.__defaults__ = (5,)
    _check(f, path, lambda: f(3), lambda: 15)


def test_kwdefaults_assigned(c, mod, path):
    f = c.cache(mod.with_kwdefault)
    _warm(c, f, mod.with_kwdefault, path, lambda: f(3))
    mod.with_kwdefault.__kwdefaults__ = {"k": 7}
    _check(f, path, lambda: f(3), lambda: 21)


def test_closure_cell_assigned(c, mod, path):
    f = c.cache(mod.closed)
    _warm(c, f, mod.closed, path, lambda: f(3))
    mod.closed.__closure__[0].cell_contents = 40
    _check(f, path, lambda: f(3), lambda: 43)
