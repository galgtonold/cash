"""A state segment that folds a list or an array is reused while its content holds.

No watcher sees a list, a set or a numpy array changed in place, so a
function whose globals held one built its state segment again on every call
(`state_memo`): for a module with a numpy table and a list of weights, every
hit and every miss folded the whole helper tree. The scan now reads such data
whole again before each reuse (`hit_memo.ContentProbe`) and reuses the
segment only while it is what it was.

Each scenario checks the segment IS reused first, then changes one thing the
fold read, between two calls in one process -- the data in place, a helper
redefined, a helper's code swapped, a library attribute patched, a closure
cell -- and checks the segment is a fresh fold's and the call returns what
plain Python does. Each runs again against a sabotaged memo that takes every
change for none, which must serve the stale segment: that is what shows the
scenario goes through the memo at all.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import textwrap
import threading
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

np = pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    # Setting an array's shape in place is deprecated, not gone: still a change.
    pytest.mark.filterwarnings("ignore:Setting the shape:DeprecationWarning"),
]

WEIGHTS = [1, 2, 3]
TAGS = {"a", "b"}
TABLE = np.arange(6.0)
CONF = {"w": [1, 2]}


def weighted(x):
    return x * sum(WEIGHTS)


def tagged(x):
    return x + len(TAGS) + sorted(TAGS)[0].__len__()


def looked_up(i):
    return float(TABLE.flat[i]) + TABLE.shape[0]


def nested(x):
    return x + sum(CONF["w"])


def with_tau(x):
    return x * math.tau + sum(WEIGHTS)


def make_scaled(scale):
    def scaled(x):
        return x * scale + sum(WEIGHTS)

    return scaled


class Box:
    pass


HELD = [Box()]


def holds_an_object(x):
    return x + len(HELD)


@pytest.fixture(autouse=True)
def _put_back():
    global TABLE
    saved = (list(WEIGHTS), set(TAGS), TABLE, TABLE.copy(), CONF["w"][:])
    yield
    WEIGHTS[:] = saved[0]
    TAGS.clear()
    TAGS.update(saved[1])
    TABLE = saved[2]
    TABLE.shape = (6,)
    TABLE[:] = saved[3]
    CONF["w"][:] = saved[4]


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


def _spec(c, f):
    return c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = _spec(c, f)
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
        lambda: WEIGHTS.append(4),
        lambda: WEIGHTS.__setitem__(0, 10),
        lambda: WEIGHTS.__setitem__(0, 1.0),  # equal, but keyed as another value
    ],
    ids=["appended", "item_set", "int_to_float"],
)
def test_a_list_changed_in_place(c, memo_kind, change):
    f = c.cache(weighted)
    old = _warm(c, f, 2)
    change()
    _check(c, f, memo_kind, old, 2)


def test_a_set_changed_in_place(c, memo_kind):
    f = c.cache(tagged)
    old = _warm(c, f, 1)
    TAGS.add("A")
    _check(c, f, memo_kind, old, 1)


@pytest.mark.parametrize(
    "change",
    [
        lambda: TABLE.__setitem__(2, 99.0),
        lambda: setattr(TABLE, "shape", (2, 3)),
        lambda: TABLE.__setitem__(slice(None), TABLE[::-1].copy()),
    ],
    ids=["element", "reshaped_in_place", "reordered"],
)
def test_an_array_changed_in_place(c, memo_kind, change):
    f = c.cache(looked_up)
    old = _warm(c, f, 2)
    change()
    _check(c, f, memo_kind, old, 1)


def test_a_list_held_in_a_dict_changed_in_place(c, memo_kind):
    f = c.cache(nested)
    old = _warm(c, f, 1)
    CONF["w"].append(5)
    _check(c, f, memo_kind, old, 1)


def test_a_rebound_global(c, memo_kind):
    global TABLE
    f = c.cache(looked_up)
    old = _warm(c, f, 2)
    TABLE = np.arange(6.0) * 2
    _check(c, f, memo_kind, old, 2)


def test_a_patched_library_attribute(c, memo_kind, monkeypatch):
    f = c.cache(with_tau)
    old = _warm(c, f, 1)
    monkeypatch.setattr(math, "tau", 3.0)
    _check(c, f, memo_kind, old, 1)


def test_a_changed_closure_cell(c, memo_kind):
    f = c.cache(make_scaled(2))
    old = _warm(c, f, 1)
    f.__wrapped__.__closure__[0].cell_contents = 7
    _check(c, f, memo_kind, old, 1)


MODULE = textwrap.dedent("""
    import numpy as np

    GRID = np.linspace(0.0, 1.0, 11)
    STEPS = [1, 2]

    def helper(x):
        return x + sum(STEPS)

    def study(i):
        return helper(i) + float(GRID[i])
""")


@pytest.fixture
def study_module(tmp_path, monkeypatch):
    name = f"study_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)


def test_a_helper_redefined_in_process(c, memo_kind, study_module):
    f = c.cache(study_module.study)
    old = _warm(c, f, 3)
    exec("def helper(x):\n    return x * 100 + sum(STEPS)\n", study_module.__dict__)
    _check(c, f, memo_kind, old, 3)


def test_a_helpers_code_swapped(c, memo_kind, study_module):
    f = c.cache(study_module.study)
    old = _warm(c, f, 3)
    scope: dict = {"STEPS": study_module.STEPS}
    exec("def other(x):\n    return x - sum(STEPS)\n", scope)
    study_module.helper.__code__ = scope["other"].__code__
    _check(c, f, memo_kind, old, 3)


def test_data_holding_an_object_is_still_built_every_time(c):
    """A list holding an instance is no plain data: nothing proves it
    unchanged, and the segment is folded on every call."""
    f = c.cache(holds_an_object)
    f(1)
    before = state_memo.STATS["reused"]
    for _ in range(3):
        got, fresh = _state(c, f)
        assert got == fresh
    assert state_memo.STATS["reused"] == before
    HELD.append(Box())
    try:
        assert f(1) == f.__wrapped__(1) == 3
    finally:
        del HELD[1:]


def test_an_array_of_objects_is_not_probed():
    assert hit_memo.ContentProbe.of(np.array([Box()], dtype=object)) is None
    assert hit_memo.ContentProbe.of([Box()]) is None
    assert hit_memo.ContentProbe.of(np.zeros(3)) is not None


def test_verify_mode_agrees_on_every_reuse(c, monkeypatch):
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(looked_up)
    before = state_memo.STATS["verified"]
    for i in range(6):
        assert f(i) == f.__wrapped__(i)
    TABLE[0] = -1.0
    WEIGHTS.append(9)
    for i in range(6):
        assert f(i) == f.__wrapped__(i)
    assert state_memo.STATS["verified"] > before


def handshake(x, signal_fd, wait_fd):
    """Reads WEIGHTS; between the key and the end of the body, tells the
    test it started and waits for it to say the list changed."""
    os.write(signal_fd, b"s")  # @cash:assume-safe
    os.read(wait_fd, 1)  # @cash:assume-safe
    return x * sum(WEIGHTS)


def test_a_list_another_thread_changes_during_the_body_is_checked_again(disk_cash):
    """With another thread alive, a miss checks the segment again after
    the body (`CallRunner._inputs_moved_check`); that check reuses the
    segment while nothing moved, and a list changed mid-call is caught
    there: the result is returned, not stored under the old key."""
    f = disk_cash.cache(handshake)
    started_r, started_w = os.pipe()
    go_r, go_w = os.pipe()
    idle = threading.Event()
    other = threading.Thread(target=idle.wait, daemon=True)
    other.start()

    def call(x, *, hit=False):
        """*f(x)*: a miss's body says it started and waits to go on."""
        if hit:
            return f(x, started_w, go_r)
        os.write(go_w, b"g")
        got = f(x, started_w, go_r)
        os.read(started_r, 1)
        return got

    try:
        for x in (1, 2):
            assert call(x) == x * 6
        before = state_memo.STATS["reused"]
        assert call(3) == 18
        assert state_memo.STATS["reused"] >= before + 2, "the key and the check after the body both reuse it"
        out: list = []
        t = threading.Thread(target=lambda: out.append(f(10, started_w, go_r)))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            t.start()
            os.read(started_r, 1)
            WEIGHTS.append(4)
            os.write(go_w, b"g")
            t.join(10)
        assert out == [100]
        assert any("STORE-STATE-CHANGED" in str(w.message) for w in rec), [str(w.message) for w in rec]
        assert call(10) == 100
        assert call(10, hit=True) == 100 and f.cache_info()["hits"] == 1
    finally:
        idle.set()
        for fd in (started_r, started_w, go_r, go_w):
            os.close(fd)
