"""A state segment that reads a frame, an enum or the global random stream
is reused while what it read holds.

Three kinds of global kept a function's state segment (`state_memo`) from
being reused, so every call built it again:

* data no watcher sees changed in place, beyond lists and arrays: a pandas
  frame or series of plain columns, an ``OrderedDict``, a number or string
  of a subclass, and an enum's members. The scan now reads such data again
  before each reuse (`hit_memo.ContentProbe`), or watches what holds it;
* the global random stream (``np.random.normal``, ``random.random``): its
  position moves with every draw, so the scan refused it. What a key reads
  of it is its seed epoch, now folded outside the remembered part
  (`KeyBuilder._code_state_post`), on every call, with the environment and
  ``key=``;
* ``warnings.catch_warnings(record=True)`` in a body: it swaps the display
  hooks of ``warnings`` and back, which moved the world on every call
  unless the scan that watched that namespace said they were quiet.

Each scenario checks the segment IS reused first, then changes one thing the
fold read between two calls in one process, and checks the segment is a
fresh fold's and the call returns what plain Python does. Each runs again
against a sabotaged memo that takes every change for none, which must serve
the stale segment: that is what shows the scenario goes through the memo.
"""

from __future__ import annotations

import collections
import enum
import random
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo
from cash.decorator.runtime import KeyBuilder

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.filterwarnings("ignore::cash.tracking.randomness.detect.CashRandomnessWarning"),
]


def _prices():
    return pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.5],
            "volume": [100, 200, 300],
            "symbol": pd.array(["a", "b", "c"], dtype="string"),
            "day": pd.date_range("2024-01-01", periods=3),
        }
    )


PRICES = _prices()
RETURNS = pd.Series([0.1, -0.2, 0.05], name="r")
ORDER = collections.OrderedDict(a=1, b=2)


class Meters(float):
    pass


LENGTH = Meters(2.5)


class Side(enum.Enum):
    BUY = 1
    SELL = -1


def priced(i):
    return (
        float(PRICES["close"].iloc[i]),
        int(PRICES["volume"].iloc[i]),
        PRICES.shape,
        str(PRICES.dtypes.iloc[1]),
        dict(PRICES.attrs),
        PRICES.index.name,
    )


def returned(i):
    return float(RETURNS.iloc[i]) + len(RETURNS)


def ordered(x):
    return (x, list(ORDER.items()))


def measured(x):
    return (x * LENGTH, sorted(getattr(LENGTH, "__dict__", {})))


def sided(x):
    return x * Side.BUY.value


def drawn(x):
    return x + float(np.random.normal())


def picked(x):
    return x + random.random()


def shown(x):
    return (x, warnings.showwarning.__name__)


@pytest.fixture(autouse=True)
def _put_back():
    global PRICES, RETURNS, Side
    saved = (RETURNS.copy(), list(ORDER.items()), Side)
    yield
    PRICES = _prices()
    RETURNS = saved[0]
    ORDER.clear()
    ORDER.update(saved[1])
    LENGTH.__dict__.clear()
    Side = saved[2]


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


def _cell_set():
    PRICES.iloc[1, 0] = 99.0


def _column_added():
    PRICES["extra"] = 1


def _attrs_set():
    PRICES.attrs["unit"] = "usd"


def _index_renamed():
    PRICES.index.name = "row"


def _dtype_changed():
    PRICES["volume"] = PRICES["volume"].astype("float64")


@pytest.mark.parametrize(
    "change",
    [_cell_set, _column_added, _attrs_set, _index_renamed, _dtype_changed],
    ids=["cell_set", "column_added", "attrs_set", "index_renamed", "dtype_changed"],
)
def test_a_frame_changed_in_place(c, memo_kind, change):
    f = c.cache(priced)
    old = _warm(c, f, 1)
    change()
    _check(c, f, memo_kind, old, 1)


def test_a_series_changed_in_place(c, memo_kind):
    f = c.cache(returned)
    old = _warm(c, f, 0)
    RETURNS.iloc[0] = 7.0
    _check(c, f, memo_kind, old, 0)


@pytest.mark.parametrize(
    "change",
    [
        lambda: ORDER.__setitem__("c", 3),
        lambda: ORDER.move_to_end("a"),
        lambda: ORDER.__setitem__("a", 1.0),  # equal, but keyed as another value
    ],
    ids=["item_added", "reordered", "int_to_float"],
)
def test_an_ordered_dict_changed_in_place(c, memo_kind, change):
    f = c.cache(ordered)
    old = _warm(c, f, 1)
    change()
    _check(c, f, memo_kind, old, 1)


def test_a_number_of_a_subclass_given_an_attribute(c, memo_kind):
    f = c.cache(measured)
    old = _warm(c, f, 2)
    LENGTH.unit = "cm"
    _check(c, f, memo_kind, old, 2)


def test_an_enum_redefined_with_another_value(c, memo_kind):
    global Side
    f = c.cache(sided)
    old = _warm(c, f, 3)

    class Side(enum.Enum):  # the redefinition is the change
        BUY = 2
        SELL = -2

    _check(c, f, memo_kind, old, 3)


def test_frames_with_objects_or_attrs_are_not_probed():
    class Box:
        pass

    assert hit_memo.ContentProbe.of(pd.DataFrame({"a": [Box()]})) is None
    assert hit_memo.ContentProbe.of(pd.DataFrame({"a": [1]}, index=pd.Index([Box()]))) is None
    framed = pd.DataFrame({"a": [1]})
    framed.attrs["k"] = [1]
    assert hit_memo.ContentProbe.of(framed) is None
    assert hit_memo.ContentProbe.of(collections.OrderedDict(a=Box())) is None
    assert hit_memo.ContentProbe.of(_prices()) is not None


def test_rebinding_a_warnings_hook_that_scanned_code_names_moves_the_world(c, monkeypatch):
    """The display hooks are quiet only for code that does not name them:
    once a scanned function reads ``warnings.showwarning``, rebinding it
    moves the world epoch, and the segment is checked again: it agrees
    with a fresh fold."""
    f = c.cache(shown)
    _warm(c, f, 1)
    world = hit_memo.world_epochs()[0]

    def my_hook(*args, **kwargs):
        return None

    monkeypatch.setattr(warnings, "showwarning", my_hook)
    assert hit_memo.world_epochs()[0] != world
    got, fresh = _state(c, f)
    assert got == fresh


QUIET = textwrap.dedent("""
    import sys, warnings
    warnings.simplefilter("ignore")
    from cash import Cash
    from cash.decorator import helper_fold_memo, hit_memo

    class Thing:
        def __init__(self):
            self.k = 2

    OBJ = Thing()

    def helper(x):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return x + 1

    def g(x):
        return helper(x) * OBJ.k

    f = Cash(backend="memory", register_magic=False).cache(g)
    for i in range(30):
        assert f(i) == g(i)
    print(hit_memo.status()["bumps"], helper_fold_memo.STATS["served"])
""")


def test_catch_warnings_in_a_body_leaves_the_world_still():
    """A function whose own scan stops early (it reads an instance) still
    has its helper's fold remembered while that helper's body records
    warnings, in a process where no scan reached the module watch first:
    the display hooks it swaps are quiet for every scan."""
    out = subprocess.run([sys.executable, "-c", QUIET], capture_output=True, text=True, check=True)
    bumps, served = map(int, out.stdout.split())
    assert bumps < 5, out.stdout  # two per call without the quiet hooks
    assert served >= 25, out.stdout


@pytest.mark.parametrize("func", [drawn, picked], ids=["numpy", "stdlib"])
def test_a_function_drawing_from_the_global_stream_is_reused_and_keyed_by_its_seed(c, monkeypatch, func):
    monkeypatch.setattr(state_memo, "VERIFY", True)
    seed = np.random.seed if func is drawn else random.seed
    f = c.cache(func)
    seed(1)
    f(1)
    seed(1)
    f(1)  # the draw is learned: keyed by the seed epoch from here on
    seed(1)
    old = _warm(c, f, 1)
    values = {}
    for s in (1, 2, 1):
        seed(s)
        plain = func(1)
        seed(s)
        assert f(1) == plain
        values.setdefault(s, plain)
    assert values[1] != values[2]
    seed(2)
    got, fresh = _state(c, f)
    assert got == fresh != old
    assert state_memo.STATS["verified"] > 0


def test_the_seed_epoch_kept_in_the_remembered_part_would_be_stale(c, monkeypatch):
    """Sabotage: the whole segment, the seed epoch with it, remembered as
    one. Seeding again then serves the old segment; that is what folding
    the epoch outside the remembered part guards against."""
    monkeypatch.setattr(state_memo, "VERIFY", False)
    monkeypatch.setattr(state_memo, "_outside", lambda: 0)
    monkeypatch.setattr(state_memo, "_PRE_STAGES", 7)

    def whole(self, spec, chain, *, note=False):
        keys = type("Keys", (), {"_code_state_pre": self._code_state_now, "_registry": self._registry})()
        return state_memo.memo_for(spec).code_state(keys, spec, chain, note)

    monkeypatch.setattr(KeyBuilder, "_code_state", whole)
    f = c.cache(drawn)
    for _ in range(2):
        np.random.seed(1)
        f(1)
    np.random.seed(1)
    old = _warm(c, f, 1)
    np.random.seed(2)
    got, fresh = _state(c, f)
    assert fresh != old
    assert got == old
