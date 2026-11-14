"""`in_place.refusal` gives the same verdict, reading a list of frames faster.

A loop over sessions growing a list of small frames (``frames.append(new)``)
asks at every end whether ``frames`` can be restored into, and is refused
(too many objects, or a container too big to look through) or accepted
after walking every frame and then the notebook. That walk cost each loop
end about 0.35 s: each of the 14 objects of a small frame was asked whether
it was a stream, a function or a pandas object, and the ``str`` and ``hex``
of every ``id()`` met were spelled for dicts whose keys are short names.
How a class is read is now settled once, and the ids are spelled only when
a text of their length is met. What it says of each value is what it said:
which objects are looked at, in which order, and so which reason it gives.
"""

from __future__ import annotations

import sys
import time
import types
import weakref

import pytest

from cash.notebook.statement import in_place

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _frames(count: int) -> list:
    frames = []
    for i in range(count):
        new = pd.DataFrame({"t": range(3), "v": np.arange(3) * 0.5})
        new["sid"] = i
        frames.append(new)
    return frames


class _Box:
    pass


class _Disguised:
    """Says it is a function: ``isinstance`` takes it at its word."""

    @property
    def __class__(self):
        return types.FunctionType


class _SameText:
    """Hashes and compares as the text it carries."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other):
        return other == self.text

    def __hash__(self):
        return hash(self.text)


def _holding_itself(value):
    """A list holding *value*, which holds the list back."""
    held = []
    value.back = held
    held.append(value)
    return held


ACCEPTED = None
ITSELF = "it holds itself, and a copy would hold the copy"
TOO_MANY = "it holds too many objects to check that it does not hold itself"
WEAK = "a weak reference points at an object it holds, and a restore would hand it a copy"
BY_ID = "a container keeps the id() of an object it holds, and a restore would hand it a copy"
TOO_BIG = "the notebook holds containers too big to look through ('{}') for a weak reference to, or the id() of, an object it holds"


def _corpus():
    """(name, value, namespace, the verdict the walk gave before)."""
    small = _frames(5)
    many = _frames(40)
    others = _frames(60)
    looped = _frames(10)
    yield "a few frames", small, {"frames": small}, ACCEPTED
    yield "a list of frames past the bound", many, {"frames": many}, TOO_MANY
    yield (
        "frames beside a list of frames too big to look through",
        looped,
        {"others": others, "frames": looped},
        TOO_BIG,
    )
    column_holder: list = []
    column_holder.append(pd.DataFrame({"o": [column_holder, 1]}))
    yield "an object column holding its list", column_holder, {"rows": column_holder}, ITSELF
    arrays = {f"a{i}": np.arange(30) for i in range(20)}
    yield "a dict of arrays", arrays, {"d": arrays}, ACCEPTED
    keyed = {f"a{i}": np.arange(3) for i in range(5)}
    keyed["ids"] = np.array([id(keyed["a3"])])
    yield "a dict of arrays keeping an id in one", keyed, {"d": keyed}, BY_ID
    for label, keep in (
        ("an id as a value", lambda f: {"k": id(f)}),
        ("an id as text", lambda f: [str(id(f.index))]),
        ("an id as hex", lambda f: [hex(id(f.columns))]),
        ("a key comparing as an id's text", lambda f: {_SameText(str(id(f))): 1}),
        ("an id in a small integer array", lambda f: np.array([1, id(f)], dtype=np.int64)),
        ("an id in a large integer array", lambda f: np.r_[np.zeros(500, dtype=np.int64), id(f)]),
    ):
        frames = _frames(6)
        yield label, frames, {"frames": frames, "keep": keep(frames[2])}, BY_ID
    weak = _frames(6)
    yield "a weak reference to a frame", weak, {"frames": weak, "w": weakref.ref(weak[3])}, WEAK
    ids_like = _frames(4)
    texts = {"ids": [str(10**14 + i) for i in range(3000)]}
    yield "many texts as long as an id", ids_like, {"frames": ids_like, "texts": texts}, ACCEPTED
    box = _Box()
    box.items, box.me = _frames(3), box
    yield "an object holding itself", box, {"b": box}, ITSELF
    plain = _Box()
    plain.items, plain.d = _frames(3), {"x": [1, 2]}
    yield "a plain object", plain, {"b": plain}, ACCEPTED
    yield "an object passing for a function", _holding_itself(_Disguised()), {}, ACCEPTED
    yield "a standard stream", [sys.__stdout__, _frames(2)], {}, ACCEPTED
    series = pd.Series([1, 2, 3])
    series.attrs["x"] = None
    yield "a series and an index", [series, pd.Index(["a", "b"])], {}, ACCEPTED


@pytest.fixture
def small_bounds(monkeypatch):
    """300 objects and 50 items a container are past the bounds: 40 frames
    are past them as 15,000 are past the real ones."""
    monkeypatch.setattr(in_place, "_MAX_SEEN", 300)
    monkeypatch.setattr(in_place, "_MAX_SCANNED", 50)


@pytest.mark.parametrize("case", list(range(len(list(_corpus())))))
def test_each_value_gets_the_verdict_it_got(small_bounds, case):
    name, value, namespace, expected = list(_corpus())[case]
    if expected == TOO_BIG:
        expected = TOO_BIG.format(next(n for n in namespace if n != "frames"))
    assert in_place.refusal(value, namespace) == expected, name


def _loop_shape(count: int):
    logs = [{"uid": k, "actions": [{"t": j, "v": (k * 7 + j) % 11 * 0.5} for j in range(3)]} for k in range(count)]
    frames = []
    started = time.perf_counter()
    for i, s in enumerate(logs):
        new = pd.DataFrame(s["actions"])
        new["sid"] = i
        frames.append(new)
    built = time.perf_counter() - started
    return frames, {"logs": logs, "frames": frames, "new": new, "s": s}, built


def test_the_loops_frames_are_read_a_class_at_a_time(monkeypatch):
    """Of 3,000 frames, each class is read once, and no id is spelled."""
    frames, namespace, _ = _loop_shape(3000)
    modes = []
    spelled = []
    real_mode, real_all = in_place._reach_mode, in_place._Texts.all

    def mode(kind, *args):
        modes.append(kind)
        return real_mode(kind, *args)

    def spell(self):
        spelled.append(len(self.parts))
        return real_all(self)

    monkeypatch.setattr(in_place, "_reach_mode", mode)
    monkeypatch.setattr(in_place._Texts, "all", spell)
    assert in_place.refusal(frames, namespace) is None
    assert len(modes) == len(set(modes)) < 30
    assert spelled == []


def test_the_loops_refusal_costs_a_fraction_of_making_its_frames():
    """Asked at the end of a loop that made 3,000 small frames, the check
    costs less than half of making them: here a quarter on Python 3.13
    (0.19 s, a third before) and a third on 3.10. Which work it skips is
    pinned by the test above; this bounds what is left."""
    frames, namespace, built = _loop_shape(3000)
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        assert in_place.refusal(frames, namespace) is None
        best = min(best, time.perf_counter() - started)
    assert best < built / 2, (best, built)
