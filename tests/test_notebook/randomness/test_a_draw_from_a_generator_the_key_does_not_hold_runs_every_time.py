"""A draw from a generator held where cash never saw it made is not stored.

A generator bound to a variable is followed: its draw moves the variable's
lineage, and a hit puts it where the run left it. One held deeper in an
object built elsewhere -- ``tlib.MODEL.rng`` in a module of the user's, a
helper holding one in its closure -- was not: the drawing statement was
stored, so after a restart a hit served the draw and left the generator
where it was, and the next draw below gave the number the hit stood for
instead of the one plain Python gives. Such a run is now not stored, silently,
so the draws below match a run without cash.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings

import numpy as np
import pytest

from cash.notebook.call_key import changes_its_closure, holds_a_stream
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"

TLIB = (
    "import time\nimport numpy as np\n"
    "class M:\n"
    "    def __init__(self):\n        self.rng = np.random.default_rng(0)\n"
    f"    def draw(self):\n        {SLEEP}\n        return float(self.rng.normal())\n"
    "MODEL = M()\n"
    "def make_helper():\n"
    "    g = np.random.default_rng(0)\n"
    f"    def helper():\n        {SLEEP}\n        return float(g.normal())\n"
    "    def peek():\n        return float(g.normal())\n"
    "    return helper, peek\n"
)

CASES = {
    "an object a module built": ["import tlib", "x = tlib.MODEL.draw()", "y = float(tlib.MODEL.rng.normal())"],
    "a helper's closure": ["import tlib\nh, peek = tlib.make_helper()", "x = h()", "y = peek()"],
}


def _plain() -> tuple[float, float]:
    r = np.random.default_rng(0)
    return float(r.normal()), float(r.normal())


@pytest.fixture
def tlib(tmp_path, monkeypatch):
    (tmp_path / "tlib.py").write_text(TLIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("tlib", None)


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("tlib", "h", "peek", "x", "y", "time", "np", "draw", "rng"):
        magics.shell.user_ns.pop(name, None)
    sys.modules.pop("tlib", None)


def _run_all(magics, cells) -> list[str]:
    """Each cell's last statement's status; no warning is said for a
    generator the key does not hold."""
    statuses = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
            statuses.append(str(magics.cash_status("dict")["last_cell"]["statements"][-1].get("status")))
    assert not [w for w in caught if str(getattr(w.message, "code", "")).startswith("RANDOM-HIDDEN")]
    return statuses


@pytest.fixture
def every_call_stored(monkeypatch):
    """Every call past the cost floor, as on a busy machine: ``h, peek =
    tlib.make_helper()`` took over 3 ms there, and only then was it stored."""
    from cash.notebook.call_unit import CallUnit

    monkeypatch.setattr(CallUnit, "_cost_floor_s", lambda self: 0.0)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_draw_below_matches_plain_after_a_restart(cash_magics, tlib, every_call_stored, case):
    """Stored, the factory's result was handed back after the restart as the
    functions the last run made, their generator moved on by its draws: x
    was the second number. A closure holding a generator is not stored."""
    cells = CASES[case]
    _run_all(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert (ns["x"], ns["y"]) == _plain(), "control: the first run"
    _restart(cash_magics)
    statuses = _run_all(cash_magics, cells)
    assert "RESTORED" not in statuses[1], "the draw was stored"
    assert (ns["x"], ns["y"]) == _plain()


def test_a_generator_in_a_variable_is_still_followed(cash_magics):
    cells = [
        f"import time, numpy as np\ndef draw(g):\n    {SLEEP}\n    return float(g.normal())",
        "rng = np.random.default_rng(0)",
        "x = draw(rng)",
    ]
    _run_all(cash_magics, cells)
    _restart(cash_magics)
    assert "RESTORED" in _run_all(cash_magics, cells)[2], "the draw is stored"


def test_a_closure_holding_a_generator_changes_what_it_holds():
    def make(held):
        def draw():
            return held.normal() if hasattr(held, "normal") else next(held)

        return draw

    assert changes_its_closure(make(np.random.default_rng(0)))
    assert changes_its_closure(make(iter(range(3))))
    assert not changes_its_closure(make((1, 2)))


def test_a_generator_or_iterator_held_as_a_default_changes_what_it_holds():
    """``def h(it=it)`` holds it in its defaults, not its closure: the gate
    read the closure only, and served ``y = h()`` the item ``x = h()`` got."""
    it = iter(range(3))
    rng = np.random.default_rng(0)

    def by_default(it=it):
        return next(it)

    def by_keyword(*, g=rng):
        return g.normal()

    def in_a_dict_default(st={"g": rng}):
        return st["g"].normal()

    assert holds_a_stream(by_default) and changes_its_closure(by_default)
    assert holds_a_stream(by_keyword) and changes_its_closure(by_keyword)
    assert holds_a_stream(in_a_dict_default)

    def plain(k=3, *, scale=1.0):
        return k * scale

    assert not holds_a_stream(plain) and not changes_its_closure(plain)


@dataclasses.dataclass
class _Point:
    x: int
    y: int


class _Steps:
    """An iterator class: told by its methods, its ``__iter__`` never asked."""

    asked = 0

    def __iter__(self):
        type(self).asked += 1
        return self

    def __next__(self):
        return 1


def _closing_over(held):
    def score(k):
        return len(held) * k

    return score


def test_plain_data_of_any_size_holds_no_stream():
    """Objects of one class are one class looked into: 30,000 points, or
    records holding a list, were past the walk's bound and never stored."""
    points = [_Point(i, i * i) for i in range(30_000)]
    records = [{"id": i, "tags": [i, i + 1]} for i in range(30_000)]
    rows = [[i, [i]] for i in range(30_000)]
    for held in (points, records, rows, {"pts": points}):
        assert not holds_a_stream(_closing_over(held))
    assert holds_a_stream(_closing_over([*points, iter(range(3))]))
    assert holds_a_stream(_closing_over([*records, {"g": np.random.default_rng(0)}]))


def test_an_iterator_is_told_by_its_class():
    _Steps.asked = 0
    assert holds_a_stream(_closing_over([_Steps()]))
    assert _Steps.asked == 0, "the walk ran the object's __iter__"


def test_the_module_generators_are_not_its_own():
    import random

    assert not holds_a_stream(_closing_over([random.__dict__["_inst"], np.random.mtrand._rand]))
    assert holds_a_stream(_closing_over([random.Random(0)]))


def test_a_lookup_table_of_millions_of_plain_items_holds_no_stream():
    # Counted toward the walk's bound, a table past 2,000,000 numbers or
    # strings was taken as holding a stream: never served.
    for held in (list(range(2_100_000)), {i: i for i in range(2_100_000)}, [str(i) for i in range(2_100_000)]):
        assert not holds_a_stream(_closing_over(held))


def test_a_class_given_next_later_is_a_stream_from_then_on():
    class Steps:
        def __iter__(self):
            return self

    fn = _closing_over([Steps()])
    assert not holds_a_stream(fn)
    Steps.__next__ = lambda self: 1
    assert holds_a_stream(fn)


@pytest.mark.parametrize(
    "change",
    [
        "table['it'] = iter(range(3))",
        "rows[7] = iter(range(3))",
        "rows[3]['tags'].append(iter(range(3)))",
        "rows.append(np.random.default_rng(0))",
    ],
)
def test_a_stream_put_in_a_table_read_before_is_found(cash_magics, change):
    """What a held table was found to hold is kept while the watchers prove
    the table unchanged: a stream put in it since is found."""
    from cash import value_watch

    run_cash_cell(
        cash_magics,
        "import numpy as np\n"
        "rows = [{'id': i, 'tags': ['a']} for i in range(30_000)]\n"
        "table = {i: [i] for i in range(30_000)}\n"
        "def look(k, rows=rows, table=table):\n    return rows[k]['id'] + table[k][0]",
    )
    ns = cash_magics.shell.user_ns
    for _ in range(3):  # watched from the second time asked
        assert not holds_a_stream(ns["look"])
    if value_watch.WATCHING:
        for name in ("table", "rows"):
            assert value_watch.peek(ns[name], value_watch.KIND_NO_STREAM) is not None, f"control: kept for {name}"
    exec(change, ns)
    assert holds_a_stream(ns["look"])


def test_a_stream_in_a_librarys_dict_is_found():
    import collections

    assert holds_a_stream(_closing_over(collections.OrderedDict(it=iter(range(3)))))
    assert holds_a_stream(_closing_over(collections.defaultdict(list, it=iter(range(3)))))
    assert not holds_a_stream(_closing_over(collections.OrderedDict(a=[1, 2])))
