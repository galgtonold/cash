"""A function or class whose calls change what it holds is not handed back
by reference.

A hit keeps a function or a class by reference: it hands back the one the
last run made. Run again, a ``def`` or ``class`` makes it afresh. When its
calls changed it since (a memo filled, a generator in its closure drawn
from), the hit handed back that state: after a restart, ``class Cases``
with a cached ``make`` that seeds ``random`` came back with ``make(3)``
answered, so the seed was skipped and ``r`` below differed from plain
Python; ``h, peek = make_helper()`` came back with its generator moved on,
so ``x = h()`` gave the second number. Only on a busy machine, where these
statements took long enough to be stored: the floors are 0 here.
"""

from __future__ import annotations

import sys

import pytest

from cash.notebook.call_unit import CallUnit
from tests._cell_driver import run_cash_cell

HELPERS = (
    "import functools, random\n"
    "import numpy as np\n"
    "def make_helper():\n"
    "    g = np.random.default_rng(0)\n"
    "    def helper():\n        return float(g.normal())\n"
    "    def peek():\n        return float(g.normal())\n"
    "    return helper, peek\n"
    "def make_cases():\n"
    "    class Cases:\n        @staticmethod\n        @functools.cache\n        def make(n):\n"
    "            random.seed(n)\n            return tuple(random.random() for _ in range(n))\n"
    "    return Cases\n"
    "def make_plain():\n"
    "    class Plain:\n        def twice(self, n):\n            return 2 * n\n"
    "    return Plain\n"
)

# Made by a module's factory: a ``class`` or ``def`` in a cell names
# ``__main__`` in a kernel and is never stored by value.
CASES = {
    "a class whose memo is filled": (
        [
            "import vbr_helpers, random\nCases = vbr_helpers.make_cases()",
            "random.seed(0)",
            "bench = sum(Cases.make(3)) + random.random()",
            "r = (Cases.make(3), random.random())[1]",
        ],
        ("r", "bench"),
    ),
    "a closure holding a generator": (
        ["import vbr_helpers\nh, peek = vbr_helpers.make_helper()", "x = h()", "y = peek()"],
        ("x", "y"),
    ),
    "a closure holding an iterator": (
        [
            "def make():\n    it = iter(range(100))\n    def h():\n        return next(it)\n    return h",
            "h = make()",
            "x = h()",
            "y = h()",
        ],
        ("x", "y"),
    ),
}


@pytest.fixture
def floors_at_zero(cash_instance, monkeypatch, tmp_path):
    """Every statement and call stored, as on a busy machine."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    monkeypatch.setattr(CallUnit, "_cost_floor_s", lambda self: 0.0)
    (tmp_path / "vbr_helpers.py").write_text(HELPERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("vbr_helpers", None)


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("random", "vbr_helpers", "Cases", "make", "bench", "r", "h", "peek", "x", "y", "Plain"):
        magics.shell.user_ns.pop(name, None)
    sys.modules.pop("vbr_helpers", None)


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]]


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_run_and_a_restart_match_plain(cash_magics, floors_at_zero, case):
    cells, names = CASES[case]
    plain: dict = {}
    for code in cells:
        exec(code, plain)
    sys.modules.pop("vbr_helpers", None)
    want = {n: plain[n] for n in names}
    for run in ("first", "second", "third", "restart", "after the restart"):
        if run == "restart":
            _restart(cash_magics)
        statuses = []
        for code in cells:
            run_cash_cell(cash_magics, code)
            statuses.append(_statuses(cash_magics))
        got = {n: cash_magics.shell.user_ns[n] for n in names}
        assert got == want, f"{run} run, statuses: {statuses}"


def test_a_class_without_state_is_still_served(cash_magics, floors_at_zero):
    """Ordinary notebooks keep their hits: a class whose calls change
    nothing is served. A call that makes a ``functools`` memo is never
    stored, even an empty one, so ``Cases`` runs each time."""
    cell = "import vbr_helpers\nPlain = vbr_helpers.make_plain()\nCases = vbr_helpers.make_cases()"
    run_cash_cell(cash_magics, cell)
    run_cash_cell(cash_magics, cell)
    assert _statuses(cash_magics)[1:] == ["MIXED", "COMPUTED"]
