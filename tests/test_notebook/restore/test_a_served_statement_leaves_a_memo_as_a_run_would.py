"""A served statement leaves a helper's memo as running it would have.

``@functools.cache`` on a helper that reseeds the global ``random`` (a
reproducible set of test cases), a slow statement that calls it, and a cheap
statement below that calls it again and then draws. In a run the slow
statement fills the memo, so the call below hits it, never reseeds, and the
draw continues the stream where the slow statement left it. After a restart
the slow statement was served: the memo stayed empty, the call below ran the
helper, reseeded, and the draw came out different from the one a run makes.

A memo that is a module dict the helper fills is a global the statement
writes, so the statement is not served over it; that case is the control.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, random, time"
HELPER_CACHE = (
    "@functools.cache\ndef cases(n):\n    random.seed(n)\n    return tuple(random.random() for _ in range(n))"
)
HELPER_DICT = (
    "_cases = {}\n"
    "def cases(n):\n"
    "    if n not in _cases:\n"
    "        random.seed(n)\n"
    "        _cases[n] = tuple(random.random() for _ in range(n))\n"
    "    return _cases[n]"
)
SLOW = f"bench = (sum(cases(3)) + random.random(), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
BELOW = "r = (cases(3), random.random())[1]"


def _restart(magics, names) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]]


def _check(magics, helper) -> None:
    cells = [IMPORTS, "random.seed(0)", helper, SLOW, BELOW]
    plain: dict = {}
    for code in cells:
        exec(code, plain)
    for code in cells:
        run_cash_cell(magics, code)
    assert magics.shell.user_ns["r"] == plain["r"], "control: the first run"
    _restart(magics, ["functools", "random", "time", "cases", "_cases", "bench", "r"])
    statuses = []
    for code in cells:
        run_cash_cell(magics, code)
        statuses.append(_statuses(magics))
    assert magics.shell.user_ns["bench"] == plain["bench"]
    assert magics.shell.user_ns["r"] == plain["r"], f"statuses after the restart: {statuses}"


@pytest.mark.xfail(strict=True, reason="a served statement leaves a functools memo empty")
def test_a_functools_cache_filled_by_a_served_statement(cash_magics):
    _check(cash_magics, HELPER_CACHE)


def test_a_module_dict_filled_by_a_served_statement(cash_magics):
    _check(cash_magics, HELPER_DICT)
