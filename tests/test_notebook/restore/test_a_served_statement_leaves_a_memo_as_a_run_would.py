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


def test_a_functools_cache_filled_by_a_served_statement(cash_magics):
    _check(cash_magics, HELPER_CACHE)


def test_a_module_dict_filled_by_a_served_statement(cash_magics):
    _check(cash_magics, HELPER_DICT)


def test_a_memo_that_holds_the_answers_still_serves(cash_magics):
    # In the session that filled it, the memo still holds what the run left:
    # nothing differs, so a statement that only reads it is served.
    look = f"b = (cases(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    cells = [IMPORTS, HELPER_CACHE, SLOW, look]
    for code in cells:
        run_cash_cell(cash_magics, code)
    run_cash_cell(cash_magics, look)
    assert _statuses(cash_magics) == ["RESTORED"]


def test_a_statement_stored_when_its_memo_missed_is_not_served_where_it_hits(cash_magics):
    # Stored when `cases(3)` ran its body and reseeded; run again with the
    # memo filled, the body does not run and the draw reads the stream as
    # the seed above left it.
    cells = [IMPORTS, "random.seed(0)", HELPER_CACHE, SLOW, "random.seed(0)", SLOW]
    plain: dict = {}
    for code in cells:
        exec(code, plain)
    for code in cells:
        run_cash_cell(cash_magics, code)
    assert cash_magics.shell.user_ns["bench"] == plain["bench"], _statuses(cash_magics)


def test_a_memo_as_full_but_with_other_answers_does_not_serve(cash_magics):
    square = "@functools.cache\ndef square(n):\n    return n * n"
    fill = f"x = (square(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    for code in [IMPORTS, square, fill]:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "square", "x"])
    for code in [IMPORTS, square, "other = square(4)", fill]:
        run_cash_cell(cash_magics, code)
    assert _statuses(cash_magics) == ["COMPUTED"], "one answer, for another key"


def test_a_statement_that_only_read_the_memo_reruns_until_it_is_filled_again(cash_magics):
    # `look` found the answer `fill` left; after a restart the memo is empty
    # until `fill` runs again, and `look` served before that would leave it
    # empty where a run fills it.
    fill = f"a = (cases(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    look = f"b = (cases(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    for code in [IMPORTS, HELPER_CACHE, fill, look]:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "random", "time", "cases", "a", "b"])
    for code in [IMPORTS, HELPER_CACHE]:
        run_cash_cell(cash_magics, code)
    run_cash_cell(cash_magics, look)
    assert _statuses(cash_magics) == ["COMPUTED"]
    run_cash_cell(cash_magics, look)
    assert _statuses(cash_magics) == ["RESTORED"], "filled again by its own run"
