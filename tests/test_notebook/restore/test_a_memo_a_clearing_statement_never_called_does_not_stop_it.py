"""A statement whose code names ``cache_clear`` is not held to a memo it
never called.

The TSP notebook's last check runs ``held_karp_tsp``, which clears
``shortest_segment`` for each problem. Code that names ``cache_clear`` can
reset a memo's counters, so every memo it could reach was recorded as
read, ``benchmark`` too, whose answers hold run times. After a restart a
benchmark cell above ran again (its answers came out with other times),
and the check ran again too: "the helper memo benchmark does not hold the
answers the statement read", though it never calls ``benchmark``. A memo
whose counters and very key and answer objects are as they were, and that
no clear the run was seen making reached, was not called: it decides
nothing. One the statement does call still decides its hit.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = ABOVE_PERSISTENCE_FLOOR_S
CELLS = [
    "import functools, time",
    "@functools.cache\ndef bench(k):\n    return [time.perf_counter(), k]",
    "@functools.cache\ndef seg(i):\n    return i * i",
    "def solve(n):\n    seg.cache_clear()\n    return sum(seg(i) for i in range(n))",
    "b = bench(1)",
    f"r = (solve(6), time.sleep({SLEEP}))[0]",
]


def _last(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _run_all(magics, cells: list[str]) -> list[str]:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    return _last(magics)


def test_a_memo_the_statement_never_called_does_not_stop_its_hit(cash_magics):
    assert _run_all(cash_magics, CELLS) == ["COMPUTED"]
    # The benchmark computed again, as after a restart: another answer.
    run_cash_cell(cash_magics, "bench.cache_clear()", cells=CELLS)
    assert _run_all(cash_magics, CELLS) == ["RESTORED"]
    assert cash_magics.shell.user_ns["r"] == sum(i * i for i in range(6))


def test_a_memo_the_statement_calls_still_decides_its_hit(cash_magics):
    """Control: ``uses`` reads ``bench``, whose answer changed, so it runs."""
    cells = [
        *CELLS[:4],
        CELLS[4],
        "def uses(n):\n    seg.cache_clear()\n    return bench(1)[1] + n",
        f"r = (uses(6), time.sleep({SLEEP}))[0]",
    ]
    assert _run_all(cash_magics, cells) == ["COMPUTED"]
    run_cash_cell(cash_magics, "bench.cache_clear()", cells=cells)
    assert _run_all(cash_magics, cells) == ["COMPUTED"]
