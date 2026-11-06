"""A statement that reads a memo answer a statement cash did not watch put
there is stored, and served on the next Run All.

The TSP notebook's ``run(ensemble_tsp_2opt, USA)`` draws with pyplot, so it
runs every time and cash does not watch the memos it fills: it left
``subsegment_indexes(len(USA))``, a list, in the memo. The benchmark cell
below found that answer and added its own key, and ran on every pass:
"It filled the helper memo subsegment_indexes: served, the memo would stay
as it was". The answer was not one cash had seen, so whether the run
changed it in place could not be told. An answer cash has not seen is now
taken as it is just before the statement runs: a run that only reads it
is stored, keyed on it; one that changes it in place still runs every time.
"""

from __future__ import annotations

from cash.notebook.ipython.session import CashSession
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLEEP = ABOVE_PERSISTENCE_FLOOR_S
SEG = "import functools, time\n\n@functools.cache\ndef seg(n):\n    return [(i, i + 1) for i in range(n)]"
UNWATCHED = "# @cash:no-cache\nfirst = len(seg(4))"


def _last(magics) -> list[str]:
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


def _passes(magics, cells: list[str], times: int = 2) -> list[list[str]]:
    seen = []
    for _ in range(times):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        seen.append(_last(magics))
    return seen


def test_a_statement_reading_an_unwatched_answer_is_served_on_the_next_run_all(cash_magics):
    cells = [SEG, UNWATCHED, f"r = (len(seg(4)) + len(seg(5)), time.sleep({SLEEP}))[0]"]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["RESTORED"]], seen
    ns = cash_magics.shell.user_ns
    assert ns["r"] == 9
    assert ns["seg"].cache_info().currsize == 2


def test_a_statement_changing_an_unwatched_answer_in_place_still_runs(cash_magics, monkeypatch):
    """Guard: the run appends to the answer it found, so a later call of
    ``seg(4)`` gets the longer list; served, the memo would keep it as the
    unwatched cell left it."""
    reasons: list = []
    record = CashSession.record_cell

    def record_cell(self, metrics, *args, **kwargs):
        reasons.extend(m.get("uncacheable_reasons") for m in metrics)
        return record(self, metrics, *args, **kwargs)

    monkeypatch.setattr(CashSession, "record_cell", record_cell)
    grow = "def grow(n):\n    seg(n).append((9, 9))\n    return n"
    cells = [SEG, UNWATCHED, grow, f"r = (grow(4), len(seg(5)), len(seg(4)), time.sleep({SLEEP}))[2]"]
    seen = _passes(cash_magics, cells)
    assert seen == [["COMPUTED"], ["COMPUTED"]], seen
    assert any("changed an answer the helper memo seg holds in place" in str(r) for r in reasons[-1]), reasons[-1]


def test_a_statement_reading_an_unwatched_answer_runs_once_that_answer_changed(cash_magics):
    """Guard: the entry is keyed on the answer as the run read it."""
    cells = [SEG, UNWATCHED, f"r = (sum(len(p) for p in seg(4)) + len(seg(5)), time.sleep({SLEEP}))[0]"]
    assert _passes(cash_magics, cells) == [["COMPUTED"], ["RESTORED"]]
    edited = [SEG, "# @cash:no-cache\nfirst = len(seg(4))\nseg(4).append((7,))", cells[2]]
    for cell in edited:
        run_cash_cell(cash_magics, cell, cells=edited)
    assert _last(cash_magics) == ["COMPUTED"]
    assert cash_magics.shell.user_ns["r"] == 2 * 4 + 1 + 5
