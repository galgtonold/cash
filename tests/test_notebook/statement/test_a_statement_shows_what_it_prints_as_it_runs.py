"""What a statement prints reaches the notebook as it runs, as in plain.

A statement's prints were held while it ran and shown when it ended. A long
statement printing progress (``df[df.apply(check, axis=1)]`` where ``check``
prints each row) showed nothing for as long as it ran, and a run stopped at
a time limit had shown nothing at all, where plain had shown every row so
far (JuNE expert_5). The prints now go on as they are written; the displays
are still held, and a hit shows both again.
"""

from __future__ import annotations

import io
import sys

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = (
    "import sys, time\n"
    "from IPython.display import display, HTML\n"
    "seen = []\n"
    "def check(i):\n"
    "    print('row', i)\n"
    "    seen.append(sys.__cash_screen__.getvalue())\n"
    "    display(HTML('<b>shown</b>'))\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S} / 2)\n"
    "    return i\n"
    "def quiet_check(i):\n"
    "    print('row', i)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S} / 2)\n"
    "    return i"
)
CELL = "kept = [check(i) for i in range(2)]"


def _status(magics, code: str):
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def test_a_print_reaches_the_notebook_while_the_statement_runs(cash_magics, monkeypatch):
    screen = io.StringIO()
    monkeypatch.setattr(sys, "stdout", screen)
    monkeypatch.setattr(sys, "__cash_screen__", screen, raising=False)
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, CELL)

    seen = cash_magics.shell.user_ns["seen"]
    assert "row 0\n" in seen[0], seen
    assert "row 1\n" in seen[1], seen
    assert screen.getvalue().count("row 0\n") == 1, "shown once, not again when it ended"


def test_a_hit_shows_the_prints_again(cash_magics, monkeypatch):
    screen = io.StringIO()
    monkeypatch.setattr(sys, "stdout", screen)
    monkeypatch.setattr(sys, "__cash_screen__", screen, raising=False)
    cell = "kept = [quiet_check(i) for i in range(2)]"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, cell)
    first = screen.getvalue()

    run_cash_cell(cash_magics, cell)

    assert _status(cash_magics, cell) == CacheStatus.RESTORED
    assert screen.getvalue()[len(first) :].count("row") == 2
