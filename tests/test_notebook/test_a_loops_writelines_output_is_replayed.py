"""Output a loop writes with ``writelines`` is replayed when the loop restores.

A loop run as one unit streams its output through the statement pipeline's
tee, which records what it passes on so a later hit can show it again. That
tee had no ``writelines``: ``__getattr__`` sent it straight to the real stream,
so the first run showed the lines and the restored run showed nothing.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

CELL = """import sys
total = 0
for i in range(3000):
    total += i
    if i % 1000 == 0:
        sys.stdout.writelines([f"w{i}\\n"])
"""


def test_the_restored_loop_shows_its_lines(cash_magics, capsys):
    cash_magics._cash_instance.reconfigure(persist_all=True)  # cache it however cheap it is
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no notebook around the cell
        run_cash_cell(cash_magics, CELL)
        assert "w0\nw1000\nw2000\n" in capsys.readouterr().out
        run_cash_cell(cash_magics, CELL)
    assert cash_magics._last_cell_metrics["statements"][-1]["status"].name == "RESTORED"
    assert "w0\nw1000\nw2000\n" in capsys.readouterr().out
