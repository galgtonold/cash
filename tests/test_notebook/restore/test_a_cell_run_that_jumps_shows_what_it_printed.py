"""A cell that rebuilds one name in steps shows its printed output when a
restore lets its run skip them.

``y = load(); y = clean(y)``: on the next run, and after a restart, cash
restores the last version (or finds both current) and skips the steps.
Neither step's output was shown, though a one-statement hit replays its
print.
"""

from __future__ import annotations

import pytest

from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
from tests._cell_driver import run_cash_cell
from tests.conftest import MockShell

HELPERS = """\
import time
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
def load():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    print("loaded 100 rows")
    return list(range(100))
def clean(rows):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    out = [r for r in rows if r > 1]
    print(f"cleaned to {len(out)}")
    return out
def quiet(rows):
    return rows
"""

CELLS = ["import cb5_helpers", "y = cb5_helpers.load()\ny = cb5_helpers.clean(y)"]
SHOWN = "loaded 100 rows\ncleaned to 98\n"


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    (tmp_path / "cb5_helpers.py").write_text(HELPERS)
    monkeypatch.syspath_prepend(str(tmp_path))


@pytest.fixture
def kernel(tmp_path):
    def start() -> CashMagics:
        magics = CashMagics(MockShell(), Cash(cache_dir=str(tmp_path / "cache"), register_magic=False))
        magics.cash_on("")
        magics.badges.mode = "off"
        return magics

    return start


def _run_all(magics, capsys, cells=CELLS) -> str:
    capsys.readouterr()
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    return capsys.readouterr().out


def test_a_second_run_shows_the_steps_output(helpers, kernel, capsys):
    magics = kernel()
    assert SHOWN in _run_all(magics, capsys)
    assert SHOWN in _run_all(magics, capsys)


def test_a_run_after_a_restart_shows_the_steps_output(helpers, kernel, capsys):
    first = kernel()
    assert SHOWN in _run_all(first, capsys)
    for queue in __import__("cash.backends._writes", fromlist=["_"]).all_pending_writes():
        queue.wait_all()
    second = kernel()
    out = _run_all(second, capsys)
    assert SHOWN in out
    assert second.shell.user_ns["y"] == list(range(2, 100))


def test_a_step_whose_output_is_too_long_to_keep_runs(helpers, kernel, capsys, tmp_path):
    # Over the cap, the entry keeps no copy of the output: the step runs
    # after the restart rather than being skipped silently.
    (tmp_path / "cb5_long.py").write_text(
        "import time\nfrom tests.conftest import ABOVE_PERSISTENCE_FLOOR_S\n"
        "def load():\n    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)\n    print('x' * 70000)\n    return [1, 2]\n"
    )
    cells = ["import cb5_long, cb5_helpers", "y = cb5_long.load()\ny = cb5_helpers.clean(y)"]
    first = kernel()
    assert "x" * 70000 in _run_all(first, capsys, cells)
    for queue in __import__("cash.backends._writes", fromlist=["_"]).all_pending_writes():
        queue.wait_all()
    out = _run_all(kernel(), capsys, cells)
    assert "x" * 70000 in out
    assert "cleaned to 1" in out
