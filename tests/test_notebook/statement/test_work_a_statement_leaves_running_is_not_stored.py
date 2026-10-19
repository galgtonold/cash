"""Work a statement leaves running in a thread is not stored over.

``start(path)`` hands ``work`` to a thread (``threading.Thread``, a ``Timer``,
``executor.submit``) and returns while it runs; the thread later fills
``RESULT['v']`` from a file. Only NOTEBOOK-THREAD-ALIVE warned: the cell
below, ``v = RESULT['v'] + ''``, was stored on ``RESULT``'s lineage, which
the thread's write never moved, so after the file was edited and both cells
ran again it served the old text where plain Python reads the new. The
decorator refuses such a call (STORE-WORK-RUNNING). Now the statement that
left the work running is not stored, nor one reading a variable that work
can reach, until a statement binds it again.
"""

from __future__ import annotations

import time

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

S = ABOVE_PERSISTENCE_FLOOR_S
T = ABOVE_PERSISTENCE_FLOOR_S + 0.3

STARTERS = {
    "a thread": (
        "import threading, time\nRESULT = {}\n"
        "def start(path):\n"
        f"    def work():\n        time.sleep({T})\n        RESULT['v'] = open(path, encoding='utf-8').read()\n"
        f"    threading.Thread(target=work).start()\n    time.sleep({S})\n"
    ),
    "a timer": (
        "import threading, time\nRESULT = {}\n"
        "def start(path):\n"
        "    def work():\n        RESULT['v'] = open(path, encoding='utf-8').read()\n"
        f"    threading.Timer({T}, work).start()\n    time.sleep({S})\n"
    ),
    "a thread pool": (
        "import concurrent.futures as cf, time\nRESULT = {}\nPOOL = cf.ThreadPoolExecutor(2)\n"
        "def start(path):\n"
        f"    def work():\n        time.sleep({T})\n        RESULT['v'] = open(path, encoding='utf-8').read()\n"
        f"    POOL.submit(work)\n    time.sleep({S})\n"
    ),
}


@pytest.mark.parametrize("starter", sorted(STARTERS))
def test_the_cell_reading_what_the_work_wrote_sees_the_new_file(cash_magics, mock_shell, tmp_path, starter):
    data = tmp_path / "d.txt"
    data.write_text("old", encoding="utf-8")
    cells = [STARTERS[starter], f"start({str(data)!r})", "v = RESULT['v'] + ''"]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    time.sleep(T + 0.4)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert mock_shell.user_ns["v"] == "old"

    data.write_text("new", encoding="utf-8")
    run_cash_cell(cash_magics, cells[1], cells=cells)
    time.sleep(T + 0.4)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert mock_shell.user_ns["RESULT"]["v"] == "new", "control: the thread read the edited file"
    assert mock_shell.user_ns["v"] == "new", "served what the thread wrote the first time"


def test_a_statement_reading_a_rebound_name_is_stored_again(cash_magics, mock_shell, tmp_path):
    data = tmp_path / "d.txt"
    data.write_text("old", encoding="utf-8")
    cells = [STARTERS["a thread"], f"start({str(data)!r})"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    time.sleep(T + 0.4)
    reach = cash_magics._cell_executor._statement_processor._thread_reach
    assert "RESULT" in reach.names
    run_cash_cell(cash_magics, "RESULT = {'v': 'set'}")
    assert "RESULT" not in reach.names
