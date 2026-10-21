"""No statement is served while work a statement left running still runs.

``threading.Timer(0.3, shutil.copyfile, (src, F)).start()``, then
``time.sleep(0.8)``, then ``x = open(F).read()``: the sleep was served after
its first run (1 s, then 0.01 s), so the read ran before the timer's copy and
``x`` held the file from before it, with no warning. A thread the user's code
starts on a library's function was not counted as work the statement left
running (only one running the user's own code was); now it is, and while it
runs a statement runs rather than be served.
"""

from __future__ import annotations

import time
import warnings

from tests._cell_driver import run_cash_cell

DELAY = 0.3
WAIT = 0.8

CELL = f"threading.Timer({DELAY}, shutil.copyfile, (SRC, F)).start()\ntime.sleep({WAIT})\nx = open(F, encoding='utf-8').read()"


def _statuses(cash_magics) -> list[str]:
    return [str(s.get("status")) for s in cash_magics._last_cell_metrics["statements"]]


def test_the_read_after_the_wait_sees_what_the_timer_wrote(cash_magics, mock_shell, tmp_path):
    src, f = tmp_path / "src.txt", tmp_path / "f.txt"
    f.write_text("v0", encoding="utf-8")
    run_cash_cell(cash_magics, "import shutil, threading, time")
    mock_shell.user_ns.update(SRC=str(src), F=str(f))
    for i in range(1, 4):
        src.write_text(f"v{i}", encoding="utf-8")
        started = time.perf_counter()
        run_cash_cell(cash_magics, CELL)
        assert mock_shell.user_ns["x"] == f"v{i}"
        assert time.perf_counter() - started >= WAIT
        assert "RESTORED" not in _statuses(cash_magics)[1]


def test_a_statement_is_served_again_once_the_work_is_done(cash_magics, mock_shell, tmp_path):
    src, f = tmp_path / "src.txt", tmp_path / "f.txt"
    src.write_text("v1", encoding="utf-8")
    run_cash_cell(cash_magics, "import shutil, threading, time")
    mock_shell.user_ns.update(SRC=str(src), F=str(f))
    for _ in range(2):
        run_cash_cell(cash_magics, CELL)
    run_cash_cell(cash_magics, f"time.sleep({WAIT})")
    run_cash_cell(cash_magics, f"time.sleep({WAIT})")
    assert "RESTORED" in _statuses(cash_magics)[0]


PROCESS_CELL = (
    "p = subprocess.Popen([sys.executable, '-c', "
    "'import sys, time; time.sleep(0.3); open(sys.argv[1], \"w\").write(sys.argv[2])', F, VAL])\n"
    "time.sleep(1.5)\n"
    "x = open(F, encoding='utf-8').read()"
)


def test_the_read_after_the_wait_sees_what_the_process_wrote(cash_magics, mock_shell, tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("v0", encoding="utf-8")
    run_cash_cell(cash_magics, "import subprocess, sys, time")
    mock_shell.user_ns["F"] = str(f)
    for i in range(1, 4):
        mock_shell.user_ns["VAL"] = f"v{i}"
        run_cash_cell(cash_magics, PROCESS_CELL)
        mock_shell.user_ns["p"].wait()
        assert mock_shell.user_ns["x"] == f"v{i}"
        assert "RESTORED" not in _statuses(cash_magics)[1]


def test_the_pause_is_told_once_naming_the_work(cash_magics, mock_shell, tmp_path):
    src, f = tmp_path / "src.txt", tmp_path / "f.txt"
    src.write_text("v1", encoding="utf-8")
    run_cash_cell(cash_magics, "import shutil, threading, time")
    mock_shell.user_ns.update(SRC=str(src), F=str(f))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            run_cash_cell(cash_magics, CELL)
    told = [str(w.message) for w in caught if "NOTEBOOK-WORK-RUNNING" in str(w.message)]
    assert len(told) == 1
    assert "copyfile()" in told[0] and "# @cash:assume-safe" in told[0]


def test_work_started_on_an_assume_safe_line_leaves_the_cells_served(cash_magics, mock_shell, tmp_path):
    src, f = tmp_path / "src.txt", tmp_path / "f.txt"
    src.write_text("v1", encoding="utf-8")
    run_cash_cell(cash_magics, "import shutil, threading, time")
    mock_shell.user_ns.update(SRC=str(src), F=str(f))
    cell = CELL.replace(".start()", ".start()  # @cash:assume-safe", 1)
    for _ in range(2):
        run_cash_cell(cash_magics, cell)
    assert "RESTORED" in _statuses(cash_magics)[1]
