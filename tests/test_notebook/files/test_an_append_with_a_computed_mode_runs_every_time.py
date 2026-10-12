"""A helper of yours appending through ``open(path, mode)`` with the mode
passed in runs every time, as a plain kernel does.

The static check skips an ``open()`` whose mode is computed (it reads as
often as it writes), and the runtime observer left every append to the
static check (a library's append is a log handler's as often as not), so the
statement was stored: a cleared output folder stayed without the log, and a
log plain Python grows on every run stopped growing. An append the user's own
``open()`` makes now counts as the write it is; a library's still does not.
"""

from __future__ import annotations

import logging

from cash.notebook.write_observer import kept_writes, observe_writes
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _write_log(rows, path, mode="a"):
    with open(path, mode, encoding="utf-8") as f:
        f.writelines(r + "\n" for r in rows)


def test_the_observer_keeps_an_append_of_the_users_code(tmp_path):
    log = tmp_path / "log.txt"
    with observe_writes() as written:
        _write_log(["a"], log)
    assert kept_writes(written, None, written.appended) == [str(log)]


def test_the_observer_leaves_a_log_handlers_append(tmp_path):
    log = tmp_path / "handler.log"
    with observe_writes() as written:
        handler = logging.FileHandler(log, encoding="utf-8")
        handler.close()
    assert kept_writes(written, None, written.appended) == []


def test_the_log_grows_on_every_run(cash_magics, tmp_path):
    log = tmp_path / "run.log"
    cells = [
        "import time\n"
        "def write_log(rows, path, mode='a'):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    with open(path, mode) as f:\n"
        "        f.writelines(r + '\\n' for r in rows)\n"
        "    return len(rows)",
        f"n = write_log(['a', 'b'], {str(log)!r})",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert log.read_text(encoding="utf-8").split() == ["a", "b"] * 3
