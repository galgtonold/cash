"""A helper of yours appending through ``os.open`` + ``os.write`` runs every
time, as ``open(path, 'a')`` in the same helper does and as a plain kernel does.

The runtime observer took every ``O_APPEND`` open for a library's append and
left it to the static check, which does not read ``os.open`` in a helper:
the statement was stored, and a hit skipped the line. Waived where it is
made (``# @cash:assume-safe`` on the ``os.open`` line), the append is one a
hit may skip, which is how the tests count runs.
"""

from __future__ import annotations

import os

from cash.notebook.write_observer import kept_writes, observe_writes
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _append_raw(path):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, b"a\n")
    finally:
        os.close(fd)


def test_the_observer_keeps_an_os_open_append_of_the_users_code(tmp_path):
    log = tmp_path / "log.txt"
    with observe_writes() as written:
        _append_raw(str(log))
    assert kept_writes(written, None, written.appended, written.waived) == [str(log)]


def _run_three_times(cash_magics, cells):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)


def test_the_log_grows_on_every_run(cash_magics, tmp_path):
    log = tmp_path / "run.log"
    cells = [
        "import os, time\n"
        f"P = {str(log)!r}\n"
        "def write_log(x):\n"
        "    fd = os.open(P, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'a\\n')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return x",
        "n = write_log(1)",
    ]
    _run_three_times(cash_magics, cells)
    assert log.read_text().split() == ["a"] * 3


def test_a_waived_append_is_written_once(cash_magics, tmp_path):
    # Another function name than above: the waiver is a comment, so the two
    # helpers' code objects would compare equal, and the observer remembers
    # its verdict per code object (`write_observer._frame_verdicts`).
    log = tmp_path / "runs.log"
    cells = [
        "import os, time\n"
        f"P = {str(log)!r}\n"
        "def count_run(x):\n"
        "    fd = os.open(P, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'a\\n')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return x",
        "n = count_run(1)",
    ]
    _run_three_times(cash_magics, cells)
    assert log.read_text().split() == ["a"]


def test_a_read_through_os_open_is_cached(cash_magics, tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("x")
    log = tmp_path / "runs.log"
    cells = [
        "import os, time\n"
        "def read_raw(p):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    fd = os.open(p, os.O_RDONLY)\n"
        "    data = os.read(fd, 10)\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return data",
        f"n = read_raw({str(data)!r})",
    ]
    _run_three_times(cash_magics, cells)
    assert log.read_text() == "x"
