"""A statement whose helper reads a file through ``os.open`` reruns when the
file changes, as one reading it through ``open`` does.

The ``open`` audit event that ``os.open`` raises carries flags in place of a
mode, and the tracker took it for a write only: the statement was stored
with no input, and after the file changed a rerun served the old bytes with
no warning.
"""

from __future__ import annotations

import os

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _cells(data, log):
    return [
        "import os, time\n"
        "def read_raw(p):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    fd = os.open(p, os.O_RDONLY)\n"
        "    data = os.read(fd, 100)\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return data",
        f"v = read_raw({str(data)!r})",
    ]


def _run(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    return cash_magics.shell.user_ns["v"]


def _settle(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def test_an_edit_to_the_file_is_read_again(cash_magics, tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(b"one")
    _settle(data)
    log = tmp_path / "runs.log"
    cells = _cells(data, log)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"

    assert _run(cash_magics, cells) == b"one"
    assert _run(cash_magics, cells) == b"one"
    assert log.read_text(encoding="utf-8") == "x"  # positive control: an unchanged file hits

    data.write_bytes(b"twotwo")
    _settle(data)
    assert _run(cash_magics, cells) == b"twotwo"
    assert log.read_text(encoding="utf-8") == "xx"
