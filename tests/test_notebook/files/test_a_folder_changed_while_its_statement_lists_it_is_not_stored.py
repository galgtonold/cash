"""A folder that gains a file while a statement lists it is not stored.

``names = names_of('inbox')``, the helper listing the folder and working on,
while another process drops a file into it: the statement was stored against
the folder with the new file, next to a listing without it, and every Run
All printed the short list until a restart.
"""

from __future__ import annotations

import threading

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def test_the_next_run_lists_the_file_that_landed(cash_magics, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(3):
        (inbox / f"f{i}.csv").write_text(str(i), encoding="utf-8")

    def drop():
        """Another process, as far as cash can tell: a thread of its own,
        started by code that is not the notebook's."""
        writer = threading.Thread(target=lambda: (inbox / "late.csv").write_text("9", encoding="utf-8"))
        writer.start()
        writer.join()

    # Kept on a library module, so the notebook's key holds none of it.
    threading.cash_test_drops = [drop]
    cells = [
        "import os, threading, time\n"
        "def names_of(d):\n"
        "    names = sorted(os.listdir(d))\n"
        "    for act in threading.cash_test_drops: act()\n"
        "    threading.cash_test_drops.clear()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return names",
        f"names = names_of({str(inbox)!r})",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    ns = cash_magics.shell.user_ns
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["names"] == ["f0.csv", "f1.csv", "f2.csv"]
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            assert ns["names"] == ["f0.csv", "f1.csv", "f2.csv", "late.csv"]
    finally:
        del threading.cash_test_drops
