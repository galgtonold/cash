"""A cell whose loop rebinds a name a loop above built does not rebuild
what that loop built when it runs again.

The statement that last bound ``actions`` is the running cell's own loop,
which is in no cell above it: cash took it for an unsaved edit above, stopped
trusting what loops built, and ran the loop building ``data`` again.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from tests._cell_driver import run_cash_cell

HELPER = """
import os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows.log")
def note():
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"row\\n")
    os.close(fd)
"""
CELLS = [
    "import h\nraw = ['u1,a:1,a:2,a:3', 'u2,b:1,c:2,c:3']",
    "data = []\nactions = []\nfor row in raw:\n    h.note()\n    parts = row.split(',')\n    acts = []\n"
    "    for p in parts[1:]:\n        a, t = p.split(':')\n        acts.append((a, int(t)))\n"
    "        actions.append(a)\n    data.append((parts[0], acts))",
    "data.sort(key=lambda x: x[-1][0][-1])",
    "k = 0\nfor uid, actions in data:\n    for i in range(len(actions) - 2):\n"
    "        if actions[i][0] == actions[i + 2][0]:\n            k += 1\n            print({word!r}, uid)\n"
    "            break\nk",
]


@pytest.mark.parametrize("word", ["first", "second"], ids=["unchanged", "edited"])
def test_the_loop_above_does_not_run_again(cash_magics, tmp_path, monkeypatch, word):
    (tmp_path / "h.py").write_text(HELPER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "h", raising=False)
    cells = [*CELLS[:3], CELLS[3].format(word="first")]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        cells[3] = CELLS[3].format(word=word)
        run_cash_cell(cash_magics, cells[3], cells=cells)

    assert (tmp_path / "rows.log").read_text(encoding="utf-8").split() == ["row", "row"]
    assert cash_magics.shell.user_ns["k"] == 1
