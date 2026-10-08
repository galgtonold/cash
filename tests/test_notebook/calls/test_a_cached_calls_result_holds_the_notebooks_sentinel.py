"""A cached call's result holds the notebook's own sentinel, not a copy of it.

``MISSING = object()`` in one cell and ``res = lookup(keys)`` holding it in a
list, with the call to ``lookup`` cached on its own: a hit's result is a
copy, and the ``MISSING`` in it was another object, so ``v is MISSING`` was
False on every hit. A Run All also defines a new ``MISSING``, which plain
Jupyter's re-run of the call would hold.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _cells(counter):
    """The notebook; each run of ``lookup``'s body appends a byte to *counter*."""
    return [
        "import os, time\nMISSING = object()\nTABLE = {'a': 1}",
        "def lookup(keys):\n"
        f"    fd = os.open({str(counter)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'x'); os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return [TABLE.get(k, MISSING) for k in keys]",
        "res = lookup(('a', 'b'))\npair = (lookup(('b',)), {'k': [MISSING]})",
    ]


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.parametrize("runs", [2, 3])
def test_a_restored_list_holds_the_sentinel_defined_now(cash_magics, mock_shell, tmp_path, runs):
    counter = tmp_path / "runs"
    cells = _cells(counter)
    for _ in range(runs):
        _run_all(cash_magics, cells)
        ns = mock_shell.user_ns
        assert [v is ns["MISSING"] for v in ns["res"]] == [False, True]
        assert ns["pair"][0][0] is ns["MISSING"]
        assert ns["pair"][1]["k"][0] is ns["MISSING"]
    assert counter.read_bytes() == b"xx", "the later runs did not come from the cache"
