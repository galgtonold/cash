"""A statement on disk finds the call result it refers to there after a restart.

``chg_tab = arm_table(chg[...])`` stores a reference to the call's entry, not
the pivot table again, and writing the statement's entry to disk writes the
call's entry with it. A pivot table's RAM copy shares parts with the user's
frame (its labels), and an entry whose RAM copy may share was never written
from memory: it could hold what the user changed since. So the statement's
entry reached the disk pointing at an entry that never did, and after a
restart it missed and ran the call again. Written as the statement is stored,
the result is what the call returned: the statement has just checked it.
"""

from __future__ import annotations

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("pandas")


@pytest.fixture
def tiers(cash_magics, cash_instance, tmp_path):
    """The notebook over a RAM + disk backend, as a kernel's default one."""
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    yield cash_magics, ram, disk
    disk._writes.wait_all()
    disk.shutdown()


def _cells(log) -> list[str]:
    # The call is quicker than the floor for a disk entry of its own; the
    # statement, which also picks the rows, is not. A table with a
    # two-level index, as ``pivot_table`` makes, is one the RAM tier may
    # share with the user's.
    return [
        "import os, time\nimport pandas as pd\n"
        "raw = pd.DataFrame({'k': ['a', 'b'] * 500, 'v': ['V1', 'V1', 'V4', 'V4'] * 250,\n"
        "                    'arm': ['P', 'L'] * 500, 'x': [float(i) for i in range(1000)]})",
        "def table(d):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'run\\n')\n"
        "    os.close(fd)\n"
        "    time.sleep(0.02)\n"
        "    return d.pivot_table(index=['k', 'v'], columns='arm', values='x', aggfunc='mean')",
        f"tab = table((time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), raw)[1])",
    ]


def _runs(log) -> int:
    return log.read_text(encoding="utf-8").count("run") if log.exists() else 0


def test_the_statement_is_served_after_a_restart(tiers, tmp_path):
    magics, ram, disk = tiers
    log = tmp_path / "runs.log"
    cells = _cells(log)
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    assert _runs(log) == 1
    expected = magics.shell.user_ns["tab"].copy()
    # Positive control: the result is one the RAM tier may share.
    assert any(ram.may_share_with_caller(key) for key in list(ram._store) if key.startswith("call:"))
    disk._writes.wait_all()
    # A restart: memory, and the names, are gone; the disk stays.
    ram.clear()
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in ("os", "time", "pd", "raw", "table", "tab"):
        magics.shell.user_ns.pop(name, None)
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    assert _runs(log) == 1, "the call ran again after the restart"
    assert magics.shell.user_ns["tab"].equals(expected)
