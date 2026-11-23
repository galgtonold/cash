"""A cell-run plan skips a step whose value a later restore replaces.

A cell rebuilds ``sales`` in steps, and one cheap step,
``sales['big'] = is_big.astype(int)``, reads ``is_big``, which the plan
restores. A statement reading a name the plan binds anew runs
(``_readers_of_rebound``), so this one ran -- and running it needs the
``sales`` of its place in the cell, so every costly step that built that
``sales`` ran again after a restart, though ``sales['c'] = ...`` below it
is restored with the final ``sales``. What the step built is gone when the
cell ends and nothing reads it in between: it is skipped. A step whose
value something below it still reads keeps running
(``test_a_cell_plan_runs_what_reads_a_restored_name``).
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pd = pytest.importorskip("pandas")

S = ABOVE_PERSISTENCE_FLOOR_S
SETUP = "import time\nimport pandas as pd"
RAW = f"raw = (time.sleep({S}), pd.DataFrame({{'a': range(300)}}))[1]"
#: The mask selects a copy, so the run writes only into frames it made
#: (``jumpable_runs``).
CLEAN = (
    "sales = raw[raw['a'] >= 0]\n"
    f"sales = (time.sleep({S}), sales.drop_duplicates())[1]\n"
    f"sales['t'] = (time.sleep({S}), sales['a'] * 2)[1]\n"
    f"is_big = (time.sleep({S}), sales['a'] > 100)[1]\n"
    "sales['big'] = is_big.astype(int)\n"
    f"sales['c'] = (time.sleep({S}), sales['t'] + 1)[1]"
)
CELLS = [SETUP, RAW, CLEAN]
NAMES = ["time", "pd", "raw", "sales", "is_big"]


@pytest.fixture
def clean_backend(tmp_path):
    """RAM over disk, as a kernel has: a restart keeps only the disk tier."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


def _restart(magics, backend) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    backend.backends[0].clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> dict[str, str]:
    return {
        m.get("code"): m.get("status") for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")
    }


def test_the_costly_steps_are_not_run_again_after_a_restart(cash_magics, clean_backend):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    _restart(cash_magics, clean_backend)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    statuses = _statuses(cash_magics)
    ran = [code for code, status in statuses.items() if status == CacheStatus.COMPUTED]
    assert not [code for code in ran if "sleep" in code], statuses
    ns = cash_magics.shell.user_ns
    assert list(ns["sales"].columns) == ["a", "t", "big", "c"]
    assert int(ns["sales"]["big"].sum()) == 199
    assert int(ns["sales"]["c"].iloc[-1]) == 599
    assert int(ns["is_big"].sum()) == 199
