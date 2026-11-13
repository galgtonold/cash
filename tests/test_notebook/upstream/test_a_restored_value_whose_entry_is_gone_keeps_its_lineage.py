"""A value restored this session keeps its lineage once its entry is gone.

After a restart ``full_pipeline = Pipeline([...])`` was restored, its entry
was dropped (evicted, or judged no faster than its compute), and the next
cell's upstream check simulated it as a statement that never ran: with no
record of the files it read itself, it built the lineage from the files its
value inherited (``housing.csv``), which no run of it makes. That lineage
did not match the restored value's, so the value was taken for stale and
rebuilt, with everything sharing objects with it. The simulation now takes
the metadata the statement was restored with (`TrackingState.restored_entries`),
checked against the files it recorded as a stored entry is.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from tests._cell_driver import run_cash_cell

CELLS = [
    "import time",
    "raw = (time.sleep(0.2), open(PATH).read())[1]",
    "small = (time.sleep(0.2), len(raw))[1]",
    "n = small + 1",
]
NAMES = ["time", "raw", "small", "n"]


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend: what a later kernel reads is what the disk tier kept."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _restored_then_dropped(magics, backend, tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("abc", encoding="utf-8")
    cells = [cell.replace("PATH", repr(str(data))) for cell in CELLS]
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    _restart(magics)
    run_cash_cell(magics, cells[0], cells=cells)
    run_cash_cell(magics, cells[3], cells=cells)
    ns = magics.shell.user_ns
    assert ns["n"] == 4
    assert "raw" not in ns, "control: small was restored, not rebuilt"
    gone = [m["key"] for m in backend.list_entries() if str(m.get("code", "")).startswith("small = ")]
    assert gone, "control: small has an entry"
    for key in gone:
        backend.delete(key)
    # Simulated afresh, as after a cell above it changed.
    magics._upstream_checker.simulator.cache.reset()
    return data, cells


def test_the_cell_below_is_served(cash_magics, clean_backend, tmp_path):
    _, cells = _restored_then_dropped(cash_magics, clean_backend, tmp_path)

    run_cash_cell(cash_magics, cells[3], cells=cells)

    ns = cash_magics.shell.user_ns
    assert ns["n"] == 4
    assert "raw" not in ns, "small was taken for stale and rebuilt"


def test_a_changed_file_still_rebuilds_it(cash_magics, clean_backend, tmp_path):
    data, cells = _restored_then_dropped(cash_magics, clean_backend, tmp_path)
    data.write_text("abcdef", encoding="utf-8")

    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert cash_magics.shell.user_ns["n"] == 7
