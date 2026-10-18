"""A cell's final value is weighed against what a restart would re-run.

``arr = load()`` then ``arr = step(arr, 3.0)`` in one cell: the last step
took 0.27 s and restoring its 38 MB took about as long, so on some runs the
value was refused by its own statement's time alone. The end-of-cell pass,
which persists the final version by what rebuilding it costs, then had no
value to write, and after a restart the whole chain ran again. The refusal
now weighs the final value by its own time plus the unsaved entries its
inputs came through; an intermediate is still weighed by its own.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends import memory_backend
from tests._cell_driver import run_cash_cell

SETUP = (
    "import time\n"
    "import numpy as np\n"
    "def load():\n"
    "    time.sleep(0.4)\n"
    "    return np.full(2_000_000, 1.0)\n"
    "def step(a, k):\n"
    "    time.sleep(0.02)\n"
    "    return a * k"
)


@pytest.fixture
def slow_copy(monkeypatch):
    """A 16 MB array costs 0.16 s to copy back: more than one step (0.02 s),
    less than the chain behind it (0.4 s). Pinned, not measured: the RAM
    tier's own readings would take over as it copies."""

    def hit_seconds(value, nbytes):
        return 0.16 if nbytes > 10_000_000 else None

    monkeypatch.setattr(memory_backend, "hit_seconds", hit_seconds)


def _entries(cash, code):
    return [e for e in cash.backend.list_entries() if e.get("code") == code]


def _kept(cash, code):
    return [e for e in _entries(cash, code) if not e.get("metadata_only")]


def test_the_final_value_over_an_unsaved_chain_is_kept(cash_magics, cash_instance, slow_copy):
    cell = "arr = load()\narr = step(arr, 2.0)\narr = step(arr, 3.0)"
    cells = [SETUP, cell]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for c in cells:
        run_cash_cell(cash_magics, c, cells=cells)
    assert cash_magics.shell.user_ns["arr"][0] == 6.0
    assert _kept(cash_instance, "arr = step(arr, 3.0)"), _entries(cash_instance, "arr = step(arr, 3.0)")


def test_an_intermediate_is_still_weighed_by_its_own_time(cash_magics, cash_instance, slow_copy):
    """The control: the cell replaces ``arr = step(arr, 2.0)``, so a
    restart never restores it, and copying it back costs more than its
    0.02 s."""
    cell = "arr = load()\narr = step(arr, 2.0)\narr = step(arr, 3.0)"
    cells = [SETUP, cell]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for c in cells:
        run_cash_cell(cash_magics, c, cells=cells)
    assert not _kept(cash_instance, "arr = step(arr, 2.0)")
