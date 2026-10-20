"""A cell that changes a value it reads (``w -= x``), run again on its own,
starts from the value the cells above give it, whatever its own cache entry
does.

An array carries no lineage, so the upstream check put the lineage of ``w``
back to the cell's entry state and left the value to the cell's own cache
entry. When the entry could not be used -- the file the cell reads changed,
or the entry outlived the cell's TTL -- the cell ran on the value it had
written itself: ``w -= float(open(p).read())`` gave -4 after the file went
from 1 to 3, where a run from the top gives -3. Now ``w`` is rebuilt by the
cells above unless the entry serves it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.mark.parametrize("pinned", [True, False], ids=["every-step-stored", "cheap-steps-not-stored"])
def test_a_changed_file_is_read_onto_the_base(cash_magics, tmp_path, monkeypatch, pinned):
    if pinned:
        # A loaded machine stores the cheap steps too.
        monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    path = tmp_path / "n.txt"
    path.write_text("1")
    cells = ["import numpy as np", "w = np.zeros(3)", f"w -= float(open({str(path)!r}).read())"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    path.write_text("3")
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["w"].tolist() == _plain(cells)["w"].tolist() == [-3.0] * 3
    # Stored under the base's key, the -3 is what a second run alone gets too.
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["w"].tolist() == [-3.0] * 3


def test_an_entry_past_the_cells_ttl_is_not_left_to_restore_it(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["import numpy as np", "w = np.zeros(3)", "w -= 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    time.sleep(1.1)
    run_cash_cell(cash_magics, cells[2], cells=cells, ttl=1)
    assert cash_magics.shell.user_ns["w"].tolist() == [-1.0] * 3


def test_a_change_a_cell_below_made_is_not_kept_by_a_cell_run_alone(cash_magics, monkeypatch):
    """``vs['b'] += 100`` below wrote into ``vs``; ``vs['a'][:] = -5`` run
    alone starts from ``vs`` as the cells above it give it, and a read
    further down sees the change again once the cell below has run."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import numpy as np",
        "base = np.arange(10.0)",
        "vs = {'a': base[0:3], 'b': base[5:8]}",
        "vs['a'][:] = -1",
        "other = [vs['b']]",
        "vs['b'] += 100",
        "r = float(base.sum())\nq = float(other[0].sum())",
    ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[3] = "vs['a'][:] = -5"
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert np.array_equal(ns["vs"]["b"], _plain(cells[:4])["vs"]["b"])
    run_cash_cell(cash_magics, cells[6], cells=cells)
    plain = _plain(cells)
    assert (ns["r"], ns["q"]) == (plain["r"], plain["q"])
