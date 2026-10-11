"""A cell run that skips steps leaves every name at the value the edited
cell makes.

``x = load(); x = heavy(x); y = ok(x); x = x + 1``, edited to ``y =
triple(x)`` and ``x = x + 2``: ``ok`` returned ``x`` itself, so the run
recorded ``y`` as an alias of ``x``, and the simulation of the edited cell
credited ``x = x + 2`` with a new version of ``y``. The plan restored
``heavy``, ran the last line and skipped ``y = triple(x)`` as covered: ``y``
kept the removed line's value, and with ``y = boom(x)`` the cell did not
raise at all. A name the plan leaves to a statement that does not name it
is not covered; the cell runs as it would without cash.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

HELPERS = f"""
import time, numpy as np
def load():
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S}); return np.arange(10.0)
def heavy(x):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S}); return x + 100
def ok(x):
    return x
def triple(x):
    return x * 3
def boom(x):
    raise ValueError("boom")
"""
FIRST = "x = h.load()\nx = h.heavy(x)\ny = h.ok(x)\nx = x + 1"


def _first_run(cash_magics, tmp_path, monkeypatch, edited):
    (tmp_path / "h.py").write_text(HELPERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "h", raising=False)
    run_cash_cell(cash_magics, "import h", cells=["import h", FIRST])
    run_cash_cell(cash_magics, FIRST, cells=["import h", FIRST])
    return ["import h", edited]


def test_the_edited_line_runs(cash_magics, tmp_path, monkeypatch):
    edited = "x = h.load()\nx = h.heavy(x)\ny = h.triple(x)\nx = x + 2"
    cells = _first_run(cash_magics, tmp_path, monkeypatch, edited)
    run_cash_cell(cash_magics, edited, cells=cells)
    ns = cash_magics.shell.user_ns
    assert float(ns["y"].sum()) == 3135.0  # triple(heavy(load())), not the old ok() value 1045
    assert float(ns["x"].sum()) == 1065.0


def test_an_edited_line_that_raises_raises(cash_magics, tmp_path, monkeypatch):
    edited = "x = h.load()\nx = h.heavy(x)\ny = h.boom(x)\nx = x + 2"
    cells = _first_run(cash_magics, tmp_path, monkeypatch, edited)
    with pytest.raises(ValueError, match="boom"):
        run_cash_cell(cash_magics, edited, cells=cells)
    ns = cash_magics.shell.user_ns
    assert float(ns["x"].sum()) == 1045.0  # as a plain run leaves it: the last line never ran


def test_only_the_y_line_edited_still_runs_it(cash_magics, tmp_path, monkeypatch):
    """The control: the case that was already right stays right."""
    edited = "x = h.load()\nx = h.heavy(x)\ny = h.triple(x)\nx = x + 1"
    cells = _first_run(cash_magics, tmp_path, monkeypatch, edited)
    run_cash_cell(cash_magics, edited, cells=cells)
    ns = cash_magics.shell.user_ns
    assert float(ns["y"].sum()) == 3135.0
    assert float(ns["x"].sum()) == 1055.0
