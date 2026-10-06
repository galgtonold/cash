"""An in-place write to an array or a large list keeps the very object.

``x[3] = 99.0`` in a later cell changes ``x`` in place. For a value whose
content cash does not hash (an array, a frame, a list or dict of more than 200
items), the check for "this cell's own earlier write is already in the value"
could not compare contents and always answered yes, already on the first Run
All: the producer ``x = np.arange(10.0)`` ran again and ``x`` was rebound to a
new array, so a view of ``x``, an alias and every container holding it kept the
old values (and the producer's side effects ran twice).

The check now asks whether the cell already started from this very object:
on a forward run the producer above has bound a new one. On an isolated re-run
of the writing cell it is the same object, and the producer still re-runs.
"""

from __future__ import annotations

import numpy as np

from tests._cell_driver import run_cash_cell

CELLS = [
    "import numpy as np",
    "x = np.arange(10.0)\nwindow = x[2:5]\nparams = {'w': x}",
    "x[3] = 99.0",
    "w = np.zeros(3)\nlayers = [w]",
    "w -= 1",
    "scores = [0.0] * 5000\nreport = {'scores': scores}",
    "scores[0] = 1.0",
]


def _run_all(magics):
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)


def test_a_run_all_writes_into_the_object_everything_else_holds(cash_magics):
    ns = cash_magics.shell.user_ns
    _run_all(cash_magics)
    assert ns["window"][1] == 99.0 and ns["params"]["w"][3] == 99.0
    assert ns["layers"][0].tolist() == [-1.0, -1.0, -1.0]
    assert ns["report"]["scores"] is ns["scores"] and ns["report"]["scores"][0] == 1.0


def test_a_second_run_all_too(cash_magics):
    ns = cash_magics.shell.user_ns
    _run_all(cash_magics)
    _run_all(cash_magics)
    assert ns["window"][1] == 99.0 and ns["params"]["w"][3] == 99.0
    assert ns["layers"][0].tolist() == [-1.0, -1.0, -1.0]
    assert ns["report"]["scores"] is ns["scores"]


def test_rerunning_the_writing_cell_alone_still_starts_from_its_base(cash_magics):
    """Control: the second run of ``w -= 1`` alone must not reach -2."""
    ns = cash_magics.shell.user_ns
    _run_all(cash_magics)
    run_cash_cell(cash_magics, "w -= 1", cells=CELLS)
    assert np.array_equal(ns["w"], [-1.0, -1.0, -1.0])
    run_cash_cell(cash_magics, "w -= 1", cells=CELLS)
    assert np.array_equal(ns["w"], [-1.0, -1.0, -1.0])
