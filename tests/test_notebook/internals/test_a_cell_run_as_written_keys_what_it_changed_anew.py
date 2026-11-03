"""A cell cash runs as written keys what it changed in place anew.

When the upstream check cannot bring a cell up to date (``NOTEBOOK-REPAIR-GAP``
here), cash steps aside and IPython runs the cell as written. Cash watches
none of its statements, so a change it makes in place (``m.data[3]['v'] =
1000``) did not move ``m``'s lineage or the tag beside the object. A
``@cash.cache`` method keys ``self`` by that tag: called after the change, in
the cell or in a cell below, it was served the result stored for ``m`` as it
was before, and a statement below reading a list the cell appended to was
served its value from before the append.

The gap is made by taking ``Y = [7]`` out of what the backward scan schedules
and switching off the completion that would bring it back, as in
``test_a_plan_that_would_leave_the_cell_an_earlier_value_reruns_nothing``;
the cell then runs through IPython's own ``run_cell``, as in a kernel.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from cash.lineage_tag import own_tag
from cash.notebook.upstream.mismatch_classifier import MismatchClassifier
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

MODEL = (
    "import cash\n"
    "class Model:\n"
    "    def __init__(self, rows):\n"
    "        self.data = rows\n"
    "    @cash.cache\n"
    "    def total(self, k):\n"
    "        return sum(r['v'] for r in self.data) * k"
)
CELLS = [
    "import time\n" + MODEL,
    "base = 1",
    "X, Y = [base], [10, 20]",
    "Y = [7]",
    "m = Model([{'v': i} for i in range(200)])\nbefore = m.total(2)\nlst = [1, 2, 3]",
    "Z = (X, Y)",
    f"later = m.total(2)\nn = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), len(lst))[1]",
]
GAP_CELL = "m.data[3]['v'] = 1000\nedited = m.total(2)\nlst.append(4)\nZ = (X, Y)"
EDITED = [CELLS[0], "base = 2", *CELLS[2:5], GAP_CELL, CELLS[6]]
NAMES = ("before", "edited", "later", "n")


def _plain(cells: list[str]) -> dict:
    """What restart and Run All in plain Python leaves."""
    ns: dict = {"__name__": "__main__"}
    for cell in cells:
        exec(
            cell.replace("    @cash.cache\n", "").replace("time.sleep", "float"), ns
        )  # the notebook under test, without cash
    return {name: ns[name] for name in NAMES}


_scan = MismatchClassifier.backward_scan_pass


def _scan_without_y(self, sim, result, **kwargs):
    run, restored, cost = _scan(self, sim, result, **kwargs)
    return [i for i in run if sim.trace[i].stmt_code != "Y = [7]"], restored, cost


@pytest.fixture
def shell(cash_magics, mock_shell):
    """The notebook shell under ``%cash_on``. A cell goes through the
    magics' ``run_cell`` hook, which runs the cell cash steps aside from
    through the shell's own ``run_cell``. ``n = ...`` sleeps past the
    persistence floor, so a stale entry for it would be found."""
    cash_magics.cash_on("")
    return mock_shell


def _run(shell, cell: str, cells: list[str]) -> None:
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        result = shell.run_cell(cell)
    error = getattr(result, "error_in_exec", None)
    if isinstance(error, BaseException):
        raise error


def _run_the_gap(shell) -> list:
    with (
        patch.object(MismatchClassifier, "backward_scan_pass", _scan_without_y),
        patch.object(ReexecutionPlanner, "_complete_shadowed_var_producers", lambda self, indices, *a, **k: indices),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        _run(shell, GAP_CELL, EDITED)
    return [w.message.code for w in caught if isinstance(w.message, CashCacheIneffectiveWarning)]


def test_a_cached_method_called_after_a_change_in_the_cell_sees_it(shell):
    for cell in CELLS:
        _run(shell, cell, CELLS)

    told = _run_the_gap(shell)
    _run(shell, EDITED[-1], EDITED)

    assert "NOTEBOOK-REPAIR-GAP" in told, told
    ns = shell.user_ns
    assert {name: ns[name] for name in NAMES} == _plain(EDITED)


def test_what_the_cell_did_not_change_keeps_its_lineage(shell, statement_processor):
    """Control: a value the cell run as written only read keeps its lineage
    and its tag, so the statements below reading it are still served; the
    one it changed in place loses both, and is keyed by what it holds."""
    for cell in CELLS:
        _run(shell, cell, CELLS)
    lineage = statement_processor.tracking_state.variable_lineage
    m_before = lineage["m"]

    reader = "again = m.total(2)\nlst.append(5)\nZ = (X, Y)"
    with (
        patch.object(MismatchClassifier, "backward_scan_pass", _scan_without_y),
        patch.object(ReexecutionPlanner, "_complete_shadowed_var_producers", lambda self, indices, *a, **k: indices),
        warnings.catch_warnings(record=True),
    ):
        warnings.simplefilter("always")
        _run(shell, reader, [*EDITED[:5], reader, EDITED[6]])

    assert shell.user_ns["again"] == shell.user_ns["before"]
    assert lineage.get("m") == m_before
    assert own_tag(shell.user_ns["m"]) == m_before
    assert "lst" not in lineage
