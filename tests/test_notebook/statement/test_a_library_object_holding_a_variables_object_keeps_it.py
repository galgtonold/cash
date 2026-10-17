"""A library's object holding another variable's objects inside it is stored
with that variable, so a hit gives back one graph.

``optimizer = torch.optim.SGD(model.parameters())`` holds ``model``'s
parameters in its ``param_groups``. The share check read the optimizer as one
closed object, saw nothing else holding it, and stored it alone: a hit bound
a copy whose parameters were not ``model``'s, ``optimizer.step()`` trained
the copy, and ``model`` never trained (E11, E12). The objects inside a
library object are now read down, and a variable holding one of them is
stored with it (`shared_objects._inside_library_objects`).
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 0.5)[1]"


def _status(magics, code: str) -> str | None:
    statements = magics.cash_status("dict")["last_cell"]["statements"]
    return next((m.get("status") for m in statements if m.get("code") == code), None)


def _run_twice(magics, cells: list[str]) -> str | None:
    """Run *cells*, then again in the same kernel; the second run's status
    of the second cell's last statement."""
    for _ in range(2):
        for i, cell in enumerate(cells):
            run_cash_cell(magics, cell, cells=cells)
            if i == 1:
                status = _status(magics, cell.splitlines()[-1])
    return status


def test_a_restored_optimizer_trains_the_model(cash_magics):
    torch = pytest.importorskip("torch")
    cells = [
        "import time\nimport torch\nimport torch.nn as nn\ntorch.manual_seed(0)\n"
        "X = torch.randn(256, 4)\ny = (X[:, 0] + X[:, 1] > 0).long()",
        f"model = nn.Linear(4, 2)\noptimizer = torch.optim.SGD(model.parameters(), lr={SLOW})",
        "for _ in range(100):\n"
        "    optimizer.zero_grad()\n"
        "    loss = nn.functional.cross_entropy(model(X), y)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "final = loss.item()",
    ]

    assert _run_twice(cash_magics, cells) == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["optimizer"].param_groups[0]["params"][0] is ns["model"].weight
    assert round(ns["final"], 4) == 0.1076, "the model trained, as in plain Python"
    assert isinstance(ns["model"], torch.nn.Linear)


def test_a_library_object_holding_a_variables_list_keeps_it(cash_magics):
    """The same without torch: an ``argparse.Namespace`` is a library's
    object the walk does not read, holding the list ``rows`` holds."""
    cells = [
        "import time\nimport argparse",
        f"rows = [1, 2]\nopts = argparse.Namespace(rows=rows, rate={SLOW})",
    ]

    assert _run_twice(cash_magics, cells) == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["opts"].rows is ns["rows"]
    ns["rows"].append(3)
    assert ns["opts"].rows == [1, 2, 3]


def test_a_notebook_class_holding_a_variables_list_keeps_it(cash_magics):
    """A class a cell defines: in a kernel its module is ``__main__`` and the
    walk reads its objects; under this test shell it is not, and they are
    read as a library's are. Either way ``opts.rows`` stays ``rows``."""
    cells = [
        "import time\nclass Opts:\n    def __init__(self, rows, rate):\n        self.rows, self.rate = rows, rate",
        f"rows = [1, 2]\nopts = Opts(rows, {SLOW})",
    ]

    assert _run_twice(cash_magics, cells) == CacheStatus.RESTORED
    ns = cash_magics.shell.user_ns
    assert ns["opts"].rows is ns["rows"]


def test_a_library_object_holding_only_its_own_objects_is_stored_alone(cash_magics, statement_processor):
    """Control: nothing else holds what is inside, so nothing joins it."""
    run_cash_cell(cash_magics, "import time\nimport argparse")
    metrics = statement_processor.process_statement(f"opts = argparse.Namespace(rows=[1, 2], rate={SLOW})")
    assert metrics["status"] == CacheStatus.COMPUTED and not metrics.get("uncacheable_reasons")
    assert not statement_processor.tracking_state.held_with


def test_figures_whose_parts_hold_them_are_stored_alone(cash_magics, statement_processor):
    """Control, as decided for E23: a plotly figure's parts hold it, and
    share plotly's validators with every other figure. It keeps the
    library-object rule: stored alone, with no other figure joining it."""
    pytest.importorskip("plotly")
    run_cash_cell(cash_magics, "import time\nimport plotly.graph_objects as go")
    statement_processor.process_statement(f"first = go.Figure(go.Scatter(x=[1, 2], y=[{SLOW}, 1]))")
    metrics = statement_processor.process_statement(f"second = go.Figure(go.Scatter(x=[1, 2], y=[{SLOW}, 2]))")
    assert metrics["status"] == CacheStatus.COMPUTED and not metrics.get("uncacheable_reasons")
    assert not any(statement_processor.tracking_state.held_with.values())
