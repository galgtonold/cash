"""A statement that sets state on a local module runs again; it is never restored.

``metrics.increment(5)`` with the module keeping a counter: run again, the
statement was served from the cache, and the counter stayed at 5 where a
plain kernel has 10. A hit restores what the statement bound, never the
module's globals. After a kernel restart the counter stayed at 0. The
statement now re-executes, like one whose callee changes a notebook global.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell

LIB = (
    "COUNT = 0\n"
    "def increment(n):\n    global COUNT\n    COUNT += n\n    return COUNT\n"
    "def _bump(n):\n    global COUNT\n    COUNT += n\n"
    "def increment_via_helper(n):\n    _bump(n)\n    return n\n"
)


@pytest.fixture
def metrics_module(tmp_path, monkeypatch):
    name = f"_metrics_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("body", "per_run"),
    [
        ("{m}.increment(5)", 5),
        ("{m}.increment_via_helper(5)", 5),
        ("from {m} import increment\nincrement(5)", 5),
        ("def run():\n    {m}.increment(5)\nrun()", 5),
        ("for _ in range(2):\n    {m}.increment(5)", 10),
    ],
    ids=["module_function", "through_a_helper", "from_imported", "notebook_function", "loop_body"],
)
def test_running_the_cell_again_counts_again(cash_magics, cash_instance, mock_shell, metrics_module, body, per_run):
    # Statements are stored however cheap, as on a slow machine.
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [f"import {metrics_module}", body.format(m=metrics_module)]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert sys.modules[metrics_module].COUNT == per_run

    run_cash_cell(cash_magics, cells[1], cells=cells)

    assert sys.modules[metrics_module].COUNT == 2 * per_run
