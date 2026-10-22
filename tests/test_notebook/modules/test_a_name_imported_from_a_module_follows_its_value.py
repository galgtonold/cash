"""``from helpers import CFG`` takes the value ``helpers.CFG`` holds now.

``helpers.CFG = 5`` in a cell above it, then Run All: ``from helpers
import CFG`` ran and bound 5, but ``t = CFG + 1`` below was served as 2,
where a plain kernel gives 6. The import's lineage named only the module
and the name, the same for any value it took, so whatever read ``CFG``
kept its old key.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    name = f"_helpers_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text("CFG = 1\nTABLE = {'k': 1}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


def _run_all(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    return cash_magics.shell.user_ns


@pytest.mark.parametrize(
    ("setting", "imported", "read", "first", "then"),
    [
        ("{m}.CFG = {v}", "from {m} import CFG", "t = CFG + 1", 2, 6),
        ("{m}.CFG = {v}", "from {m} import CFG as c", "t = c + 1", 2, 6),
        ("{m}.TABLE['k'] = {v}", "from {m} import TABLE", "t = TABLE['k'] + 1", 2, 6),
    ],
    ids=["rebound", "aliased", "changed_in_place"],
)
def test_a_value_set_on_the_module_above_reaches_what_reads_the_import(
    cash_magics, cash_instance, helpers, setting, imported, read, first, then
):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [
        f"import {helpers}",
        setting.format(m=helpers, v=1),
        imported.format(m=helpers),
        read,
        "u = t + 1",
    ]
    ns = _run_all(cash_magics, cells)
    assert (ns["t"], ns["u"]) == (first, first + 1)

    cells[1] = setting.format(m=helpers, v=5)
    ns = _run_all(cash_magics, cells)

    assert (ns["t"], ns["u"]) == (then, then + 1)


def test_an_unchanged_value_is_still_served(cash_magics, cash_instance, helpers):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [f"import {helpers}", f"from {helpers} import CFG", "t = CFG + 1"]
    _run_all(cash_magics, cells)

    _run_all(cash_magics, cells)

    assert cash_magics.shell.user_ns["t"] == 2
    assert str(cash_magics._last_cell_metrics["statements"][-1]["status"]) == "RESTORED"
