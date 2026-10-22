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
from tests.test_notebook.restore.test_review_memofill4_e2e import _restart


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


@pytest.mark.parametrize(
    ("setting", "imported", "read"),
    [
        ("{m}.CFG = 5", "from {m} import CFG", "t = CFG + 1"),
        ("{m}.TABLE['k'] = 5", "from {m} import TABLE", "t = TABLE['k'] + 1"),
    ],
    ids=["rebound", "changed_in_place"],
)
def test_a_jump_after_a_restart_sets_the_module_before_importing_from_it(
    cash_magics, cash_instance, helpers, setting, imported, read
):
    """Run All, restart, run the last cell alone: the jump imported
    ``helpers`` again without running ``helpers.CFG = 5`` above, bound 1,
    and computed ``t`` as 2 where a Run All gives 6. The setting runs
    before the import, as it does in a Run All."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [f"import {helpers}", setting.format(m=helpers), imported.format(m=helpers), read, "u = t + 1"]
    ns = _run_all(cash_magics, cells)
    assert (ns["t"], ns["u"]) == (6, 7)

    sys.modules.pop(helpers, None)
    _restart(cash_magics, [helpers, "CFG", "TABLE", "t", "u"])
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    ns = cash_magics.shell.user_ns

    assert (ns["t"], ns["u"]) == (6, 7)


@pytest.mark.parametrize(
    ("defined", "setting"),
    [
        (
            "import functools, operator\nPART = functools.partial(operator.mul, 1)\n",
            "{m}.PART = functools.partial(operator.mul, {v})",
        ),
        (
            "import functools\ndef mul(x, k):\n    return x * k\nPART = functools.partial(mul, k=1)\n",
            "{m}.PART.keywords['k'] = {v}",
        ),
        (
            "class Scale:\n    k = 1\n    def __call__(self, x):\n        return x * self.k\nPART = Scale()\n",
            "{m}.PART.k = {v}",
        ),
    ],
    ids=["partial_rebound", "partial_changed_in_place", "callable_instance"],
)
def test_a_configured_callable_imported_by_name_follows_its_value(
    cash_magics, cash_instance, tmp_path, monkeypatch, defined, setting
):
    """``from helpers import PART``, a partial or an object with
    ``__call__`` that a cell above sets: it was keyed by its module's
    source alone, and ``t = PART(1)`` was served as 1 after the setting
    changed to 5."""
    name = f"_part_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(defined, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [
        f"import functools, operator, {name}",
        setting.format(m=name, v=1),
        f"from {name} import PART",
        "t = PART(1)",
        "u = t + 1",
    ]
    try:
        ns = _run_all(cash_magics, cells)
        assert (ns["t"], ns["u"]) == (1, 2)
        cells[1] = setting.format(m=name, v=5)
        ns = _run_all(cash_magics, cells)
        assert (ns["t"], ns["u"]) == (5, 6)
    finally:
        sys.modules.pop(name, None)


def test_a_value_too_costly_to_read_is_new_each_time(cash_magics, monkeypatch, helpers):
    """A 3-million entry dict imported by name was hashed in full, 6-7.5 s
    per Run All. Past the budget a change check has, it is taken as new:
    what reads it runs again, and nothing is served from an old value."""
    from cash import value_hash
    from cash.notebook import lineage_formula

    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.0)
    big = {i: str(i) for i in range(200_000)}
    first = lineage_formula._imported_value_component(helpers, "BIG", big)
    second = lineage_formula._imported_value_component(helpers, "BIG", big)

    assert first != second
