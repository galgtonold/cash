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

from tests._cell_driver import restart_kernel, run_cash_cell

HELPERS = (
    "CFG = 1\n"
    "TABLE = {'k': 1}\n"
    "class Scale:\n"
    "    def __init__(self):\n"
    "        self.k = 1\n"
    "    def __call__(self, x):\n"
    "        return x * self.k\n"
    "SCALE = Scale()\n"
)


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    name = f"_helpers_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(HELPERS, encoding="utf-8")
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
    ("imports", "setting", "imported", "read"),
    [
        ("import {m}", "{m}.CFG = 5", "from {m} import CFG", "t = CFG + 1"),
        ("import {m}", "{m}.TABLE['k'] = 5", "from {m} import TABLE", "t = TABLE['k'] + 1"),
        ("import {m} as h", "h.CFG = 5", "from {m} import CFG", "t = CFG + 1"),
        ("from {m} import TABLE", "TABLE['k'] = 5", "from {m} import TABLE as T2", "t = T2['k'] + 1"),
        ("import {m} as h", "h.SCALE.k = 5", "from {m} import SCALE", "t = SCALE(1) + 1"),
    ],
    ids=["rebound", "changed_in_place", "through_an_alias", "through_a_name_taken_from_it", "a_callable_by_alias"],
)
@pytest.mark.parametrize("imports_first", [False, True], ids=["jump", "imports_cell_then_jump"])
def test_a_jump_after_a_restart_sets_the_module_before_importing_from_it(
    cash_magics, cash_instance, helpers, imports, setting, imported, read, imports_first
):
    """Run All, restart, run the last cell alone: the jump imported
    ``helpers`` again without running ``helpers.CFG = 5`` above, bound 1,
    and computed ``t`` as 2 where a Run All gives 6. So did running the
    imports cell first (the module then loaded, the setting not run), and a
    setting made through an alias or a name taken from the module. The
    setting runs before the import, as it does in a Run All."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [imports.format(m=helpers), setting.format(m=helpers), imported.format(m=helpers), read, "u = t + 1"]
    ns = _run_all(cash_magics, cells)
    assert (ns["t"], ns["u"]) == (6, 7)

    sys.modules.pop(helpers, None)
    restart_kernel(cash_magics, [helpers, "h", "CFG", "TABLE", "T2", "SCALE", "t", "u"])
    if imports_first:
        run_cash_cell(cash_magics, cells[0], cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    ns = cash_magics.shell.user_ns

    assert (ns["t"], ns["u"]) == (6, 7)


def test_a_jump_runs_no_setting_the_imported_name_cannot_see(cash_magics, cash_instance, tmp_path, monkeypatch):
    """``helpers.DATA = helpers.slow()`` above ``from helpers import f``,
    where ``f`` reads no ``DATA``: a jump after a restart ran ``slow()``
    again, which a Run All's entries already stood for."""
    name = f"_slow_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(
        "CALLS = []\nDATA = None\ndef slow():\n    CALLS.append(1)\n    return 5\ndef f(x):\n    return x + 1\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [f"import {name}", f"{name}.DATA = {name}.slow()", f"from {name} import f", "t = f(1)", "u = t + 1"]
    try:
        ns = _run_all(cash_magics, cells)
        assert (ns["t"], ns["u"]) == (2, 3)
        sys.modules.pop(name, None)
        restart_kernel(cash_magics, [name, "f", "t", "u"])
        run_cash_cell(cash_magics, cells[-1], cells=cells)
        assert (cash_magics.shell.user_ns["t"], cash_magics.shell.user_ns["u"]) == (2, 3)
        assert sys.modules[name].CALLS == []
    finally:
        sys.modules.pop(name, None)


def test_a_partial_imported_by_name_is_served_after_a_jump(cash_magics, cash_instance, tmp_path, monkeypatch):
    """A partial digested one way while its import ran and another after:
    a jump after a restart never matched the Run All, and every Run All
    ran the import again before what read it."""
    name = f"_partj_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(
        "import functools\ndef _inc(x):\n    return x + 1\npart = functools.partial(_inc)\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [f"from {name} import part", "t = part(1)", "u = t + 1"]
    try:
        _run_all(cash_magics, cells)
        sys.modules.pop(name, None)
        restart_kernel(cash_magics, ["part", "t", "u"])
        run_cash_cell(cash_magics, cells[-1], cells=cells)
        statuses = [str(m["status"]) for m in cash_magics._last_cell_metrics["statements"]]
        assert statuses[-2:] == ["RESTORED", "RESTORED"]
        assert cash_magics.shell.user_ns["u"] == 3
    finally:
        sys.modules.pop(name, None)


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
