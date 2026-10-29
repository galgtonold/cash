"""Two made modules swapped, and a made module read by a function.

``m, k = k, m`` between ``u = m.v + 1`` and ``u2 = m.v + 1``: the swap
rule re-ran the producers of ``m`` and ``k`` to rebuild what the cell
started from, and those made two empty modules, without the ``m.v = ...``
set on them; then a hit on ``m = types.ModuleType('m')`` kept whatever
``m`` held, after the swap the other module. Both raised where plain
Python gives ``1 102``.

``def g(): return m.v`` and ``u = g() + 1``: what ``g`` read of ``m`` was
not in ``u``'s key, only what a statement names itself, so after a restart
``u`` was built again after the cell below moved ``m.v`` on (9 against 8).
"""

import types

import pytest

from cash.notebook.callee_reach import reached_user_code
from tests._cell_driver import run_cash_cell
from tools.plain_checker import check, notebook

SWAP = [
    "import itertools, types\nit = itertools.count()\nm = types.ModuleType('m')\nk = types.ModuleType('k')",
    "m.v = next(it)\nk.v = 100 + next(it)",
    "u = m.v + 1",
    "m, k = k, m",
    "u2 = m.v + 1",
]


@pytest.fixture
def every_statement_cached(monkeypatch):
    monkeypatch.setenv("CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS", "0")
    monkeypatch.setenv("CASH_CALL_COST_FLOOR_SECONDS", "0")
    import cash.notebook.call_unit as call_unit

    monkeypatch.setattr(call_unit, "_COST_FLOOR_S", 0)


def test_swapping_two_made_modules_keeps_what_was_set_on_them(cash_magics, mock_shell):
    for cell in SWAP:
        run_cash_cell(cash_magics, cell, cells=SWAP)
    assert (mock_shell.user_ns["u"], mock_shell.user_ns["u2"]) == (1, 102)


def test_a_hit_on_making_a_module_gives_that_module_again(every_statement_cached, cash_magics, mock_shell):
    cells = list(SWAP)
    for _run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (mock_shell.user_ns["u"], mock_shell.user_ns["u2"]) == (1, 102)
        cells[0] = cells[0].replace("count()", "count(0)")


def test_a_function_reading_a_made_module_reads_it_as_data():
    m = types.ModuleType("m_made_here")
    m.v = 3

    namespace = {"__name__": "__main__", "m": m}
    exec(compile("def g():\n    return m.v\n", "<cell>", "exec"), namespace)
    assert reached_user_code("u = g() + 1", namespace).data == (("@m.v", 3),)


SHAPES = [
    notebook(
        "made-modules-swapped",
        SWAP[:4] + [SWAP[4] + "\nprint(u, u2)"],
        names=["u", "u2"],
        edits={1: {0: SWAP[0].replace("count()", "count(7)")}},
    ),
    notebook(
        "made-module-read-by-a-function",
        [
            "import itertools, types\nit = itertools.count()\nm = types.ModuleType('m')\ndef g():\n    return m.v",
            "m.v = next(it)",
            "u = g() + 1",
            "m.v = next(it)",
            "u2 = g() + 1\nprint(u, u2)",
        ],
        names=["u", "u2"],
        edits={
            1: {
                0: "import itertools, types\nit = itertools.count(7)\nm = types.ModuleType('m')\ndef g():\n    return m.v"
            }
        },
    ),
]


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(900)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_a_restart_and_run_all_give_what_plain_python_gives(shape, tmp_path):
    check(shape, tmp_path).assert_ok()
