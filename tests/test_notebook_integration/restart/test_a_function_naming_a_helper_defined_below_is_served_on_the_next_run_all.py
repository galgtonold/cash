"""A function whose body names a helper defined in a cell below it keeps
one lineage, however often the notebook runs.

``def solve`` names ``seg``, defined in the next cell. Its lineage folded
the lineage ``seg`` had when the ``def`` ran: none on a first Run All or
after a restart, the earlier run's on the next Run All of the same kernel,
and the kernel's own in the upstream simulation. So the upstream check
re-ran ``solve`` as broken ("Lineage mismatch ... READ-ONLY"), and a cell
calling it through a tuple that holds it missed on the next Run All, as
the TSP notebook's exact benchmark cells did. A global only the body reads
is looked up at the call, and the call's key reads it there.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "%cash_badge print\nimport functools, time"
DEF = "def solve(n):\n    seg.cache_clear()\n    return seg(n) + 1"
HELPER = "@functools.cache\ndef seg(n):\n    time.sleep(0.3)\n    return n * {k}"
HOLDER = "pair = (len, solve)"
CALL = "r = [f([1, 2]) if f is len else f(3) for f in pair]\nprint('R', r)"
CELLS = [SETUP, DEF, HELPER.format(k=2), HOLDER, CALL]


def _again_after_a_restart(nb_runner) -> None:
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()


def _raises_name_error(nb_runner, cell: int) -> None:
    with pytest.raises(Exception, match="An error occurred"):
        nb_runner.run_cell(cell)
    errors = [o.get("ename") for o in nb_runner.get_cell(cell).outputs if o.output_type == "error"]
    assert errors == ["NameError"], errors


def test_a_call_through_a_function_naming_a_helper_below_is_served_on_the_next_run_all(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R [2, 7]" in nb_runner.get_output(5)

    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "R [2, 7]" in out, out
    assert shows_cached(out), out

    _again_after_a_restart(nb_runner)
    out = nb_runner.get_output(5)
    assert "R [2, 7]" in out, out
    assert shows_cached(out), out


def test_a_helper_below_defined_again_with_other_code_runs_the_call_again(nb_runner):
    """Control: the call reads the helper at the call, so new code below
    re-runs it, in the same kernel and after a restart."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, HELPER.format(k=3))
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "R [2, 10]" in out, out
    assert not shows_cached(out), out

    _again_after_a_restart(nb_runner)
    assert "R [2, 10]" in nb_runner.get_output(5), nb_runner.get_output(5)


def test_a_call_above_the_helper_raises_name_error_after_a_restart(nb_runner):
    """Control: a cell calling the function before the helper's cell has
    run raises ``NameError`` on a fresh Run All, as plain Python does,
    though the same kernel's next Run All found the helper there."""
    early = "t = solve(1)\nprint('T', t)"
    nb_runner.create_notebook([SETUP, DEF, early, HELPER.format(k=2)])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    _raises_name_error(nb_runner, 3)
    nb_runner.run_cell(4)

    nb_runner.run_all()
    assert "T 3" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cells([1, 2])
    _raises_name_error(nb_runner, 3)
