"""IPython's input history holds the code the user ran, with cash on.

Cash runs a cell's statements itself and then hands IPython a stand-in cell
(``pass``, or ``raise __cash_exception__`` when the cell failed) so the
execution count and the events still happen once. IPython stored the
stand-in as the cell's input, so ``_i``, ``In``, ``_ih``, ``%history``,
``%save`` and ``%rerun`` all lost the user's code.
"""

import asyncio

import pytest

from cash.notebook.ipython.magics import CashMagics


@pytest.fixture
def shell(cash_instance):
    """A real IPython shell under ``%cash_on``: the history is IPython's own,
    which the mock shell does not have."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    magics = CashMagics(shell, cash_instance)
    magics.cash_on("")
    try:
        yield shell
    finally:
        InteractiveShell.clear_instance()


def _inputs(shell):
    return [s for s in shell.history_manager.input_hist_raw if s]


def test_each_cell_is_stored_once_as_its_own_source(shell):
    shell.run_cell("k = 2", store_history=True)
    shell.run_cell("k * 10", store_history=True)

    assert _inputs(shell)[-2:] == ["k = 2", "k * 10"]
    assert "pass" not in _inputs(shell)
    assert shell.user_ns["In"][-1] == "k * 10"


def test_underscore_i_inside_a_cell_is_the_cell_before(shell):
    """IPython stores a cell's input before running it: ``_i`` read in the
    cell is the previous input, and ``In[-1]`` the cell itself."""
    shell.run_cell("k = 2", store_history=True)
    shell.run_cell("seen = (_i, In[-1])", store_history=True)

    assert shell.user_ns["seen"] == ("k = 2", "seen = (_i, In[-1])")


def test_a_magic_cell_is_stored_once_raw_and_parsed(shell):
    """A cell IPython runs itself is stored once, not once by cash and once
    by IPython."""
    shell.run_cell("k = 1", store_history=True)
    shell.run_cell("%time k = 2", store_history=True)

    assert _inputs(shell)[-2:] == ["k = 1", "%time k = 2"]
    assert "run_line_magic" in shell.history_manager.input_hist_parsed[-1]


def test_a_failing_cell_is_stored_as_its_own_source(shell):
    shell.run_cell("k = 1 / 0", store_history=True)

    assert _inputs(shell)[-1] == "k = 1 / 0"
    assert "__cash_exception__" not in "".join(_inputs(shell))


def test_the_execution_count_matches_the_entries(shell):
    start = shell.execution_count
    shell.run_cell("a = 1", store_history=True)
    shell.run_cell("b = 2", store_history=True)

    assert shell.execution_count == start + 2
    assert shell.user_ns[f"_i{start}"] == "a = 1"
    assert shell.user_ns[f"_i{start + 1}"] == "b = 2"


def test_a_cell_without_history_stores_nothing(shell):
    """``store_history=False`` leaves the history alone."""
    shell.run_cell("a = 1", store_history=True)
    before = list(shell.history_manager.input_hist_raw)
    shell.run_cell("c = 3", store_history=False)

    assert shell.history_manager.input_hist_raw == before
    assert _inputs(shell)[-1] == "a = 1"
    assert shell.user_ns["c"] == 3


def test_a_top_level_await_cell_is_stored_as_its_own_source(shell):
    source = "async def f():\n    return 41\nx = (await f()) + 1"

    async def run():
        return await shell.run_cell_async(
            source,
            store_history=True,
            transformed_cell=shell.transform_cell(source),
            preprocessing_exc_tuple=None,
        )

    asyncio.run(run())

    assert shell.user_ns["x"] == 42
    assert _inputs(shell)[-1] == source
    assert "pass" not in _inputs(shell)
