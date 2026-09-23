"""A top-level-await cell finishes on every supported IPython.

ipykernel runs a cell with top-level ``await`` through ``shell.run_cell_async``,
passing ``transformed_cell=``. Cash runs the statements itself, then hands
IPython a stand-in cell (``"pass"``, or ``"raise __cash_exception__"`` when the
cell failed) so the execution count, history and events still happen once.

From IPython 9.16 ``run_cell_async`` raises ``TypeError`` when
``transformed_cell`` is missing, so the stand-in must carry its own transform:
dropping it made every top-level-await cell fail (in a kernel: hang at ``[*]``).
Passing the caller's transform instead would re-run the whole user cell.
"""

import asyncio

import pytest

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


@pytest.fixture
def shell():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    magics = CashMagics(shell, Cash(backend=InMemoryBackend(), register_magic=False))
    magics._auto_cache_enabled = True
    try:
        yield shell
    finally:
        InteractiveShell.clear_instance()


def _run_like_ipykernel(shell, source):
    async def run():
        return await shell.run_cell_async(
            source,
            store_history=True,
            transformed_cell=shell.transform_cell(source),
            preprocessing_exc_tuple=None,
        )

    return asyncio.run(run())


def test_a_top_level_await_cell_runs_once_and_succeeds(shell):
    shell.user_ns["calls"] = []
    source = "async def f():\n    calls.append(1)\n    return 41\nx = (await f()) + 1\n"

    result = _run_like_ipykernel(shell, source)

    assert result.error_before_exec is None
    assert result.error_in_exec is None
    assert shell.user_ns["x"] == 42
    assert shell.user_ns["calls"] == [1], "the user cell ran more than once"


def test_a_failing_top_level_await_cell_reports_its_own_error(shell):
    source = "async def f():\n    raise ValueError('boom')\nx = await f()\n"

    result = _run_like_ipykernel(shell, source)

    assert isinstance(result.error_in_exec, ValueError)
    assert str(result.error_in_exec) == "boom"
