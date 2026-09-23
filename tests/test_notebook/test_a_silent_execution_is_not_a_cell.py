"""A ``silent`` execution is a frontend's own code, not a notebook cell.

Jupyter frontends and tools send ``silent=True`` for code of their own: a
variable explorer's query, a coverage hook relabelling the kernel between
tests. Cash put it through the cell pipeline: its names got lineage (the
coverage hook's ``_cov`` and ``_cov_mod`` were in the lineage store after
the hook had deleted them) and it was planned against the notebook's cells
as if it were one of them. It runs as plain IPython now.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.cell_executor import EarlyReturn
from cash.notebook.ipython.magics import CashMagics


class _Shell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.user_global_ns = self.user_ns
        self.input_transformers_cleanup = []
        self.ast_transformers = []
        self.events = MagicMock()
        self.run_cell = MagicMock(return_value="plain")
        self.run_cell_async = AsyncMock(return_value="plain-async")


@pytest.fixture
def magics():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    magics = CashMagics(_Shell(), cash)
    magics._auto_cache_enabled = True
    magics._cell_executor = MagicMock()
    magics._cell_executor.execute_cell.return_value = EarlyReturn("cash")
    magics._cell_executor.execute_cell_async = AsyncMock(return_value=EarlyReturn("cash-async"))
    return magics


@pytest.mark.parametrize(
    "args, kwargs",
    [((), {"store_history": False, "silent": True}), ((False, True), {})],
    ids=["keyword", "positional"],
)
def test_a_silent_run_cell_bypasses_cash(magics, args, kwargs):
    original = magics._original_run_cell
    assert magics._execute_cell("_cov = 1", *args, **kwargs) == "plain"
    original.assert_called_once_with("_cov = 1", *args, **kwargs)
    magics._cell_executor.execute_cell.assert_not_called()


def test_a_silent_run_cell_async_bypasses_cash(magics):
    original = magics._original_run_cell_async
    result = asyncio.run(magics._execute_cell_async("_cov = 1", store_history=False, silent=True))
    assert result == "plain-async"
    original.assert_awaited_once()
    magics._cell_executor.execute_cell_async.assert_not_called()


def test_a_cell_the_user_runs_still_goes_through_cash(magics):
    """Positive control: ``silent=False`` is every cell a user runs."""
    assert magics._execute_cell("x = 1", store_history=True, silent=False) == "cash"
    assert asyncio.run(magics._execute_cell_async("x = 1", store_history=True, silent=False)) == "cash-async"
