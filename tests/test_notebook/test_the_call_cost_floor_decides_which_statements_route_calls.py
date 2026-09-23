"""The configured call cost floor decides which statements keep routing calls.

A statement that ran under the call cost floor, with no call served from the
cache inside it, has no call worth storing, so its calls stop being routed
through the call cache. That floor is ``call_cost_floor_seconds`` -- the same
one a call must clear to be stored. The router looked for it where it was
not, and always used the 3 ms default instead: ``call_cost_floor_seconds=0``
still stopped routing a fast statement's calls, and a raised floor still
routed a statement too fast for any of its calls to be stored.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
from tests._cell_driver import run_cash_cell


class _Shell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    magics = CashMagics(_Shell(), cash)
    magics.test_cash = cash
    magics._auto_cache_enabled = True
    yield magics
    backend.clear()


def _routes_calls(magics, code: str) -> bool:
    tree = ast.parse(code)
    routed, _ = magics._statement_processor._calls.code_and_tree_for_execution(code, tree, None)
    return routed is not code


def test_a_zero_floor_keeps_routing_a_fast_statement(magics):
    magics.test_cash.config.call_cost_floor_seconds = 0.0
    run_cash_cell(magics, "def bump(v):\n    return v + 1\nx = 1")
    # The statement's run time is wall time under cash, which a loaded
    # machine stretches past any small floor; learn from a 1 ms run directly.
    magics._statement_processor._calls.learn_call_wrapping("y = bump(x)", 0.001, [])

    assert _routes_calls(magics, "y = bump(x)")


def test_a_raised_floor_stops_routing_a_statement_below_it(magics):
    magics.test_cash.config.call_cost_floor_seconds = 5.0
    run_cash_cell(magics, "import time\ndef slow(v):\n    time.sleep(0.02)\n    return v + 1\nx = 1")
    run_cash_cell(magics, "y = slow(x)")

    assert not _routes_calls(magics, "y = slow(x)")
