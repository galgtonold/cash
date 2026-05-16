"""Regression test: ``analyze_statement`` must be computed at most once per
``process_statement`` call.

Before this fix, the same ``StatementAnalysis`` was computed twice — once in
``_check_skip_conditions`` (pre-execution, l.816) and again in ``_post_execute``
(post-execution, l.606). Both passed identical ``(code, tree)`` args, so three
AST visitors ran twice per statement on the hot path.
"""
from __future__ import annotations

import pytest
from traitlets.config import Configurable
from unittest.mock import MagicMock, patch

from cash.backends.backend import InMemoryBackend
from cash.core import Cash
from cash.notebook.magics import CashMagics


class _MockShell(Configurable):
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
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def test_analyze_statement_called_once_per_processed_statement(magics_fixture):
    """A single cacheable assignment must trigger exactly one analyze_statement
    call, not two.

    We patch at the call site (statement_processor) rather than at the source
    (cacheability) so that internal cacheability tests are unaffected.
    """
    magics, shell, _ = magics_fixture

    real_analyze = __import__(
        "cash.notebook.statement_processor", fromlist=["analyze_statement"]
    ).analyze_statement

    with patch(
        "cash.notebook.statement_processor.analyze_statement",
        wraps=real_analyze,
    ) as spy:
        magics.cash("", "y = 1 + 2")

    # The cell runs one statement (`y = 1 + 2`). Pre-fix this was 2 calls; the
    # de-dup contract is exactly 1 per processed statement.
    assert spy.call_count == 1, (
        f"Expected exactly 1 analyze_statement call, got {spy.call_count}. "
        f"calls: {spy.call_args_list}"
    )
