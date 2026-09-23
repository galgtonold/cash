"""The async statement path takes the same requests as the sync one.

``process_statement_async`` runs a statement holding a top-level ``await``.
Everything but the execution itself is shared with ``process_statement``,
so a request the sync path honours must not be dropped on the async one.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class _Shell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def processor():
    backend = InMemoryBackend()
    magics = CashMagics(_Shell(), Cash(backend=backend, register_magic=False))
    yield magics._statement_processor
    backend.clear()


def test_forced_outputs_are_captured_on_the_async_path(processor):
    """The accumulator-loop fast path names its accumulator as a forced
    output. The async path ignored the argument, so the accumulator was
    neither captured nor restored there."""
    processor.shell.user_ns["acc"] = []
    code = "for e in [1, 2]:\n    acc.append(e)"

    sync = processor.process_statement(code, silent=True, force_outputs={"acc", "e"})
    processor.shell.user_ns["acc"] = []
    asynchronous = asyncio.run(processor.process_statement_async(code, silent=True, force_outputs={"acc", "e"}))

    assert {"acc", "e"} <= set(sync["evaluated_vars"])
    assert set(asynchronous["evaluated_vars"]) == set(sync["evaluated_vars"])
