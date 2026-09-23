"""A loop that moved a global RNG leaves no outcome record.

A persisted loop outcome is trusted in a later kernel instead of replaying the
loop, so a loop whose draws move a generator must not get one: skipping it
would leave every later draw on a different stream. The control-structure
processor used to judge "did the RNG move" with its own fingerprint of
``random`` and numpy only, so a loop drawing from torch was recorded. It now
reads the same capture the statement engine replays with.
"""

import ast
import sys
import types
from unittest.mock import MagicMock

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import FileBackend
from cash.core import Cash
from cash.notebook.cache_key import control_outcome_key
from cash.notebook.ipython.magics import CashMagics


class _FakeTorch(types.ModuleType):
    """Just enough of torch for ``capture_rng_state``: a stream that draws move."""

    def __init__(self):
        super().__init__("torch")
        self._position = 0
        self.cuda = types.SimpleNamespace(is_available=lambda: False)

    def get_rng_state(self):
        return self._position

    def set_rng_state(self, state):
        self._position = state

    def rand(self):
        self._position += 1
        return self._position / 100


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
def notebook(monkeypatch, tmp_path):
    torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    # A tier that keeps metadata alone: that is where the record goes.
    backend = FileBackend(cache_dir=str(tmp_path))
    shell = _Shell()
    magics = CashMagics(shell, Cash(backend=backend, register_magic=False))
    magics._auto_cache_enabled = True
    shell.user_ns["torch"] = torch
    yield magics, backend
    backend.clear()


def _loop_record(magics, backend, cell):
    magics._execute_cell(cell)
    loop = ast.unparse(ast.parse(cell).body[1])
    return backend.get_metadata(control_outcome_key(loop))


def test_a_loop_that_draws_nothing_is_recorded(notebook):
    """The control: the same loop shape without a draw keeps its record."""
    magics, backend = notebook
    assert _loop_record(magics, backend, "OUT = {}\nfor i in range(3):\n    OUT[i] = i * 2") is not None


def test_a_loop_that_draws_from_torch_is_not_recorded(notebook):
    magics, backend = notebook
    record = _loop_record(magics, backend, "OUT = {}\nfor i in range(3):\n    OUT[i] = torch.rand()")
    assert record is None, "the loop moved torch's generator and was still recorded as having no effect"
