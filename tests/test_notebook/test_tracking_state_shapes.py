"""`executed_cell_hashes` maps a variable to a SET of hashes, never one hash.

A variable can be defined by more than one statement across a session -- a
re-run cell, a loop body, a restore from cache -- so the checker asks whether
the code it simulated is *among* the hashes recorded for that variable. Every
writer builds it that way (`statement/lineage.py`, `statement/restore.py`,
`restore.py`, `upstream/_types.py`: `[var] = set()` then `.add(...)`).

The declared type said `dict[str, str]` for long enough that a test fixture
believed it and stored a bare string. That is not a cosmetic mismatch: the next
writer's `.add()` raised `AttributeError: 'str' object has no attribute 'add'`,
cash bailed out of its own pipeline, and the cell ran UNCACHED through IPython
-- while the test that planted the string still passed, because the value it
asserted on was produced by the fallback rather than by the code path it named.

So this file pins the shape at both ends: what a real run writes, and the
declared type that told the fixture otherwise.

Deliberately NOT pinned: what cash does when handed a string anyway. It depends
on which writer touches the value first -- the hook path bails out with a
NOTEBOOK-BAILOUT warning, the magic path re-raises into the caller -- and a test
that asserted one of those would be pinning the environment, not the contract.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

pytest.importorskip("IPython")

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.user_ns["_ih"] = []
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
    shell = _MockShell()
    m = CashMagics(shell, cash)
    m._auto_cache_enabled = True
    yield m
    backend.clear()


def test_a_real_run_records_a_set_per_variable(magics):
    magics.cash("", "a = 1\nb = a + 1\n")
    recorded = magics._tracking_state.executed_cell_hashes
    assert recorded, "nothing was recorded; the run never reached the writer"
    wrong = {k: type(v).__name__ for k, v in recorded.items() if not isinstance(v, set)}
    assert not wrong, f"executed_cell_hashes holds non-sets: {wrong}"


def test_redefining_a_variable_accumulates_rather_than_replaces(magics):
    """The reason it is a set at all -- one variable, two defining statements."""
    magics.cash("", "a = 1\n")
    magics.cash("", "a = 2\n")
    assert len(magics._tracking_state.executed_cell_hashes["a"]) == 2


def test_the_declared_type_matches_what_is_stored():
    """Guards the annotation itself: it drifted once and nothing noticed."""
    import dataclasses

    from cash.notebook._protocols import TrackingState

    field = {f.name: f for f in dataclasses.fields(TrackingState)}["executed_cell_hashes"]
    assert field.type in ("dict[str, set[str]]", dict[str, set[str]]), field.type
