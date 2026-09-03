"""The metrics dict carries a display-only copy of the statement's source."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics

pytest.importorskip("IPython")

# Force caching regardless of the 10 ms min-execution-time floor -- same
# convention as test_already_executed_optimization.py.
_PERSIST = CacheAnnotation(persist=True)


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def processor_fixture():
    """Provide StatementProcessor instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    processor = magics._statement_processor
    yield processor, shell, backend, magics
    backend.clear()
    shell.user_ns.clear()


def test_display_code_is_recorded_when_supplied(processor_fixture):
    processor, shell, backend, magics = processor_fixture
    shell.user_ns["a"] = 1

    metrics = processor.process_statement(
        "x = a + 1", display_code="x = (\n    a + 1\n)",
    )

    assert metrics["display_code"] == "x = (\n    a + 1\n)"
    assert metrics["code"] == "x = a + 1", "the keyed text must not change"


def test_display_code_defaults_to_none(processor_fixture):
    """An older caller that does not pass it must behave exactly as before."""
    processor, shell, backend, magics = processor_fixture
    shell.user_ns["a"] = 1

    metrics = processor.process_statement("x = a + 1")

    assert metrics.get("display_code") is None


def test_display_code_does_not_change_the_cache_key(processor_fixture):
    """The whole point: it is display-only.

    Same code, different display text -- the second call must still hit.

    Two fixture gates have nothing to do with display_code but block a hit
    on their own, so both must be cleared for this test to mean anything:
    a raw ``shell.user_ns[...] =`` write (as used by the other two tests in
    this file, and by the sibling fixture files) leaves the variable without
    cash-tracked lineage, which alone makes ANY statement reading it
    uncacheable -- so 'a' is established via process_statement instead.
    And ``x = a + 1`` runs in well under a millisecond, under the 10ms
    "too cheap to cache" floor -- so both calls force persistence.
    """
    processor, shell, backend, magics = processor_fixture
    processor.process_statement("a = 1")

    first = processor.process_statement("x = a + 1", display_code="x = a + 1", annotation=_PERSIST)
    second = processor.process_statement(
        "x = a + 1", display_code="x = (\n    a + 1\n)", annotation=_PERSIST,
    )

    assert first["status"] == CacheStatus.COMPUTED
    assert second["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED), (
        f"display text changed the key: second call was {second['status']}"
    )
