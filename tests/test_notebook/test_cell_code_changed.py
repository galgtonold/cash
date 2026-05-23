from cash.notebook.cache_status import CacheStatus
from cash.notebook.annotations import CacheAnnotation
"""
Tests for cache behavior when re-executing statements.

The ALREADY_EXECUTED skip optimization was removed because it was redundant
with cache lookups and caused correctness bugs (especially with self-assignments
like df['col'] = ...). Now all re-executed statements go through the normal
cache lookup path: COMPUTED on first run, RESTORED on subsequent identical runs.
"""
import pytest
from unittest.mock import MagicMock

from cash.notebook.magics import CashMagics
from cash.notebook.annotations import CacheAnnotation
from cash.core import Cash
from cash.backends import InMemoryBackend
from traitlets.config.configurable import Configurable

# Force caching regardless of the 10 ms min-execution-time floor.
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


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    processor = magics._statement_processor
    yield magics, shell, backend, processor
    backend.clear()
    shell.user_ns.clear()


class TestCacheRestoreBehavior:
    """Test that statements are properly cached and restored without ALREADY_EXECUTED optimization."""

    def test_first_run_computes(self, magics_fixture):
        """First execution of a statement should COMPUTE."""
        magics, shell, backend, processor = magics_fixture

        metrics = processor.process_statement("x = 42")
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns.get('x') == 42

    def test_second_run_restores_from_cache(self, magics_fixture):
        """Second identical execution should RESTORE from cache.
        _PERSIST overrides the 10 ms min-execution-time floor."""
        magics, shell, backend, processor = magics_fixture

        metrics1 = processor.process_statement("x = 42", annotation=_PERSIST)
        assert metrics1['status'] == CacheStatus.COMPUTED

        metrics2 = processor.process_statement("x = 42", annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.RESTORED
        assert shell.user_ns.get('x') == 42

    def test_multiple_reruns_always_restore(self, magics_fixture):
        """Multiple re-runs of same statement should always RESTORE.
        _PERSIST overrides the 10 ms min-execution-time floor."""
        magics, shell, backend, processor = magics_fixture

        processor.process_statement("x = 42", annotation=_PERSIST)
        for _ in range(3):
            metrics = processor.process_statement("x = 42", annotation=_PERSIST)
            assert metrics['status'] == CacheStatus.RESTORED

    def test_cache_still_stores_results(self, magics_fixture):
        """Results should be stored in cache and retrievable.
        _PERSIST overrides the 10 ms min-execution-time floor."""
        magics, shell, backend, processor = magics_fixture

        metrics = processor.process_statement("z = 99", annotation=_PERSIST)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns.get('z') == 99

        # Re-run should restore from cache
        metrics2 = processor.process_statement("z = 99", annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.RESTORED

    def test_mutation_pattern_executes(self, magics_fixture):
        """Mutation calls should be detected and handled properly."""
        magics, shell, backend, processor = magics_fixture

        processor.process_statement("items = []")
        assert shell.user_ns.get('items') == []

        # Mutation: append modifies items in-place
        processor.process_statement("items.append(1)")
        assert shell.user_ns.get('items') == [1]

        # Re-running the mutation after resetting should work
        shell.user_ns['items'] = []
        processor.process_statement("items.append(1)")
        assert shell.user_ns.get('items') == [1]
