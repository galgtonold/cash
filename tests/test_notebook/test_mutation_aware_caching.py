from cash.notebook.cache_status import CacheStatus
"""
Tests for mutation-aware caching in statement_processor.

When a statement mutates external variables in-place (e.g., lst.append(x)),
the cache should be skipped because the mutated state can't be captured
by lineage alone.
"""

import pytest
from unittest.mock import MagicMock
from cash.notebook.magics import CashMagics
from cash.notebook.cacheability import analyze_statement
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable


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
def processor_fixture():
    """Provide StatementProcessor via CashMagics for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    processor = magics._statement_processor
    yield processor, shell, backend
    backend.clear()
    shell.user_ns.clear()


# --- Tests for analyze_statement(...).top_level_mutated_vars ---

def _top_level(code: str) -> set:
    return set(analyze_statement(code, None).top_level_mutated_vars)


class TestTopLevelMutationDetection:
    """Tests for the top-level-only mutation detection."""

    def test_class_body_mutations_excluded(self):
        """Mutations inside class bodies (self.x = y) should NOT be detected."""
        code = """
class Foo:
    def __init__(self, val):
        self.val = val
    def add(self, x):
        self.val += x
"""
        result = _top_level(code)
        assert result == set(), f"Expected no top-level mutations, got {result}"

    def test_function_body_mutations_excluded(self):
        """Mutations inside function bodies should NOT be detected."""
        code = """
def process(items):
    items.append('done')
    result = []
    result.extend(items)
    return result
"""
        result = _top_level(code)
        assert result == set(), f"Expected no top-level mutations, got {result}"

    def test_top_level_append_detected(self):
        """Top-level list.append() should be detected."""
        code = "data.append(42)"
        result = _top_level(code)
        assert result == {'data'}

    def test_top_level_augmented_assign_detected(self):
        """Top-level augmented assignment (x += 1) should be detected."""
        code = "counter += 1"
        result = _top_level(code)
        assert result == {'counter'}

    def test_top_level_subscript_assign_detected(self):
        """Top-level dict subscript assignment should be detected."""
        code = "config['key'] = 'value'"
        result = _top_level(code)
        assert result == {'config'}

    def test_mixed_top_level_and_class(self):
        """Only top-level mutations detected, not class body ones."""
        code = """
class MyClass:
    def __init__(self):
        self.data = []
    def add(self, item):
        self.data.append(item)

obj = MyClass()
results.append(obj)
"""
        result = _top_level(code)
        assert result == {'results'}

    def test_async_function_body_excluded(self):
        """Mutations inside async function bodies should NOT be detected."""
        code = """
async def update(state):
    state['status'] = 'done'
    state.update({'ts': 123})
"""
        result = _top_level(code)
        assert result == set()

    def test_empty_code(self):
        """Empty code returns empty set."""
        assert _top_level("") == set()

    def test_syntax_error_returns_empty(self):
        """Syntax errors return empty set gracefully."""
        assert _top_level("def (") == set()

    def test_pure_assignment_no_mutation(self):
        """Simple assignments are not mutations."""
        code = "x = 10\ny = x + 5"
        result = _top_level(code)
        assert result == set()


# --- Tests for mutation-aware caching in StatementProcessor ---

class TestMutationAwareCaching:
    """Tests that mutation-aware caching works correctly in statement processing."""

    def test_pure_mutation_skips_cache(self, processor_fixture):
        """Statement that only mutates an existing variable should skip cache."""
        processor, shell, backend = processor_fixture

        # Set up existing list in namespace
        shell.user_ns['data'] = [1, 2, 3]

        code = "data.append(4)"
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED

        # Run again — should still be COMPUTED (not RESTORED) because it's uncacheable
        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.COMPUTED
        # Check that uncacheable reason is recorded
        assert any('mutation' in r.lower() or 'In-place' in r for r in metrics2.get('uncacheable_reasons', []))

    def test_class_definition_still_cacheable(self, processor_fixture):
        """Class definitions with internal self.x=y should still be cacheable."""
        processor, shell, backend = processor_fixture

        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED

        # Clear and re-run — should be RESTORED from cache
        shell.user_ns.pop('Point', None)
        shell.user_ns.pop('p', None)

        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.RESTORED

    def test_augmented_assign_as_output(self, processor_fixture):
        """Augmented assignment on output variable (x += 1 where x is output) is cacheable."""
        processor, shell, backend = processor_fixture

        code = "x = 10\nx += 5"
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        assert shell.user_ns.get('x') == 15

    def test_dict_update_mutation_detected(self, processor_fixture):
        """Dict update mutation on non-output var should skip cache."""
        processor, shell, backend = processor_fixture

        shell.user_ns['config'] = {'a': 1}
        code = "config.update({'b': 2})"

        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED

        # Run again — should be COMPUTED, not RESTORED
        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.COMPUTED

    def test_function_def_with_mutation_cacheable(self, processor_fixture):
        """Function that internally mutates args should still be cacheable as a definition."""
        processor, shell, backend = processor_fixture

        code = """
def transform(lst):
    lst.sort()
    return lst[0]
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED

        shell.user_ns.pop('transform', None)
        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.RESTORED
