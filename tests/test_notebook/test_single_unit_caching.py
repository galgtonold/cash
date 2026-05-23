from cash.notebook.cache_status import CacheStatus
"""Tests for control structure caching behaviour.

For-loops are decomposed per-iteration.  Each body statement goes through
the statement processor individually.  Statements that mutate external
variables (detected by MutationDetector) are executed directly (skip_cache),
while pure computation statements are cached per-iteration.

If/while/with/try are still processed as single cacheable units.
"""
import pytest
from unittest.mock import MagicMock
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.magics import CashMagics
from cash.backends import InMemoryBackend


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
def magics_fixture():
    """Provide CashMagics instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


class TestForLoopPerIteration:
    """Tests that for-loop body statements are processed per-iteration."""

    def test_first_run_computes(self, magics_fixture):
        """First run should compute all iterations."""
        magics, shell, backend = magics_fixture

        code = """
total = 0
for i in [1, 2, 3, 4, 5]:
    total += i
"""
        magics.cash("", code.strip())
        assert shell.user_ns['total'] == 15

    def test_second_run_correct(self, magics_fixture):
        """Second run should produce correct results."""
        magics, shell, backend = magics_fixture

        code = """
total = 0
for i in [1, 2, 3, 4, 5]:
    total += i
"""
        magics.cash("", code.strip())
        assert shell.user_ns['total'] == 15

        magics.cash("", code.strip())
        assert shell.user_ns['total'] == 15

    def test_loop_with_append_correct(self, magics_fixture):
        """Loop with .append() mutation should produce correct results on both runs."""
        magics, shell, backend = magics_fixture

        code = """
results = []
for x in ['a', 'b', 'c', 'd']:
    results.append(x.upper())
"""
        magics.cash("", code.strip())
        assert shell.user_ns['results'] == ['A', 'B', 'C', 'D']

        magics.cash("", code.strip())
        assert shell.user_ns['results'] == ['A', 'B', 'C', 'D']

    def test_pure_body_statements_cached(self, magics_fixture):
        """Pure computations inside loops should be cached per-iteration.
        sum(range(500_000)) keeps each body statement above the 10 ms
        min-execution-time floor so the iterations are actually stored in cache."""
        magics, shell, backend = magics_fixture

        code = """
for i in [1, 2, 3]:
    x = sum(range(5_000_000)) * 0 + i * 100
"""
        # First run: compute
        magics.cash("", code.strip())
        assert shell.user_ns['x'] == 300  # Last iteration value

        # Second run: should restore from cache
        magics.cash("", code.strip())
        assert shell.user_ns['x'] == 300

        # Check that iterations were cached (look at metrics)
        statements = magics._last_cell_metrics.get('statements', [])
        restored = [s for s in statements if s.get('status') == CacheStatus.RESTORED]
        assert len(restored) > 0, "Expected some statements to be RESTORED from cache"

    def test_mutation_statements_not_cached(self, magics_fixture):
        """Statements with in-place mutations should not be cached (skip_cache=True)."""
        magics, shell, backend = magics_fixture

        code = """
acc = []
for v in [10, 20]:
    acc.append(v)
"""
        magics.cash("", code.strip())
        assert shell.user_ns['acc'] == [10, 20]

        # On second run, mutation statements re-execute
        magics.cash("", code.strip())
        assert shell.user_ns['acc'] == [10, 20]

    def test_mixed_cache_and_mutation(self, magics_fixture):
        """Expensive pure statements cached, mutation statements re-executed."""
        magics, shell, backend = magics_fixture

        code = """
results = {}
for key in ['x', 'y']:
    val = sum(range(100))
    results[key] = val
"""
        magics.cash("", code.strip())
        assert shell.user_ns['results'] == {'x': 4950, 'y': 4950}

        # Second run: val = sum(range(100)) should restore from cache
        # results[key] = val is a subscript assignment on results, but results IS output
        magics.cash("", code.strip())
        assert shell.user_ns['results'] == {'x': 4950, 'y': 4950}


class TestIfSingleUnitCaching:
    """Tests that if statements are cached as single units."""

    def test_if_true_branch(self, magics_fixture):
        magics, shell, backend = magics_fixture
        code = """
x = 10
if x > 5:
    result = 'big'
else:
    result = 'small'
"""
        magics.cash("", code.strip())
        assert shell.user_ns['result'] == 'big'

    def test_if_false_branch(self, magics_fixture):
        magics, shell, backend = magics_fixture
        code = """
x = 3
if x > 5:
    result = 'big'
else:
    result = 'small'
"""
        magics.cash("", code.strip())
        assert shell.user_ns['result'] == 'small'


class TestWhileSingleUnitCaching:
    """Tests that while loops are cached as single units."""

    def test_while_loop(self, magics_fixture):
        magics, shell, backend = magics_fixture
        code = """
counter = 0
i = 0
while i < 5:
    counter += 1
    i += 1
"""
        magics.cash("", code.strip())
        assert shell.user_ns['counter'] == 5
        assert shell.user_ns['i'] == 5


class TestBadgeDisplayWithPerIteration:
    """Tests for badge display with per-iteration for-loop metrics."""

    def test_loop_iteration_metrics_have_context(self, magics_fixture):
        """For-loop body statement metrics should have __iteration_context__ in code."""
        magics, shell, backend = magics_fixture

        code = """
for key in ['a', 'b']:
    val = len(key)
"""
        magics.cash("", code.strip())

        statements = magics._last_cell_metrics.get('statements', [])
        loop_stmts = [s for s in statements if '# __iteration_context__:' in s.get('code', '')]
        assert len(loop_stmts) > 0, "Expected per-iteration metrics with __iteration_context__"

    def test_if_no_iteration_context(self, magics_fixture):
        """If-statement metrics should NOT have __iteration_context__."""
        magics, shell, backend = magics_fixture

        code = """
x = 5
if x > 0:
    y = 1
else:
    y = -1
"""
        magics.cash("", code.strip())

        statements = magics._last_cell_metrics.get('statements', [])
        for s in statements:
            assert '# __iteration_context__:' not in s.get('code', '')

class TestErrorPropagationInLoops:
    """Tests that runtime errors inside loops propagate correctly and are NOT masked by cache."""

    def test_error_in_body_statement_propagates(self, magics_fixture):
        """A runtime error in a body statement should crash the cell, not be masked by cache."""
        magics, shell, backend = magics_fixture

        # This code has a bug: stats['mean'] = 1 overwrites the list with int,
        # then stats['mean'][0] tries to subscript an int → TypeError.
        # Cash should NOT mask this error on ANY iteration.
        code = """
for x in ['a', 'b', 'c']:
    stats = {'mean': [1.0, 2.0, 3.0]}
    stats['mean'] = 1
    result = stats['mean'][0]
"""
        with pytest.raises(TypeError, match="not subscriptable"):
            magics.cash("", code.strip())

    def test_error_on_first_iteration_not_masked(self, magics_fixture):
        """Error must occur on the very first iteration, not be silently cached."""
        magics, shell, backend = magics_fixture

        code = """
for x in ['a']:
    d = {'key': [10]}
    d['key'] = 42
    val = d['key'][0]
"""
        with pytest.raises(TypeError):
            magics.cash("", code.strip())

    def test_error_after_code_change_not_masked(self, magics_fixture):
        """If code changes to introduce a bug, the error should NOT be masked by old cache."""
        magics, shell, backend = magics_fixture

        # Run 1: correct code — no error, caches successfully
        code_v1 = """
for x in ['a', 'b']:
    stats = {'mean': [1.0, 2.0]}
    val = stats['mean'][0]
"""
        magics.cash("", code_v1.strip())
        assert shell.user_ns['val'] == 1.0

        # Run 2: modified code introduces a bug (overwrite list with int)
        code_v2 = """
for x in ['a', 'b']:
    stats = {'mean': [1.0, 2.0]}
    stats['mean'] = 1
    val = stats['mean'][0]
"""
        with pytest.raises(TypeError, match="not subscriptable"):
            magics.cash("", code_v2.strip())


class TestErrorLineNumberAnnotation:
    """Tests that errors from control structures have correct line number annotations."""

    def test_for_loop_error_points_to_body_statement(self, magics_fixture):
        """_cash_error_lineno should point to the body statement, not the for-loop header."""
        magics, shell, backend = magics_fixture

        # Line 1: for x in ['a']:
        # Line 2:     d = {'key': [10]}
        # Line 3:     d['key'] = 42
        # Line 4:     val = d['key'][0]   <-- error here
        code = """for x in ['a']:
    d = {'key': [10]}
    d['key'] = 42
    val = d['key'][0]"""

        try:
            magics.cash("", code.strip())
            pytest.fail("Expected TypeError")
        except TypeError as e:
            # The error should be annotated with line 4 (the body statement),
            # not line 1 (the for-loop header)
            assert hasattr(e, '_cash_error_lineno'), \
                "_cash_error_lineno not set on exception"
            assert e._cash_error_lineno == 4, \
                f"Expected error at line 4, got line {e._cash_error_lineno}"

    def test_if_single_unit_error_points_to_body_line(self, magics_fixture):
        """For single-unit if statements, error lineno should point inside the body."""
        magics, shell, backend = magics_fixture

        # Line 1: x = 5
        # Line 2: if x > 0:
        # Line 3:     val = 1 / 0   <-- error at line 2 of unparsed if block
        code = """x = 5
if x > 0:
    val = 1 / 0"""

        try:
            magics.cash("", code.strip())
            pytest.fail("Expected ZeroDivisionError")
        except ZeroDivisionError as e:
            # The if-block starts at line 2 in the cell.  The error is on
            # the 2nd line of the unparsed block → cell line 2 + 2 - 1 = 3
            assert hasattr(e, '_cash_error_lineno'), \
                "_cash_error_lineno not set on exception"
            assert e._cash_error_lineno == 3, \
                f"Expected error at line 3, got line {e._cash_error_lineno}"

    def test_nested_for_loop_error_preserves_inner_lineno(self, magics_fixture):
        """Nested control structure errors should preserve the innermost error line."""
        magics, shell, backend = magics_fixture

        # Line 1: for x in [1]:
        # Line 2:     for y in [1]:
        # Line 3:         val = 1 / 0   <-- error here
        code = """for x in [1]:
    for y in [1]:
        val = 1 / 0"""

        try:
            magics.cash("", code.strip())
            pytest.fail("Expected ZeroDivisionError")
        except ZeroDivisionError as e:
            assert hasattr(e, '_cash_error_lineno'), \
                "_cash_error_lineno not set on exception"
            assert e._cash_error_lineno == 3, \
                f"Expected error at line 3, got line {e._cash_error_lineno}"


class TestShowCleanErrorIntegration:
    """Tests that _show_clean_error uses _cash_error_lineno for correct tracebacks."""

    def test_execute_cell_error_shows_correct_line(self, magics_fixture):
        """_execute_cell should call _show_clean_error with correct line number."""
        magics, shell, backend = magics_fixture

        # Add execution_count attribute for _show_clean_error
        shell.execution_count = 1

        # Track what _show_clean_error receives
        captured_calls = []

        def patched_show(exc, raw_cell, node):
            # Check _cash_error_lineno before _show_clean_error uses it
            lineno = getattr(exc, '_cash_error_lineno', None) or getattr(node, 'lineno', None)
            captured_calls.append({'lineno': lineno, 'exc_type': type(exc).__name__})
            # Don't actually call the real one (it would try to use showtraceback)
            return

        magics._show_clean_error = patched_show
        magics._auto_cache_enabled = True

        # Cell code: for loop with error on line 4
        raw_cell = "for x in ['a']:\n    d = {'key': [10]}\n    d['key'] = 42\n    val = d['key'][0]"
        magics._original_run_cell = MagicMock()

        magics._execute_cell(raw_cell, store_history=True)

        assert len(captured_calls) == 1, f"Expected 1 _show_clean_error call, got {len(captured_calls)}"
        assert captured_calls[0]['lineno'] == 4, \
            f"Expected error line 4, got {captured_calls[0]['lineno']}"
        assert captured_calls[0]['exc_type'] == 'TypeError'


class TestUpstreamFunctionTrackerSharing:
    """Verify that function_tracker is shared between StatementProcessor and UpstreamChecker.

    This prevents false lineage mismatches during upstream simulation for
    statements that use user-defined functions (e.g., df['RSI'] = ...(calculate_rsi)).
    Without function_tracker, simulation computes different cache keys than runtime,
    causing forward propagation to fail and virtual lineage to diverge.
    """

    def test_function_tracker_shared(self, magics_fixture):
        """UpstreamChecker should have same function_tracker as StatementProcessor."""
        magics, shell, backend = magics_fixture

        sp_ft = magics._statement_processor.function_tracker
        uc_ft = magics._upstream_checker.function_tracker

        assert sp_ft is uc_ft, \
            "function_tracker must be the same object instance on both components"

    def test_function_tracker_not_none(self, magics_fixture):
        """UpstreamChecker's function_tracker should not be None."""
        magics, shell, backend = magics_fixture

        assert magics._upstream_checker.function_tracker is not None, \
            "UpstreamChecker.function_tracker should not be None"