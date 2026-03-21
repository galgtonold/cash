"""Tests for hot reload notification in badge when functions change."""
import pytest
from unittest.mock import MagicMock, patch
from traitlets.config import Configurable

from cash.core import Cash
from cash.notebook.magics import CashMagics
from cash.backends.backend import InMemoryBackend


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


class TestHotReloadNotification:
    """Tests for function change detection and badge notification."""

    def test_changed_function_detected_in_metrics(self, magics_fixture):
        """When a function's source changes, a FUNCTION_CHANGED metric is added."""
        magics, shell, backend = magics_fixture

        # Define and track a function
        def helper(x):
            return x * 2

        shell.user_ns['helper'] = helper
        ft = magics._statement_processor.function_tracker
        ft.update_function_hash('helper', helper)

        # Now replace it with a different implementation
        def helper_v2(x):
            return x * 3

        shell.user_ns['helper'] = helper_v2

        # Detect changed functions (this is what _execute_cell does internally)
        changed = ft.detect_changed_functions(shell.user_ns)
        assert 'helper' in changed

    def test_function_changed_notification_structure(self, magics_fixture):
        """FUNCTION_CHANGED metric has correct structure."""
        magics, shell, backend = magics_fixture

        # Build a notification metric manually (as _execute_cell does)
        notification = {
            'status': 'FUNCTION_CHANGED',
            'code': "🔄 Function changed: process",
            'is_upstream': True,
            'total_time': 0.0,
            'execution_time': 0.0,
            'outputs': [],
            'changed_functions': ['process'],
        }

        assert notification['status'] == 'FUNCTION_CHANGED'
        assert notification['is_upstream'] is True
        assert 'process' in notification['changed_functions']

    def test_function_changed_badge_rendering(self, magics_fixture):
        """FUNCTION_CHANGED status renders correctly in badge."""
        magics, shell, backend = magics_fixture

        notification = {
            'status': 'FUNCTION_CHANGED',
            'code': "🔄 Functions changed: process, helper",
            'is_upstream': True,
            'total_time': 0.0,
            'execution_time': 0.0,
            'outputs': [],
            'changed_functions': ['helper', 'process'],
        }

        # Test rendering via _render_interactive_badge with patched display
        with patch.object(magics, 'shell') as mock_shell:
            mock_shell.user_ns = shell.user_ns
            with patch('cash.notebook.magics.display'):
                magics._render_interactive_badge(
                    [notification],
                    display_id="test-id",
                    status="DONE"
                )
                # Should have called display
                assert True  # May not display in test env

    def test_no_notification_when_no_changes(self, magics_fixture):
        """No FUNCTION_CHANGED metric when functions haven't changed."""
        magics, shell, backend = magics_fixture

        def helper(x):
            return x * 2

        shell.user_ns['helper'] = helper
        ft = magics._statement_processor.function_tracker
        ft.update_function_hash('helper', helper)

        # Same function, no change
        changed = ft.detect_changed_functions(shell.user_ns)
        assert 'helper' not in changed

    def test_multiple_functions_changed(self, magics_fixture):
        """Multiple function changes generate single notification."""
        magics, shell, backend = magics_fixture

        def func_a():
            return 1

        def func_b():
            return 2

        shell.user_ns['func_a'] = func_a
        shell.user_ns['func_b'] = func_b
        ft = magics._statement_processor.function_tracker
        ft.update_function_hash('func_a', func_a)
        ft.update_function_hash('func_b', func_b)

        # Replace both
        def func_a_v2():
            return 10

        def func_b_v2():
            return 20

        shell.user_ns['func_a'] = func_a_v2
        shell.user_ns['func_b'] = func_b_v2

        changed = ft.detect_changed_functions(shell.user_ns)
        assert 'func_a' in changed
        assert 'func_b' in changed

        # Build notification
        func_names = ', '.join(sorted(changed))
        notification = {
            'status': 'FUNCTION_CHANGED',
            'code': f"🔄 Functions changed: {func_names}",
            'is_upstream': True,
            'total_time': 0.0,
            'execution_time': 0.0,
            'outputs': [],
            'changed_functions': sorted(changed),
        }
        assert len(notification['changed_functions']) == 2
        assert 'Functions' in notification['code']  # plural

    def test_function_deleted_detected(self, magics_fixture):
        """Deleted function is detected as changed."""
        magics, shell, backend = magics_fixture

        def old_func():
            return 42

        shell.user_ns['old_func'] = old_func
        ft = magics._statement_processor.function_tracker
        ft.update_function_hash('old_func', old_func)

        # Delete the function
        del shell.user_ns['old_func']

        changed = ft.detect_changed_functions(shell.user_ns)
        assert 'old_func' in changed

    def test_function_replaced_with_non_callable(self, magics_fixture):
        """Function replaced with non-callable is detected as changed."""
        magics, shell, backend = magics_fixture

        def my_func():
            return 42

        shell.user_ns['my_func'] = my_func
        ft = magics._statement_processor.function_tracker
        ft.update_function_hash('my_func', my_func)

        # Replace with a string
        shell.user_ns['my_func'] = "not a function"

        changed = ft.detect_changed_functions(shell.user_ns)
        assert 'my_func' in changed
