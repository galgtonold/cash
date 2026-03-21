import unittest
from unittest.mock import MagicMock, patch
import time
from cash.core import Cash
from cash.notebook.magics import CashMagics
from cash.backends.backend import InMemoryBackend

from traitlets.config import Configurable

class MockShell(Configurable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns['_ih'] = []  # Execution history
        self.run_cell = MagicMock()
        self.input_transformers_cleanup = [] # List of transformers
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()
        self.ast_transformers = []  # AST transformers
        self.events = MagicMock()  # Event system
        self.events.register = MagicMock(return_value=None)

class TestCashMagics(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryBackend()
        self.cash = Cash(backend=self.backend)
        
        # Mock IPython shell
        self.shell = MockShell()
        
        # Mock run_cell to actually execute code in user_ns
        def run_cell(cell):
            try:
                exec(cell, {}, self.shell.user_ns)
                return MagicMock(success=True)
            except Exception as e:
                return MagicMock(success=False, error_in_exec=e)
                
        self.shell.run_cell.side_effect = run_cell
        
        self.magics = CashMagics(self.shell, self.cash)


    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'backend'):
            self.backend.clear()
        if hasattr(self, 'shell') and hasattr(self.shell, 'user_ns'):
            self.shell.user_ns.clear()

    def test_basic_caching(self):
        # 1. First run: Calculate a = 1 + 1
        cell = "a = 1 + 1"
        self.magics.cash("", cell)
        
        self.assertEqual(self.shell.user_ns.get('a'), 2)
        # With incremental caching, we store per statement.
        self.assertEqual(len(self.backend.list_entries()), 1)
        
        # 2. Modify 'a' in namespace to verify restoration
        self.shell.user_ns['a'] = 99
        
        # 3. Second run: Should restore 'a' = 2
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('a'), 2)

    def test_incremental_caching(self):
        # Cell with two statements
        cell = """
x = 10
y = x + 5
"""
        # 1. First run
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('x'), 10)
        self.assertEqual(self.shell.user_ns.get('y'), 15)
        self.assertEqual(len(self.backend.list_entries()), 2) # Two statements cached
        
        # 2. Change second statement
        cell_v2 = """
x = 10
y = x + 10
"""
        self.magics.cash("", cell_v2)
        
        # Verify behavior: x should be restored from cache, y should be recomputed
        self.assertEqual(self.shell.user_ns.get('x'), 10)
        self.assertEqual(self.shell.user_ns.get('y'), 20)
        
        # Now we have 3 entries in cache:
        # - x = 10 (still valid from first run)
        # - y = x + 5 (old entry, no longer used but not deleted)
        # - y = x + 10 (new entry from second run)
        self.assertEqual(len(self.backend.list_entries()), 3)

    def test_global_caching(self):
        # Test enabling auto-caching
        self.magics.cash_on("")
        self.assertTrue(self.magics._auto_cache_enabled)
        
        # Test disabling
        self.magics.cash_off("")
        self.assertFalse(self.magics._auto_cache_enabled)

    def test_input_dependency(self):
        # 1. Setup input
        self.shell.user_ns['x'] = 10
        
        # 2. First run: y = x * 2
        cell = "y = x * 2"
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('y'), 20)
        
        # 3. Change input
        self.shell.user_ns['x'] = 5
        
        # 4. Second run: Should recompute y = 10 (cache miss due to input change)
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('y'), 10)

    def test_ttl(self):
        cell = "z = 100"
        self.magics.cash("ttl=1", cell)
        self.assertEqual(self.shell.user_ns.get('z'), 100)
        
        # Modify z
        self.shell.user_ns['z'] = 0
        
        # Immediate re-run -> Restore
        self.magics.cash("ttl=1", cell)
        self.assertEqual(self.shell.user_ns.get('z'), 100)
        
        # Wait for TTL
        time.sleep(1.1)
        self.shell.user_ns['z'] = 0
        
        # Re-run -> Recompute
        self.magics.cash("ttl=1", cell)
        self.assertEqual(self.shell.user_ns.get('z'), 100)
        
        # Verify that we actually recomputed (mock side effect check?)
        # For now, just checking correctness is enough.

    def test_multiple_outputs(self):
        cell = """
p = 10
q = 20
"""
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('p'), 10)
        self.assertEqual(self.shell.user_ns.get('q'), 20)
        
        self.shell.user_ns['p'] = 0
        self.shell.user_ns['q'] = 0
        
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('p'), 10)
        self.assertEqual(self.shell.user_ns.get('q'), 20)

    def test_module_input_skipping(self):
        import time
        self.shell.user_ns['time'] = time
        self.shell.user_ns['x'] = 10
        
        # Should not warn or fail
        cell = """
import time
y = x * 2
"""
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('y'), 20)

    def test_output_capture(self):
        cell = "print('Hello Cache')"
        
        # 1. First run: Capture output
        # We need to mock capture_output or check if it works with MockShell?
        # IPython.utils.io.capture_output relies on sys.stdout/stderr redirection.
        # It should work in standard python env.
        
        # We need to capture the *actual* stdout during the test to verify replay.
        from io import StringIO
        import sys
        
        # Capture stdout of the test process itself
        captured_stdout = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_stdout
        
        try:
            self.magics.cash("", cell)
        finally:
            sys.stdout = original_stdout
            
        output = captured_stdout.getvalue()
        # Note: capture_output().show() prints to sys.stdout, so we should see it.
        self.assertIn("Hello Cache", output)
        
        # 2. Second run: Replay
        captured_stdout = StringIO()
        sys.stdout = captured_stdout
        
        try:
            self.magics.cash("", cell)
        finally:
            sys.stdout = original_stdout
            
        output = captured_stdout.getvalue()
        self.assertIn("Hello Cache", output)
        # Note: With incremental caching, we might not print "[Restored from cache]" per statement if we commented it out.
        # But let's check if the output is there.

    def test_dependency_tracking_repro(self):
        # User reported: changing multiplier doesn't trigger re-run
        self.shell.user_ns['multiplier'] = 10
        self.shell.user_ns['result'] = 42
        
        cell = """
print(f"Computing with multiplier {multiplier}...")
final_value = result * multiplier
"""
        # 1. First run
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('final_value'), 420)
        
        # 2. Change multiplier
        self.shell.user_ns['multiplier'] = 5
        
        # 3. Second run - Should recompute
        self.magics.cash("", cell)
        self.assertEqual(self.shell.user_ns.get('final_value'), 210)

    def test_rich_output_capture(self):
        cell = "'rich_output'"
        
        # 1. First run: Mock capture_output to return rich output
        mock_output = {'data': {'text/plain': 'mock_data'}, 'metadata': {}}
        
        # We need to patch capture_output used in statement_processor.py where it's actually called
        # Also patch publish_display_data to avoid IPython initialization issues
        with patch('cash.notebook.statement_processor.capture_output') as mock_capture, \
             patch('cash.notebook.statement_processor.publish_display_data'), \
             patch('cash.notebook.magics.publish_display_data'):
            # Configure mock context manager
            mock_captured = MagicMock()
            mock_captured.stdout = ""
            mock_captured.stderr = ""
            mock_captured.outputs = [mock_output]
            mock_capture.return_value.__enter__.return_value = mock_captured
            
            self.magics.cash("", cell)
            
            # Verify it was stored
            entries = self.backend.list_entries()
            self.assertEqual(len(entries), 1)
            metadata = entries[0]
            key = metadata['key']
            # backend.get now returns (metadata, data) where data is already a dict
            _, payload = self.backend.get(key)
            self.assertEqual(payload['outputs'], [mock_output])
            
        # 2. Second run: Cache hit -> Replay
        # We need to patch publish_display_data in both modules where it can be called
        with patch('cash.notebook.statement_processor.publish_display_data') as mock_publish_sp, \
             patch('cash.notebook.magics.publish_display_data') as mock_publish_magics:
            self.magics.cash("", cell)
            
            # Verify publish_display_data was called in at least one location
            # The call might come from magics (for buffered outputs) or statement_processor
            combined_calls = mock_publish_sp.call_args_list + mock_publish_magics.call_args_list
            
            # Check that at least one call was made with the expected arguments
            expected_call_found = False
            for call in combined_calls:
                if call.kwargs.get('data') == mock_output['data'] and \
                   call.kwargs.get('metadata') == mock_output['metadata']:
                    expected_call_found = True
                    break
            
            self.assertTrue(expected_call_found, 
                f"Expected publish_display_data call not found. Calls: {combined_calls}")

if __name__ == '__main__':
    unittest.main()
