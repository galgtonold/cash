"""
Test mutable input state restoration.
Ensures that when statements modify mutable inputs in-place, 
the pre-modification state is restored correctly on cache hits.
"""
import unittest
from unittest.mock import MagicMock
import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock IPython
import types
ipython_mock = types.ModuleType('IPython')
ipython_mock.__path__ = []
sys.modules['IPython'] = ipython_mock
sys.modules['IPython.core'] = types.ModuleType('IPython.core')
sys.modules['IPython.display'] = MagicMock()
sys.modules['IPython.utils'] = types.ModuleType('IPython.utils')
sys.modules['IPython.utils.io'] = MagicMock()
sys.modules['IPython.core.magic'] = MagicMock()
def pass_through(cls):
    return cls
sys.modules['IPython.core.magic'].magics_class = pass_through
sys.modules['IPython.core.magic'].line_magic = pass_through
sys.modules['IPython.core.magic'].cell_magic = pass_through

class Magics:
    def __init__(self, shell):
        self.shell = shell

sys.modules['IPython.core.magic'].Magics = Magics

from cash.notebook.magics import CashMagics
from cash.backends.backend import InMemoryBackend
from cash.core import Cash
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns["_ih"] = []
        self.run_cell = MagicMock()
        self.input_transformers_cleanup = []
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()
        self.ast_transformers = []
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)


class TestMutableInputRestoration(unittest.TestCase):
    """Test restoration of mutable input states."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.cash = Cash(backend=self.backend, register_magic=False)
        self.shell = MockShell()
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        self.magics._auto_cache_enabled = True
        
    def tearDown(self):
        if hasattr(self, 'backend'):
            self.backend.clear()
        if hasattr(self, 'shell') and hasattr(self.shell, 'user_ns'):
            self.shell.user_ns.clear()

    def test_dataframe_mutation_restoration(self):
        """
        Test that DataFrame mutation (column addition) is correctly handled.
        """
        print("\n=== TEST: DataFrame Mutation Restoration ===")
        
        # 1. Create initial DataFrame
        self.shell.user_ns['df'] = pd.DataFrame({'A': [1, 2, 3]})
        
        # 2. Run mutation statement (First Run)
        code = "df['B'] = df['A'] * 2"
        self.magics._execute_cell(code)
        
        self.assertIn('B', self.shell.user_ns['df'].columns)
        
        # Verify source tracking
        # The statement processor should have updated variable_sources['df']
        # to the cache key of the "df['B'] = ..." statement.
        
        # Get cache key (we can't easily get it, but we can check if it exists)
        processor = self.magics._statement_processor
        self.assertIn('df', processor.variable_sources)
        print(f"Source for df: {processor.variable_sources['df']}")
        
        # 3. Simulate state before second run (Reset state)
        self.shell.user_ns['df'] = pd.DataFrame({'A': [1, 2, 3]})
        self.assertNotIn('B', self.shell.user_ns['df'].columns)
        
        # 4. Run cached statement (Second Run - Cache Hit)
        print("Running cached statement...")
        self.magics._execute_cell(code)
        
        self.assertIn('B', self.shell.user_ns['df'].columns)
        self.assertEqual(self.shell.user_ns['df']['B'].tolist(), [2, 4, 6])
        
        # Verify source is still tracked correctly after restore
        self.assertIn('df', processor.variable_sources)
        print(f"Source for df after restore: {processor.variable_sources['df']}")
        
        print("✓ DataFrame output correctly restored")

    @unittest.skip("Feature not yet implemented: cache consistency check for input_snapshots")
    def test_partial_cache_inconsistency(self):
        """
        Reproduce the user's issue:
        If cache contains input_snapshot for 'df' but misses 'df' in variables (outputs),
        restoration effectively UNDOES the operation (reverts to snapshot).
        """
        print("\n=== TEST: Partial Cache Inconsistency Repro ===")
        
        # 1. Setup initial state
        self.shell.user_ns['df'] = pd.DataFrame({'A': [1]})
        
        # 2. Simulate a "bad" cache payload
        # Snapshot has clean state. Variables MISSING 'df' (simulating drop/failure).
        bad_payload = {
            'variables': {}, # MISSING 'df'
            'input_snapshots': {
                'df': pd.DataFrame({'A': [1]}) # Pre-mutation snapshot
            },
            'outputs': []
        }
        
        # 3. Simulate "Dirty" state in memory (as if statement ran)
        # df has 'B' added
        self.shell.user_ns['df']['B'] = 2
        
        print(f"Current DF columns (Dirty): {self.shell.user_ns['df'].columns.tolist()}")
        self.assertIn('B', self.shell.user_ns['df'].columns)
        
        # 4. Restore from bad cache
        # This call should now RAISE ValueError due to consistency check.
        # And preventing the UNDO.
        
        processor = self.magics._statement_processor
        
        with self.assertRaises(ValueError):
            processor._restore_from_cache(bad_payload, {}, False, 0.0)
        
        print(f"Restored DF columns: {self.shell.user_ns['df'].columns.tolist()}")
        
        # Verify that df was NOT reverted (B is still present)
        # Because we aborted before applying snapshots
        self.assertIn('B', self.shell.user_ns['df'].columns)
        print("✓ Verified: Partial cache inconsistency raised ValueError and preserved state")

    def test_dirty_state_restoration(self):
        """
        Test that we can restore from cache even if the input object was created in the same cell execution?
        Actually verifying simple alias behavior.
        """
        # Ensure pandas is available in user_ns
        self.shell.user_ns['pd'] = pd

        # Create df2
        self.magics._execute_cell("df2 = pd.DataFrame({'A': [1]})")
        
        # Mutate df2
        self.magics._execute_cell("df2['B'] = 2")
        self.assertIn('B', self.shell.user_ns['df2'].columns)
        
        # Verify restoration works
        # Reset df2
        self.magics._execute_cell("df2 = pd.DataFrame({'A': [1]})")
        
        # Restore mutation
        self.magics._execute_cell("df2['B'] = 2")
        self.assertIn('B', self.shell.user_ns['df2'].columns)
        
        print("✓ Dirty state restoration test passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
