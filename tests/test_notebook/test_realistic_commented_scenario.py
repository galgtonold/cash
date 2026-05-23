"""
End-to-end test for the exact scenario user reported:
1. Execute cell with two DataFrame column assignments
2. Comment out first line
3. Execute again - verify first column is NOT present
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock IPython
sys.modules['IPython'] = MagicMock()
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

sys.modules['IPython.display'] = MagicMock()
sys.modules['IPython.utils.io'] = MagicMock()

# Mock capture_output
class MockCaptured:
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.outputs = []
    
    def show(self):
        pass

capture_output_mock = MagicMock()
capture_output_mock.return_value.__enter__ = MagicMock(return_value=MockCaptured())
capture_output_mock.return_value.__exit__ = MagicMock(return_value=False)
sys.modules['IPython.utils.io'].capture_output = capture_output_mock

from cash.notebook.magics import CashMagics
from cash.backends import InMemoryBackend
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


class TestRealisticCommentedCodeScenario(unittest.TestCase):
    """Test the EXACT scenario the user reported."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.cash = Cash(backend=self.backend, register_magic=False)
        
        self.shell = MockShell()
        
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        
        # Create a realistic DataFrame
        np.random.seed(42)
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100)
        })
    

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'backend'):
            self.backend.clear()
        if hasattr(self, 'shell') and hasattr(self.shell, 'user_ns'):
            self.shell.user_ns.clear()

    @pytest.mark.xfail(reason="Known failure: commented-out line cache invalidation")
    @patch('cash.notebook.magics.get_notebook_cells')
    def test_exact_user_scenario(self, mock_get_cells):
        """
        Test the EXACT scenario:
        1. Cell has: revenue line + month line
        2. Execute and cache
        3. Comment out revenue line
        4. Execute again
        5. Verify revenue column does NOT exist
        """
        # Original cell in notebook file (both lines)
        original_cell = (
            "df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        
        print("\n=== STEP 1: Execute original cell (both lines) ===")
        mock_get_cells.return_value = [original_cell]
        
        # Simulate executing the cell with %%cash magic
        # This is what happens when user runs the cell
        self.magics.cash("", original_cell)
        
        # Verify both columns exist
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)
        print(f"Columns after first execution: {list(self.shell.user_ns['df_clean'].columns)}")
        
        print("\n=== STEP 2: User comments out revenue line (but doesn't save) ===")
        # The cell being executed now has the comment
        commented_cell = (
            "#df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        
        # But the notebook file still has the old version
        # (This is the realistic scenario - user hasn't saved yet)
        mock_get_cells.return_value = [original_cell]  # Still the old version!
        
        # Clear the DataFrame to simulate fresh execution
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        print("\n=== STEP 3: Execute cell with comment ===")
        # Execute the commented cell
        self.magics.cash("", commented_cell)
        
        print(f"Columns after commented execution: {list(self.shell.user_ns['df_clean'].columns)}")
        
        # THE KEY ASSERTION: revenue should NOT be present
        self.assertNotIn('revenue', self.shell.user_ns['df_clean'].columns, 
                        "BUG: revenue column should not exist after commenting out the line!")
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)
    
    @pytest.mark.xfail(reason="Known failure: commented-out line cache invalidation")
    @patch('cash.notebook.magics.get_notebook_cells')
    def test_saved_notebook_with_commented_line(self, mock_get_cells):
        """
        Test when user HAS saved the notebook with the commented line.
        This might be causing the issue - fuzzy match with OLD uncommented cell?
        """
        # First, execute and cache the original (both lines)
        original_cell = (
            "df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        
        print("\n=== STEP 1: Execute original (both lines active, saved in notebook) ===")
        mock_get_cells.return_value = [original_cell]
        
        self.magics.cash("", original_cell)
        
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        print(f"Columns: {list(self.shell.user_ns['df_clean'].columns)}")
        
        # Now user comments out line and SAVES the notebook
        commented_cell = (
            "#df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        
        print("\n=== STEP 2: User saves notebook with commented line ===")
        # The notebook file NOW has commented version
        mock_get_cells.return_value = [commented_cell]
        
        # Fresh DataFrame
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        print("\n=== STEP 3: Execute cell (should fuzzy match with saved commented version) ===")
        self.magics.cash("", commented_cell)
        
        print(f"Columns: {list(self.shell.user_ns['df_clean'].columns)}")
        
        # Revenue should NOT be present
        self.assertNotIn('revenue', self.shell.user_ns['df_clean'].columns,
                        "BUG: Fuzzy match might be finding OLD cached version!")
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)

if __name__ == '__main__':
    unittest.main(verbosity=2)

