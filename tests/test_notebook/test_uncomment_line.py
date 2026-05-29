"""
Test for uncommenting lines scenario - variables should reflect current session state
"""
import unittest
from unittest.mock import MagicMock
import sys
import os
import pandas as pd
import numpy as np
import types

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock IPython
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

from cash.notebook.ipython.magics import CashMagics
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



class TestUncommentLine(unittest.TestCase):
    """Test uncommenting a line and using the modified variable in next cell."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.cash = Cash(backend=self.backend, register_magic=False)
        
        self.shell = MockShell()
        
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        self.magics._auto_cache_enabled = True
        
        # Create initial DataFrame
        np.random.seed(42)
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
        
        self.shell.user_ns['selected_region'] = 'South'
    

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'backend'):
            self.backend.clear()
        if hasattr(self, 'shell') and hasattr(self.shell, 'user_ns'):
            self.shell.user_ns.clear()

    def test_uncomment_line_and_use_result(self):
        """
        Test uncommenting revenue line and then using revenue column.
        This simulates:
        1. Initially revenue line is commented
        2. User uncomments it and runs the cell
        3. Next cell tries to use revenue column - should work!
        """
        print("\n=== TEST: Uncomment Line and Use Result ===")
        
        # Step 1: Run cell WITH revenue line (uncommented)
        print("\n--- Step 1: Execute revenue calculation ---")
        cell1 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"
        
        self.magics._execute_cell(cell1)
        
        # Verify revenue column exists
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        print(f"Columns after step 1: {list(self.shell.user_ns['df_clean'].columns)}")
        
        # Step 2: Try to use revenue column in next cell
        print("\n--- Step 2: Use revenue column ---")
        cell2 = "summary = df_clean[df_clean['region'] == selected_region].groupby('product')['revenue'].sum()"
        
        self.magics._execute_cell(cell2)
        
        # Verify summary was created successfully
        self.assertIn('summary', self.shell.user_ns)
        self.assertIsNotNone(self.shell.user_ns['summary'])
        print(f"Summary created: {type(self.shell.user_ns['summary'])}")
        print("✓ Test passed: Revenue column accessible in next cell!")
    
    def test_comment_then_uncomment(self):
        """
        Test commenting out a line, then uncommenting it again.
        """
        print("\n=== TEST: Comment Then Uncomment ===")
        
        # Step 1: Run with revenue line
        print("\n--- Step 1: Execute with revenue ---")
        cell1 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"
        self.magics._execute_cell(cell1)
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        
        # Step 2: Comment out revenue line (reset df)
        print("\n--- Step 2: Comment out revenue (reset df) ---")
        # Simulate fresh df without revenue
        np.random.seed(42)
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
        self.assertNotIn('revenue', self.shell.user_ns['df_clean'].columns)


        # Step 3: Uncomment revenue line again
        print("\n--- Step 3: Uncomment revenue line ---")
        cell3 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"
        self.magics._execute_cell(cell3)
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        
        # Step 4: Use revenue in next cell
        print("\n--- Step 4: Use revenue column ---")
        cell4 = "summary = df_clean['revenue'].sum()"
        self.magics._execute_cell(cell4)
        
        self.assertIn('summary', self.shell.user_ns)
        self.assertIsInstance(self.shell.user_ns['summary'], (int, float, np.number))
        print(f"Summary value: {self.shell.user_ns['summary']}")
        print("✓ Test passed: Revenue accessible after uncommenting!")

if __name__ == '__main__':
    unittest.main(verbosity=2)

