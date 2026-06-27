"""
End-to-end test for the exact scenario user reported:
1. Execute cell with two DataFrame column assignments
2. Comment out first line
3. Execute again - verify first column is NOT present

The commented-out subscript assignment (``df_clean['revenue'] = ...``) must not
reappear via a cache restore: re-running the cell processes only the surviving
``df_clean['month'] = ...`` statement, whose mutation lineage is computed
against the fresh ``df_clean`` and therefore recomputes rather than restoring
the stale cached frame that still carried ``revenue``.
"""
import unittest
from unittest.mock import MagicMock
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

    def _fresh_df_clean(self):
        """Replace df_clean with a fresh frame (no revenue/month columns)."""
        self.shell.user_ns['df_clean'] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.rand(100) * 1000,
            'units': np.random.randint(1, 20, 100),
            'product': np.random.choice(['A', 'B', 'C'], 100)
        })

    def test_exact_user_scenario(self):
        """
        Comment out a subscript-assignment line and re-run the cell:
        1. Cell has: revenue line + month line -> both columns cached
        2. Comment out the revenue line
        3. Re-run the cell against a fresh df_clean
        4. revenue must NOT reappear (no stale cache restore); month must.
        """
        original_cell = (
            "df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        self.magics.cash("", original_cell)
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)

        # User comments out the revenue line and re-runs against a fresh frame.
        commented_cell = (
            "#df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        self._fresh_df_clean()
        self.magics.cash("", commented_cell)

        # The commented-out mutation must not be restored from cache.
        self.assertNotIn('revenue', self.shell.user_ns['df_clean'].columns,
                         "revenue column should not reappear after commenting out the line")
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)

    def test_saved_notebook_with_commented_line(self):
        """
        Same invariant when the user re-runs the cell after caching the
        original: the surviving ``month`` statement recomputes against the
        fresh frame instead of restoring the cached frame that held
        ``revenue``. (The cell content executed is the source of truth for the
        current cell; the on-disk notebook version is irrelevant to this path.)
        """
        original_cell = (
            "df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        self.magics.cash("", original_cell)
        self.assertIn('revenue', self.shell.user_ns['df_clean'].columns)

        commented_cell = (
            "#df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )
        self._fresh_df_clean()
        self.magics.cash("", commented_cell)

        self.assertNotIn('revenue', self.shell.user_ns['df_clean'].columns,
                         "revenue column should not reappear after commenting out the line")
        self.assertIn('month', self.shell.user_ns['df_clean'].columns)

if __name__ == '__main__':
    unittest.main(verbosity=2)

