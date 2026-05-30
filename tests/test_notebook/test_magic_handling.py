"""
Test handling of Jupyter magics in upstream cells
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
import json

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


class TestMagicHandling(unittest.TestCase):
    """Test that Jupyter magics are correctly ignored in upstream cells."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.backend.clear()
        self.cash = Cash(backend=self.backend, register_magic=False)
        self.shell = MockShell()
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        self.magics._auto_cache_enabled = True
        
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, 'test.ipynb')
        
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_notebook(self, cells):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [cell]
                }
                for cell in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)
    
    def test_upstream_cell_with_magic_is_ignored(self):
        """Test that an upstream cell with magics doesn't cause SyntaxError."""
        print("\n=== TEST: Upstream Cell with Magic ===")
        
        # Cell 1: Mixed magic and code
        cell1 = "%matplotlib inline\nx = 10"
        # Cell 2: Uses x
        cell2 = "result = x * 2"
        
        self.create_notebook([cell1, cell2])
        
        with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells:
            def get_cells(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return [cell['source'][0] for cell in nb['cells']]
            
            mock_get_cells.side_effect = get_cells
            
            # Simulate initial execution state
            print("--- Step 1: Simulate initial execution of 'x=10' ---")
            # We need to manually populate tracking dicts to simulate that x=10 was executed and cached
            # effectively ignoring the magic part which CashMagics usually skips caching for anyway (unless handled)
            self.magics._tracking_state.executed_cell_codes['x'] = "x = 10"
            self.magics._tracking_state.executed_cell_hashes['x'] = "hash_of_x_10"
            self.magics._tracking_state.variable_lineage['x'] = "lineage_of_x_10"
            self.shell.user_ns['x'] = 10
            
            print("--- Step 2: Modify notebook to have magic and x=20 ---")
            cell1_v2 = "%matplotlib inline\nx = 20"
            self.create_notebook([cell1_v2, cell2])
            
            print("--- Step 3: Run Cell 2 ---")
            # This should trigger re-execution of Cell 1
            self.magics._execute_cell(cell2)
            
            print(f"x in memory: {self.shell.user_ns.get('x')}")
            # If re-execution works despite magic, x should be 20
            self.assertEqual(self.shell.user_ns.get('x'), 20)
            print("[OK] Cell 1 re-executed successfully (magic ignored)")

if __name__ == '__main__':
    unittest.main(verbosity=2)
