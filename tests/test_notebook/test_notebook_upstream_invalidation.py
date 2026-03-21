"""
Test for dependency invalidation when upstream cells change in notebook
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import types
import tempfile
import json

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

# Mock capture_output
class MockCaptureOutput:
    def __init__(self, *args, **kwargs):
        self.stdout = ""
        self.stderr = ""
        self.outputs = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def show(self):
        pass

sys.modules['IPython.utils.io'].capture_output = MockCaptureOutput

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


class TestNotebookDependencyInvalidation(unittest.TestCase):
    """Test dependency invalidation with real notebook file simulation."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.backend.clear()  # Ensure clean state
        self.cash = Cash(backend=self.backend, register_magic=False)
        
        self.shell = MockShell()
        
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        self.magics._auto_cache_enabled = True
        
        # Create a temporary notebook file
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, 'test.ipynb')
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_notebook(self, cells):
        """Create a notebook file with given cell contents."""
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
    
    def test_upstream_cell_change_invalidates_cache(self):
        """
        Test scenario with real notebook file:
        1. Notebook has: Cell 1 = 'South', Cell 2 uses selected_region
        2. Run both cells - Cell 2 gets cached
        3. Run Cell 2 again - should hit cache
        4. Modify Cell 1 in notebook to 'North' (but don't execute)
        5. Run Cell 2 again - should detect Cell 1 changed and re-execute it
        """
        print("\n=== TEST: Upstream Cell Change with Notebook File ===")
        
        # Step 1: Create initial notebook
        print("\n--- Step 1: Create notebook with Cell 1 = 'South' ---")
        cell1_v1 = "selected_region = 'South'"
        cell2 = "result = f'Region: {selected_region}'"
        self.create_notebook([cell1_v1, cell2])
        
        # Mock get_notebook_cells to return our notebook
        with patch('cash.notebook.upstream.get_notebook_cells') as mock_get_cells:
            def get_cells(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return [cell['source'][0] for cell in nb['cells']]
            
            mock_get_cells.side_effect = get_cells
            
            # Step 2: Execute Cell 1
            print("\n--- Step 2: Execute Cell 1 ---")
            self.magics._execute_cell(cell1_v1)
            
            self.assertEqual(self.shell.user_ns.get('selected_region'), 'South')
            print(f"[OK] selected_region = {self.shell.user_ns['selected_region']}")
            
            # Step 3: Execute Cell 2 (first time)
            print("\n--- Step 3: Execute Cell 2 (first time) ---")
            self.magics._execute_cell(cell2)
            
            self.assertEqual(self.shell.user_ns.get('result'), 'Region: South')
            print(f"[OK] result = {self.shell.user_ns['result']}")
            
            # Step 4: Execute Cell 2 again (should cache)
            print("\n--- Step 4: Execute Cell 2 again (cache hit expected) ---")
            self.magics._execute_cell(cell2)
            
            self.assertEqual(self.shell.user_ns.get('result'), 'Region: South')
            print(f"[OK] result = {self.shell.user_ns['result']} (from cache)")
            
            # Step 5: Modify Cell 1 in notebook
            print("\n--- Step 5: Modify Cell 1 in notebook to 'North' ---")
            cell1_v2 = "selected_region = 'North'"
            self.create_notebook([cell1_v2, cell2])
            print("[OK] Notebook updated (Cell 1 now defines 'North')")
            print(f"  selected_region in memory = {self.shell.user_ns['selected_region']} (still 'South')")
            
            # Step 6: Execute Cell 2 - should detect upstream change
            print("\n--- Step 6: Execute Cell 2 - should auto-run Cell 1 first ---")
            self.magics._execute_cell(cell2)
            
            # Should have detected change and re-executed Cell 1
            print(f"  selected_region after = {self.shell.user_ns.get('selected_region')}")
            print(f"  result after = {self.shell.user_ns.get('result')}")
            
            self.assertEqual(self.shell.user_ns.get('selected_region'), 'North')
            self.assertEqual(self.shell.user_ns.get('result'), 'Region: North')
            print("[OK] Test passed: Upstream cell was re-executed automatically!")

if __name__ == '__main__':
    unittest.main(verbosity=2)


