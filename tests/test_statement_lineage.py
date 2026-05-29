"""
Test for statement-level dependency invalidation
"""
import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json

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
        self.user_global_ns = self.user_ns


class TestStatementLineage(unittest.TestCase):
    """Test statement-level dependency tracking."""
    
    def setUp(self):
        self.backend = InMemoryBackend()
        self.backend.clear()
        self.cash = Cash(backend=self.backend, register_magic=False)
        self.shell = MockShell()
        self.magics = CashMagics(self.shell, self.cash)
        self.magics._debug = True
        self.magics._auto_cache_enabled = True
        
        # Create a temporary notebook file
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, 'test_stmts.ipynb')
        
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
                    "source": [cell] # source expects list of strings
                }
                for cell in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)

    def test_multi_statement_cell_updates(self):
        """
        Test that granular statement tracking works within cells.
        Validation:
        1. Cell 1 has 2 statements: `a=1` and `b=2`.
        2. Cell 2 uses `a` and `b`.
        3. Run both.
        4. Modify Cell 1 to `a=1` (same) and `b=3` (changed).
        5. Run Cell 2.
        6. Verify ONLY `b=3` statement is re-executed (conceptually), or at least that correct result is propagated.
        """
        print("\n=== TEST: Multi-Statement Cell Updates ===")
        
        # Step 1: Initial Notebook
        # Cell 1: Defines a and b
        cell1_v1 = "a = 1\nb = 2"
        # Cell 2: Uses a and b
        cell2 = "c = a + b"
        
        self.create_notebook([cell1_v1, cell2])
        
        with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.checker.get_notebook_cells_with_ids') as mock_get_ids:
            def get_cells(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return ["".join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells']]
            
            def get_cells_with_ids(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return [(c.get('id', f'cell_{i}'), "".join(c['source']) if isinstance(c['source'], list) else c['source']) for i, c in enumerate(nb['cells'])]
            
            mock_get_cells.side_effect = get_cells
            mock_get_ids.side_effect = get_cells_with_ids
            
            # Execute Cell 1
            print("Running Cell 1 (v1)...")
            self.magics._execute_cell(cell1_v1)
            self.assertEqual(self.shell.user_ns.get('a'), 1)
            self.assertEqual(self.shell.user_ns.get('b'), 2)
            
            # Execute Cell 2
            print("Running Cell 2...")
            self.magics._execute_cell(cell2)
            self.assertEqual(self.shell.user_ns.get('c'), 3)
            
            # Step 2: Modify Cell 1
            # Keep 'a=1', change 'b=2' -> 'b=3'
            cell1_v2 = "a = 1\nb = 3"
            self.create_notebook([cell1_v2, cell2])
            
            # Execute Cell 2 again
            # Trigger: logic says `c` depends on `a` and `b`.
            # `a` and `b` are in memory (stale `b=2`).
            # `UpstreamChecker` scans notebook.
            # Simulates `a=1`. Hash matches memory `a`. Lineage OK.
            # Simulates `b=3`. Hash differs from memory `b`.
            # Should re-execute `b=3`.
            # Then check `c`. Memory `c=3`. Inputs `a`, `b`.
            # `c` will run because `c` is the current cell? No, current cell is user running it.
            # Wait, user runs Cell 2.
            # Before Cell 2 runs, hook checks upstream.
            # It finds `b=3` needs to run. It runs it. Memory `b` -> 3.
            # Then Cell 2 runs. `c = a + b` -> `1 + 3 = 4`.
            
            print("Running Cell 2 again (expecting upstream re-execution)...")
            # We clear c to ensure it's recomputed?
            # Or reliance on user running it?
            # The test simulates user running Cell 2.
            self.magics._execute_cell(cell2)
            
            # Verify results
            self.assertEqual(self.shell.user_ns.get('a'), 1)
            self.assertEqual(self.shell.user_ns.get('b'), 3, "Upstream b should be updated to 3")
            self.assertEqual(self.shell.user_ns.get('c'), 4, "Downstream c should include updated b")
            
            print("[OK] Test passed.")

    def test_redundant_execution_on_mutable_objects(self):
        """
        Test that we do NOT re-execute intermediate mutation steps if the final state is consistent.
        
        Scenario:
        1. d = {'val': 0}
        2. d['a'] = 1
        3. d['b'] = 2
        
        If we run this, d has {val:0, a:1, b:2}.
        Next time we check upstream:
        - Statement 2 produces intermediate d (with a=1, no b).
        - Memory has final d (with a=1, b=2).
        - Old logic: Mismatch! Re-execute Statement 2. (Redundant)
        - New logic: Simulate all. Final virtual d has a=1,b=2. Matches memory. No re-execution.
        """
        print("\n=== TEST: Redundant Mutable Execution Fix ===")
        
        # Cell 1: Defines dict and mutates it
        cell1 = "d = {'val': 0}\nd['a'] = 1\nd['b'] = 2"
        cell2 = "print(d)"
        
        self.create_notebook([cell1, cell2])
        
        with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.checker.get_notebook_cells_with_ids') as mock_get_ids:
            def get_cells(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return ["".join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells']]
            
            def get_cells_with_ids(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return [(c.get('id', f'cell_{i}'), "".join(c['source']) if isinstance(c['source'], list) else c['source']) for i, c in enumerate(nb['cells'])]
            
            mock_get_cells.side_effect = get_cells
            mock_get_ids.side_effect = get_cells_with_ids
            
            # 1. Initial Run
            print("Running Cell 1...")
            self.magics._execute_cell(cell1)
            d_val = self.shell.user_ns.get('d')
            self.assertEqual(d_val, {'val': 0, 'a': 1, 'b': 2})
            
            # Verify lineage matches expectation manually
            # This helps confirm if StatementProcessor did its job right
            self.magics._tracking_state.variable_lineage.get('d')
            
            # 2. Run Downstream (trigger check)
            print("Running Cell 2 (Check for redundant re-execution)...")
            
            with patch('cash.notebook.upstream.UpstreamChecker._reexecute_statements') as mock_reexec:
                 self.magics._execute_cell(cell2)
                 
                 if mock_reexec.call_count > 0:
                     args = mock_reexec.call_args[0]
                     stmts = args[0]
                     print(f"DEBUG: Re-executed statements: {stmts}")
                 
                 # Expectation: 0 re-executions because state is consistent
                 self.assertEqual(mock_reexec.call_count, 0, "Should not re-execute any statements if state is consistent")
            
            print("[OK] Test passed: No redundant re-execution.")

    def test_import_not_tracked_as_broken(self):
        """
        Test that modules are NOT tracked as broken vars by the upstream checker.

        Design decision: the upstream checker explicitly skips modules in
        ``virtual_modules``.  Modules are considered re-importable and are
        not treated as broken variables that require restoration.

        If a cell uses a module that hasn't been imported yet, the normal
        Python NameError will surface, prompting the user to run the import
        cell first.
        """
        print("\n=== TEST: Import Not Tracked As Broken ===")

        # Cell 1: Import
        cell1 = "import math"
        # Cell 2: Use it
        cell2 = "print(math.pi)"

        self.create_notebook([cell1, cell2])

        with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.checker.get_notebook_cells_with_ids') as mock_get_ids:
            def get_cells(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return ["".join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells']]

            def get_cells_with_ids(_path=None):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                return [(c.get('id', f'cell_{i}'), "".join(c['source']) if isinstance(c['source'], list) else c['source']) for i, c in enumerate(nb['cells'])]

            mock_get_cells.side_effect = get_cells
            mock_get_ids.side_effect = get_cells_with_ids

            # Run Cell 1 first so math is in user_ns
            self.magics._execute_cell(cell1)
            self.assertIn('math', self.shell.user_ns)

            # Run Cell 2 — should succeed (math is available)
            self.magics._execute_cell(cell2)

            # Verify that the upstream checker does NOT redundantly
            # re-execute the import (modules are skipped by design)
            with patch('cash.notebook.upstream.UpstreamChecker._reexecute_statements') as mock_reexec:
                self.magics._execute_cell(cell2)
                self.assertEqual(mock_reexec.call_count, 0,
                                 "Should not re-execute import statements")

            print("[OK] Test passed: Modules are not tracked as broken vars.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
