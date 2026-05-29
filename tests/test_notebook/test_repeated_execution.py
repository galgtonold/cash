"""
Integration test for repeated cell execution.

Tests that running a cell multiple times without upstream changes 
does NOT trigger unnecessary upstream restoration.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock, patch

from cash.notebook.ipython.magics import CashMagics
from cash.core import Cash
from cash.backends import InMemoryBackend
from traitlets.config.configurable import Configurable


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


@pytest.fixture
def magics_fixture():
    """Provide CashMagics instance for testing."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    magics._debug = True
    
    yield magics, shell, backend
    
    backend.clear()
    shell.user_ns.clear()


class TestRepeatedExecution:
    """Test that repeated execution doesn't cause unnecessary restoration."""
    
    def test_middle_cell_after_downstream_modification(self, magics_fixture):
        """
        Scenario that replicates the financial_analysis_demo issue:
        
        Notebook structure:
        - Cell 0: Load data -> df (upstream)
        - Cell 1: df = df.sort_values(...) (upstream, modifies df)
        - Cell 2: df (print cell - THIS is the cell we'll run repeatedly)  
        - Cell 3: df['new_col'] = ... (downstream, modifies df further)
        
        Execution sequence:
        1. Execute all cells 0, 1, 2, 3 in order
        2. Now df has lineage from cell 3 (downstream modification)
        3. Execute cell 2 again - the upstream checker sees:
           - Virtual lineage: hash after cell 1 (df.sort_values)
           - Actual lineage: hash after cell 3 (downstream modification)
           - MISMATCH! This SHOULD trigger restoration because cell 2 is a
             read-only display cell (df is input but NOT output). The cell
             should show df at its position in the notebook, not with
             downstream mutations applied.
        """
        magics, shell, backend = magics_fixture
        
        notebook_cells = [
            "import pandas as pd\ndf = pd.DataFrame({'a': [3,1,2]})",  # Cell 0: Create df
            "df = df.sort_values('a').reset_index(drop=True)",         # Cell 1: Sort df
            "df  # print cell",                                         # Cell 2: Print df (run repeatedly)
            "df['b'] = df['a'] * 2",                                   # Cell 3: Add column (downstream)
        ]
        
        temp_dir = tempfile.mkdtemp()
        notebook_path = os.path.join(temp_dir, 'test.ipynb')
        
        notebook = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}
                for cell in notebook_cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)
        
        try:
            with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells, \
                 patch('cash.notebook.upstream.checker.get_notebook_cells_with_ids') as mock_get_ids:
                    def get_cells_1(_path=None):
                        with open(notebook_path, encoding='utf-8') as nf:
                            data = json.load(nf)
                        cells = []
                        for c in data['cells']:
                            src = c['source']
                            if isinstance(src, list):
                                src = ''.join(src)
                            cells.append(src)
                        return cells
                    mock_get_cells.side_effect = get_cells_1
                    mock_get_ids.return_value = None
                    
                    # Execute all cells in order (simulating initial notebook run)
                    print("\n=== Initial execution of all cells ===")
                    for i, cell in enumerate(notebook_cells):
                        print(f"\n--- Executing Cell {i} ---")
                        magics._execute_cell(cell)
                    
                    # Verify df has the downstream modification
                    assert 'b' in shell.user_ns['df'].columns, "df should have column 'b' from cell 3"
                    
                    # Capture the lineage after cell 1 (the upstream state for cell 2)
                    # and after all cells
                    lineage_after_all_cells = magics._tracking_state.variable_lineage.get('df')
                    print(f"\ndf lineage after all cells: {lineage_after_all_cells[:16]}...")
                    
                    # Now execute cell 2 again (the middle "print" cell)
                    print("\n=== Re-executing Cell 2 (should restore to upstream state) ===")
                    
                    # Cell 2 is a read-only display cell: df is an INPUT but not an OUTPUT.
                    # The upstream checker should detect that df's lineage was advanced by
                    # downstream cell 3, and restore df to its upstream state (after cell 1).
                    # This ensures the display cell shows df without the 'b' column.
                    
                    magics._execute_cell(notebook_cells[2])
                    
                    lineage_after_rerun = magics._tracking_state.variable_lineage.get('df')
                    print(f"df lineage after re-running cell 2: {lineage_after_rerun[:16]}...")
                    
                    # The lineage SHOULD change — the downstream modification should be
                    # rolled back for this read-only cell. df should be restored to its
                    # state after cell 1 (sort), without the 'b' column from cell 3.
                    # Note: the downstream column 'b' should NOT be present because
                    # the upstream checker restores df to the upstream state.
                    assert lineage_after_rerun != lineage_after_all_cells, \
                        f"Lineage SHOULD change for read-only display cell!\n" \
                        f"  Before rerun: {lineage_after_all_cells[:32]}...\n" \
                        f"  After rerun:  {lineage_after_rerun[:32]}..."
                    
                    # The column 'b' should NOT be present since df was restored
                    # to its upstream state (after sort, before column addition)
                    assert 'b' not in shell.user_ns['df'].columns, \
                        "Column 'b' should NOT exist after re-running cell 2 — " \
                        "df should be restored to upstream state"
                    
                    print("\n[PASS] Read-only display cell correctly restored to upstream state")
                    
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_repeated_same_cell_no_change(self, magics_fixture):
        """
        Simpler test: Just run the same cell twice without any downstream modifications.
        This should definitely NOT trigger any restoration.
        """
        magics, shell, backend = magics_fixture
        
        notebook_cells = [
            "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})",
            "df.sum()"  # Read-only operation
        ]
        
        temp_dir = tempfile.mkdtemp()
        notebook_path = os.path.join(temp_dir, 'test.ipynb')
        
        notebook = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}
                for cell in notebook_cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)
        
        try:
            with patch('cash.notebook.upstream.checker.get_notebook_cells') as mock_get_cells, \
                 patch('cash.notebook.upstream.checker.get_notebook_cells_with_ids') as mock_get_ids:
                    def get_cells_2(_path=None):
                        with open(notebook_path, encoding='utf-8') as nf:
                            data = json.load(nf)
                        cells = []
                        for c in data['cells']:
                            src = c['source']
                            if isinstance(src, list):
                                src = ''.join(src)
                            cells.append(src)
                        return cells
                    mock_get_cells.side_effect = get_cells_2
                    mock_get_ids.return_value = None
                    
                    # Execute both cells
                    print("\n=== Initial execution ===")
                    for i, cell in enumerate(notebook_cells):
                        print(f"--- Cell {i} ---")
                        magics._execute_cell(cell)
                    
                    df_lineage_1 = magics._tracking_state.variable_lineage.get('df')
                    print(f"df lineage after first run: {df_lineage_1[:16]}...")
                    
                    # Run cell 1 again
                    print("\n=== Re-executing cell 1 ===")
                    magics._execute_cell(notebook_cells[1])
                    
                    df_lineage_2 = magics._tracking_state.variable_lineage.get('df')
                    print(f"df lineage after second run: {df_lineage_2[:16]}...")
                    
                    assert df_lineage_1 == df_lineage_2, "Lineage should not change"
                    print("\n[PASS] Repeated cell execution works correctly")
                    
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
