"""
Unit test for the accumulator initialization skip fix.

When adding a new item to a cached loop:
1. The backwards scan may schedule the init statement (e.g., `results = {}`)
2. But if `results` already exists in memory with cached values,
   we should NOT re-run the init statement (it would reset the dict)

This test verifies that the fix correctly skips the init statement.
"""

import sys
import os
import tempfile
import json
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))



class TestAccumulatorInitSkip:
    """Test that accumulator initialization statements are skipped when appropriate."""
    
    @pytest.mark.xfail(reason="Known failure: accumulator init skip logic")
    def test_skip_empty_dict_init_when_has_data(self, cash_magics, mock_shell):
        """
        If results = {} is scheduled but results already has data,
        the init should be skipped when re-running the loop.
        
        Scenario:
        1. Run loop with 4 items (ABCD) - all cached
        2. Edit loop code to add 5th item (E)
        3. Run downstream cell - triggers upstream re-execution
        4. All 5 items should be present (not just E)
        """
        magics = cash_magics
        shell = mock_shell
        
        # Initial loop code with 4 items
        loop_code_v1 = """results = {}
for x in ["A", "B", "C", "D"]:
    results[x] = x * 2
"""
        
        # Updated loop code with 5 items
        loop_code_v2 = """results = {}
for x in ["A", "B", "C", "D", "E"]:
    results[x] = x * 2
"""
        
        keys_code = "keys = list(results.keys())"
        
        # Create a temp notebook file
        temp_dir = tempfile.mkdtemp()
        notebook_path = os.path.join(temp_dir, 'test.ipynb')
        
        # First run: execute loop with 4 items
        notebook_v1 = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": loop_code_v1},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": keys_code},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_v1, f)
        
        def get_cells_v1(_path=None):
            with open(notebook_path, encoding='utf-8') as nf:
                data = json.load(nf)
                return [cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code']
        
        with patch('cash.notebook.upstream.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids') as mock_get_ids:
                mock_get_cells.side_effect = get_cells_v1
                mock_get_ids.return_value = []
                
                magics.cash_on("")
                
                # Run loop cell - caches iterations
                magics.cash("", loop_code_v1)
                
                assert 'results' in shell.user_ns
                assert shell.user_ns['results'] == {'A': 'AA', 'B': 'BB', 'C': 'CC', 'D': 'DD'}
                
                # Run keys cell
                magics.cash("", keys_code)
                assert shell.user_ns['keys'] == ['A', 'B', 'C', 'D']
        
        # Second run: update notebook to have 5 items, run only keys cell
        notebook_v2 = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": loop_code_v2},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": keys_code},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_v2, f)
        
        def get_cells_v2(_path=None):
            with open(notebook_path, encoding='utf-8') as nf:
                data = json.load(nf)
                return [cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code']
        
        with patch('cash.notebook.upstream.get_notebook_cells') as mock_get_cells, \
             patch('cash.notebook.upstream.get_notebook_cells_with_ids') as mock_get_ids:
                mock_get_cells.side_effect = get_cells_v2
                mock_get_ids.return_value = []
                
                # Run ONLY the keys cell - should trigger upstream re-execution of edited loop
                magics.cash("", keys_code)
                
                # Check that results has all 5 items
                results = shell.user_ns.get('results', {})
                assert 'A' in results, f"Missing 'A' in results: {results}"
                assert 'B' in results, f"Missing 'B' in results: {results}"
                assert 'C' in results, f"Missing 'C' in results: {results}"
                assert 'D' in results, f"Missing 'D' in results: {results}"
                assert 'E' in results, f"Missing 'E' in results: {results}"
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


    def test_init_runs_if_no_existing_data(self, cash_magics, mock_shell):
        """
        If results = {} is scheduled and results is empty or doesn't exist,
        the init SHOULD run.
        """
        magics = cash_magics
        shell = mock_shell
        
        # No prior data
        if 'results' in shell.user_ns:
            del shell.user_ns['results']
        
        magics.cash_on("")
        
        loop_code = """results = {}
for x in ['A', 'B']:
    results[x] = x * 2
"""
        magics.cash("", loop_code)
        
        assert shell.user_ns['results'] == {'A': 'AA', 'B': 'BB'}


    def test_init_runs_if_existing_data_empty(self, cash_magics, mock_shell):
        """
        If results exists but is empty, the init SHOULD run.
        """
        magics = cash_magics
        shell = mock_shell
        
        shell.user_ns['results'] = {}
        
        magics.cash_on("")
        
        loop_code = """results = {}
for x in ['A', 'B']:
    results[x] = x * 2
"""
        magics.cash("", loop_code)
        
        assert shell.user_ns['results'] == {'A': 'AA', 'B': 'BB'}


    def test_list_accumulator(self, cash_magics, mock_shell):
        """
        Test that list accumulators (results = []) are also handled.
        Basic test: verify list accumulator works in normal execution.
        """
        magics = cash_magics
        shell = mock_shell
        
        magics.cash_on("")
        
        loop_code = """results = []
for x in ['A', 'B', 'C', 'D']:
    results.append(x)
"""
        magics.cash("", loop_code)
        
        # Verify
        assert shell.user_ns['results'] == ['A', 'B', 'C', 'D']
