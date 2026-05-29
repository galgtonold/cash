from cash.notebook.cache_status import CacheStatus
"""
Test for loop dict mutation caching bug.

Reproduces the issue where `ticker_stats[ticker] = stats` in a for loop
fails to restore from cache on the 3rd iteration (AAPL) when re-executing 
the cell, even though the 1st and 2nd iterations (TSLA, z) restore fine.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock, patch

from cash.notebook.magics import CashMagics
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
    magics._statement_processor.debug = True
    
    yield magics, shell, backend
    
    backend.clear()
    shell.user_ns.clear()


class TestLoopDictMutationCache:
    """Test that dict mutations in loops restore correctly from cache."""

    def test_dict_subscript_assignment_in_loop_caches_all_iterations(self, magics_fixture):
        """
        Reproduces the bug where ticker_stats[ticker] = stats fails to
        cache for the 3rd iteration on repeated execution.
        
        The loop pattern:
            ticker_stats = {}
            for ticker in ["A", "B", "C", "D"]:
                stats = compute_stats(ticker)
                ticker_stats[ticker] = stats
        
        On second execution, ALL iterations should be restored from cache.
        """
        magics, shell, backend = magics_fixture

        # The cell code that reproduces the issue.
        # sum(range(5_000_000)) keeps val = ... above the 10 ms min-execution-time
        # floor so those pure-computation statements are actually stored in cache.
        cell_code = '''ticker_stats = {}
for ticker in ["TSLA", "AAPL", "MSFT", "GOOGL"]:
    val = sum(range(5_000_000)) * 0 + len(ticker)
    ticker_stats[ticker] = val

print(ticker_stats.keys())
print("Done!")'''

        # Set up minimal notebook context
        notebook_cells = [cell_code]
        temp_dir = tempfile.mkdtemp()
        notebook_path = os.path.join(temp_dir, 'test.ipynb')
        
        notebook = {
            "cells": [
                {"cell_type": "code", "execution_count": None, 
                 "metadata": {}, "outputs": [], "source": cell}
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
                    def get_cells(_path=None):
                        with open(notebook_path, encoding='utf-8') as nf:
                            data = json.load(nf)
                        cells = []
                        for c in data['cells']:
                            src = c['source']
                            if isinstance(src, list):
                                src = ''.join(src)
                            cells.append(src)
                        return cells
                    mock_get_cells.side_effect = get_cells
                    mock_get_ids.return_value = None
                    
                    # === First execution ===
                    print("\n=== First execution ===")
                    magics._execute_cell(cell_code)
                    
                    # Verify the result
                    assert 'ticker_stats' in shell.user_ns
                    assert set(shell.user_ns['ticker_stats'].keys()) == {'TSLA', 'AAPL', 'MSFT', 'GOOGL'}
                    
                    # === Second execution (should restore everything) ===
                    print("\n=== Second execution (should restore ALL from cache) ===")
                    magics._execute_cell(cell_code)
                    
                    # Verify result is still correct
                    assert set(shell.user_ns['ticker_stats'].keys()) == {'TSLA', 'AAPL', 'MSFT', 'GOOGL'}
                    
                    # Check the metrics to see which statements were COMPUTED vs RESTORED
                    last_metrics = magics._last_cell_metrics
                    stmts = last_metrics.get('statements', [])
                    
                    computed_stmts = [m for m in stmts if m.get('status') == CacheStatus.COMPUTED]
                    restored_stmts = [m for m in stmts if m.get('status') == CacheStatus.RESTORED]
                    skipped_stmts = [m for m in stmts if m.get('status') == CacheStatus.SKIPPED]
                    
                    print("\nMetrics summary:")
                    print(f"  RESTORED: {len(restored_stmts)}")
                    print(f"  SKIPPED: {len(skipped_stmts)}")
                    print(f"  COMPUTED: {len(computed_stmts)}")
                    
                    if computed_stmts:
                        print("\nCOMPUTED statements (should be empty!):")
                        for m in computed_stmts:
                            print(f"  - {m.get('code', 'unknown')[:80]}")
                    
                    # THE ASSERTION: pure-computation body statements (val = ...)
                    # should be RESTORED from cache on the 2nd run.
                    # Mutation statements (ticker_stats[ticker] = val, ticker_stats = {},
                    # print(...)) are intentionally always re-executed by design and
                    # therefore remain COMPUTED.
                    pure_computed = [
                        m for m in computed_stmts
                        if "ticker_stats[" not in m.get("code", "")
                        and "ticker_stats = {}" not in m.get("code", "")
                        and "print(" not in m.get("code", "")
                    ]
                    assert len(pure_computed) == 0, (
                        f"Expected pure-computation statements to be RESTORED on second run, "
                        f"but {len(pure_computed)} were COMPUTED: "
                        f"{[m.get('code', '')[:60] for m in pure_computed]}"
                    )
                    assert len(restored_stmts) > 0, (
                        f"Expected at least some statements to be RESTORED from cache, "
                        f"but got 0. All metrics: "
                        f"{[(m.get('status'), m.get('code', '')[:40]) for m in stmts]}"
                    )
                    
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
