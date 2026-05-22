from cash.notebook.cache_status import CacheStatus
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from cash.notebook.magics import CashMagics
from cash.core import Cash

class MockShell:
    def __init__(self):
        self.user_ns = {}
        self.user_global_ns = {}
        self.events = self.Events()
        self.run_cell = self._run_cell_dummy
    
    class Events:
        def register(self, event, callback):
            pass
            
    def _run_cell_dummy(self, raw_cell, *args, **kwargs):
        pass
        
    def get_parent(self):
        return None

@pytest.fixture
def cash_instance(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"), debug=True)

@pytest.fixture
def magics(cash_instance):
    shell = MockShell()
    return CashMagics(shell, cash_instance)

def test_restore_variable_returns_metrics(magics, cash_instance):
    """
    Test that Restorer.restore_variable returns metrics with saved_time.
    """
    # 1. Setup Cache Entry
    var_name = 'a'
    value = 100
    execution_time = 1.23
    source_code = "a = 100"
    
    # Construct cache key
    # Key usually is var_sources lookup.
    cache_key = "stmt:test_hash"
    magics._tracking_state.variable_sources[var_name] = cache_key
    
    metadata = {
        'inputs': [],
        'outputs': [var_name],
        'execution_time': execution_time,
        'code': source_code,
        'output_lineages': {var_name: 'lineage_hash'}
    }
    
    data = {
        'variables': {var_name: value}
    }
    
    cash_instance.backend.set(cache_key, data, metadata)
    
    # 2. Call restore_variable
    # Ensure it's NOT in user_ns
    if var_name in magics.shell.user_ns:
        del magics.shell.user_ns[var_name]
        
    metrics = magics._restorer.restore_variable(var_name)
    
    # 3. Verify
    assert var_name in magics.shell.user_ns
    assert magics.shell.user_ns[var_name] == value
    
    print(f"Metrics: {metrics}")
    
    assert len(metrics) == 1
    m = metrics[0]
    assert m['status'] == CacheStatus.RESTORED
    assert m['saved_time'] == execution_time
    assert m['code'] == source_code
    assert 'a' in m['restored_vars']
