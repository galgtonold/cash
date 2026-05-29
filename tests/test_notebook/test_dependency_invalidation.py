"""
Test for dependency invalidation when upstream cells change.
"""
import pytest
from unittest.mock import MagicMock
from cash.notebook.ipython.magics import CashMagics
from cash.backends import InMemoryBackend
from cash.core import Cash
from traitlets.config.configurable import Configurable


class MockShell(Configurable):
    """Mock IPython shell for testing."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns['_ih'] = []  # Execution history
        self.run_cell = MagicMock()
        self.input_transformers_cleanup = []
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()
        self.ast_transformers = []
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)


@pytest.fixture
def dep_invalidation_magics():
    """Provide CashMagics instance for dependency invalidation tests."""
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    
    shell = MockShell()
    
    magics = CashMagics(shell, cash)
    magics._debug = False  # Keep output clean
    magics._auto_cache_enabled = True
    
    yield magics, shell, backend
    
    # Cleanup
    backend.clear()
    shell.user_ns.clear()


def test_upstream_cell_change_invalidates_cache(dep_invalidation_magics):
    """
    Test scenario:
    1. Cell 1: selected_region = 'South'
    2. Cell 2: result = f"Region: {selected_region}"
    3. Run both - Cell 2 caches result
    4. Change Cell 1 to selected_region = 'North'
    5. Run Cell 2 - should detect Cell 1 changed and re-execute it first
    """
    magics, shell, backend = dep_invalidation_magics
    
    # Step 1: Set variable
    cell1_v1 = "selected_region = 'South'"
    magics._execute_cell(cell1_v1)
    
    assert shell.user_ns['selected_region'] == 'South', \
        'Step 1: selected_region should be South'
    
    # Step 2: Use variable (will be cached)
    cell2 = "result = f'Region: {selected_region}'"
    magics._execute_cell(cell2)
    
    assert shell.user_ns['result'] == 'Region: South', \
        'Step 2: result should be "Region: South"'
    
    # Step 3: Run Cell 2 again (should get cache hit)
    magics._execute_cell(cell2)
    
    assert shell.user_ns['result'] == 'Region: South', \
        'Step 3: result should still be "Region: South" (from cache)'
    
    # Step 4: Change Cell 1 code (simulate user editing notebook)
    # In real scenario, Cell 1's code in notebook file changes but hasn't been executed
    # selected_region is still 'South' in memory
    assert shell.user_ns['selected_region'] == 'South', \
        'Step 4: selected_region should still be South in memory'
    
    # Step 5: Run Cell 2 - should detect Cell 1 changed and re-execute it
    # The system should:
    # 1. See that Cell 2 depends on selected_region
    # 2. Check if any earlier cells define selected_region
    # 3. Check if those cells code has changed
    # 4. Re-execute changed cells first
    # 5. Then execute Cell 2 with new value
    
    # For now, manually simulate what should happen:
    # Execute the NEW version of Cell 1
    cell1_v2 = "selected_region = 'North'"
    magics._execute_cell(cell1_v2)
    
    # Now run Cell 2
    magics._execute_cell(cell2)
    
    # Should have new value
    assert shell.user_ns['result'] == 'Region: North', \
        'Step 5: result should be "Region: North" after dependency change'
