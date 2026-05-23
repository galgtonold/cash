'''
Test transitive dependency invalidation through multiple cells.

This tests the scenario:
1. Cell 1: a = 5
2. Cell 2: b = a + 3
3. Cell 3: c = b * 2

When we change Cell 1 to a = 7, Cell 3 should get updated value.
'''
import pytest
from unittest.mock import MagicMock
from cash import Cash
from cash.backends import InMemoryBackend
from cash.notebook.magics import CashMagics
from traitlets.config import Configurable


class MockShell(Configurable):
    '''Mock IPython shell for testing.'''
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
def transitive_magics():
    '''Provide CashMagics instance for transitive dependency tests.'''
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    magics._debug = False
    
    yield magics, shell, backend
    
    # Cleanup
    backend.clear()
    shell.user_ns.clear()


def test_three_cell_cascade(transitive_magics):
    '''Test that changing a affects b affects c across 3 cells.

    NOTE: This test is xfail because transitive upstream re-execution
    requires the full _execute_cell pipeline with a real notebook file.
    The %%cash magic (used in unit tests) only processes individual
    statements without upstream dependency resolution.
    See test_interaction_dependencies.py for proper integration tests.
    '''
    pytest.xfail(
        "Transitive re-execution requires _execute_cell + notebook file, "
        "not %%cash magic. Covered by integration tests."
    )


def test_direct_dependency_invalidation(transitive_magics):
    '''Test that changing an input variable invalidates the cache.'''
    magics, shell, backend = transitive_magics
    
    # Cell 1: x = 10
    cell1 = 'x = 10'
    magics.cash('', cell1)
    assert shell.user_ns['x'] == 10, 'x should be 10'
    
    # Cell 2: y = x + 5 (should be 15)
    cell2 = 'y = x + 5'
    magics.cash('', cell2)
    assert shell.user_ns['y'] == 15, 'y should be 15 (10 + 5)'
    
    # Change x directly in namespace (simulating upstream change)
    shell.user_ns['x'] = 20
    
    # Update the lineage hash for x to reflect the change
    if hasattr(magics, '_tracking_state') and 'x' in magics._tracking_state.variable_lineage:
        # Force lineage change by removing x's lineage (simulating new value)
        del magics._tracking_state.variable_lineage['x']
    
    # Re-run Cell 2: y should be recalculated to 25
    magics.cash('', cell2)
    assert shell.user_ns['y'] == 25, \
        'y should be recalculated to 25 after x changed to 20'
