import pytest
from unittest.mock import MagicMock, patch
from cash.notebook.magics import CashMagics
from cash.notebook.annotations import CacheAnnotation
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
from traitlets.config.configurable import Configurable

# Force caching regardless of the 10 ms min-execution-time floor.
_PERSIST = CacheAnnotation(persist=True)


class MockShell(Configurable):
    '''Mock IPython shell for testing.'''
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []


@pytest.fixture
def notebook_caching_magics():
    '''Provide CashMagics instance for notebook caching tests.'''
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    
    yield magics, shell, backend
    
    # Cleanup
    backend.clear()
    shell.user_ns.clear()


# Define a local class (picklable if at module level)
class LocalClass:
    def __init__(self, x):
        self.x = x
    def __repr__(self):
        return f'LocalClass({self.x})'

def test_cache_unpicklable_object_in_memory(notebook_caching_magics):
    '''Test that unpicklable objects can be cached in memory.
    _PERSIST forces the cache write regardless of the 10 ms min-execution-time floor.'''
    magics, shell, backend = notebook_caching_magics

    obj = LocalClass(42)

    # Mock execution
    code = 'x = LocalClass(42)'
    shell.user_ns['x'] = obj
    shell.user_ns['x'] = obj
    shell.user_ns['LocalClass'] = LocalClass

    # Needs lineage to be cacheable
    magics._tracking_state.variable_lineage['LocalClass'] = 'mock_class_hash'

    # Mock capture_output context manager
    with patch('cash.notebook.statement_processor.capture_output') as mock_capture:
        mock_context = MagicMock()
        mock_capture.return_value = mock_context
        mock_context.__enter__.return_value = mock_context
        mock_context.stdout = ''
        mock_context.stderr = ''
        mock_context.outputs = []

        # Process statement — use _PERSIST so the trivially-fast statement is cached
        magics._statement_processor.process_statement(code, annotation=_PERSIST)
        
        # Verify it is cached
        entries = backend.list_entries()
        assert len(entries) == 1, 'Should have exactly one cache entry'
        
        # Verify we can retrieve it
        key = entries[0]['key']
        metadata, payload = backend.get(key)
        
        assert payload is not None, 'Payload should not be None'
        assert 'variables' in payload, 'Payload should contain variables'
        assert 'x' in payload['variables'], 'Variable x should be in payload'
        assert isinstance(payload['variables']['x'], LocalClass), 'x should be a LocalClass instance'
        assert payload['variables']['x'].x == 42, 'LocalClass instance should have x=42'
