from cash.notebook.cache_status import CacheStatus
from cash.notebook.annotations import CacheAnnotation

import pytest
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock
from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend, FileBackend
from traitlets.config.configurable import Configurable

# Annotation that forces caching regardless of execution time.
# Used in tests that exercise cache mechanics (restore-after-write, etc.)
# using trivial statements, so the 10 ms min-execution-time floor doesn't
# prevent the cache entry from being written.
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
        self.user_global_ns = self.user_ns

@pytest.fixture
def mock_shell():
    shell = MockShell()
    return shell

@pytest.fixture(params=['memory', 'disk'])
def storage_backend(request):
    """Fixture to provide both InMemoryBackend and FileBackend."""
    if request.param == 'memory':
        backend = InMemoryBackend()
        yield backend, None # No temp dir for memory
        backend.clear()
    elif request.param == 'disk':
        temp_dir = tempfile.mkdtemp()
        backend = FileBackend(cache_dir=temp_dir)
        yield backend, temp_dir
        
        # Cleanup
        backend.clear()
        backend.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(params=['memory', 'disk'])
def processor_fixture(request, mock_shell):
    """Fixture providing StatementProcessor backed by different storages."""
    if request.param == 'memory':
        backend = InMemoryBackend()
        temp_dir = None
    elif request.param == 'disk':
        temp_dir = tempfile.mkdtemp()
        backend = FileBackend(cache_dir=temp_dir)
    
    cash = Cash(backend=backend, register_magic=False)
    
    # We create Magics to get tracking dicts initialized
    magics = CashMagics(mock_shell, cash)
    processor = magics._statement_processor
    
    yield processor, mock_shell, backend, temp_dir
    
    # Cleanup
    backend.clear()
    if request.param == 'disk':
        backend.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    mock_shell.user_ns.clear()


class TestStorageIntegration:
    
    def test_basic_caching(self, processor_fixture):
        """Test basic execute -> restore cycle for both backends.
        _PERSIST forces caching regardless of the 10 ms min-execution-time floor
        so the cache-mechanics path is exercised with trivial statements."""
        processor, shell, backend, _ = processor_fixture
        code = "a = 100\nb = a + 50"

        # 1. Execute (Miss)
        metrics1 = processor.process_statement(code, annotation=_PERSIST)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['b'] == 150

        # 2. Restore (Hit)
        shell.user_ns.pop('b', None)
        metrics2 = processor.process_statement(code, annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.RESTORED
        assert shell.user_ns['b'] == 150

        # Verify specific storage source is reported correctly
        if isinstance(backend, InMemoryBackend):
            assert 'RAM' in metrics2.get('storage', [])
        else:
            # FileBackend storage might rely on what's in metadata from saved cycle
            pass

    def test_code_change(self, processor_fixture):
        """Test cache invalidation on code change.
        _PERSIST forces caching regardless of the 10 ms min-execution-time floor."""
        processor, shell, backend, _ = processor_fixture

        # 1. Run v1
        code1 = "x = 'initial'"
        processor.process_statement(code1, annotation=_PERSIST)
        assert shell.user_ns['x'] == 'initial'

        # 2. Run v2
        code2 = "x = 'modified'"
        metrics2 = processor.process_statement(code2, annotation=_PERSIST)
        assert metrics2['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['x'] == 'modified'

        # Verify 2 entries in backend
        assert len(backend.list_entries()) == 2

    def test_input_dependency(self, processor_fixture):
        """Test invalidation when input dependency changes."""
        processor, shell, backend, _ = processor_fixture
        
        # 1. Setup Input
        code_input = "val = 10"
        processor.process_statement(code_input)
        
        # 2. Run Dependent
        code_dep = "res = val * 2"
        metrics1 = processor.process_statement(code_dep)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['res'] == 20
        
        # 3. Change Input
        shell.user_ns.pop('val', None) # Force re-exec or just overwrite
        code_input_2 = "val = 20"
        processor.process_statement(code_input_2)
        
        # 4. Run Dependent Again
        metrics2 = processor.process_statement(code_dep)
        assert metrics2['status'] == CacheStatus.COMPUTED # Should miss because 'val' hash changed
        assert shell.user_ns['res'] == 40

    def test_file_dependency(self, processor_fixture):
        """Test file dependency invalidation."""
        processor, shell, backend, _ = processor_fixture
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
            f.write("v1")
            path = f.name.replace(os.sep, '/')
            
        try:
            code = f"with open('{path}', 'r') as f: data = f.read()"
            
            # Run 1
            processor.process_statement(code)
            assert shell.user_ns['data'] == 'v1'
            
            # Modify File
            time.sleep(1.1)
            with open(path, 'w') as f:
                f.write("v2")
                
            # Run 2
            shell.user_ns.pop('data', None)
            metrics = processor.process_statement(code)
            assert metrics['status'] == CacheStatus.COMPUTED
            assert shell.user_ns['data'] == 'v2'
            
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_unpicklable_object_handling(self, processor_fixture):
        """Test that unpicklable objects don't crash the backend."""
        processor, shell, backend, _ = processor_fixture
        
        # Create unpicklable object (open file handle)
        code = "f = open(r'c:/Windows/win.ini', 'r')\nx = 10" # win.ini exists on windows usually
        # Mock file if needed, but 'open' is easier. 
        # Actually any unpicklable local class is easier and safer
        
        code = """
class Unpicklable:
    pass
obj = Unpicklable()
"""
        metrics = processor.process_statement(code)
        assert metrics['status'] == CacheStatus.COMPUTED
        
        # Check that it didn't crash and maybe saved what it could?
        # In current impl, if variable is unpicklable, it is skipped in payload.
        # But 'obj' is the output. 
        # If all outputs are unpicklable, entry might be saved with empty vars?
        pass

    def test_persistence_restart_simulation(self):
        """Test specifically for FileBackend: Persistence across 'restarts'.
        _PERSIST forces caching regardless of the 10 ms min-execution-time floor."""

        temp_dir = tempfile.mkdtemp()
        try:
            # Session 1
            backend1 = FileBackend(cache_dir=temp_dir)
            shell1 = MockShell()
            cash1 = Cash(backend=backend1, register_magic=False)
            magics1 = CashMagics(shell1, cash1)
            proc1 = magics1._statement_processor

            code = "x = 42"
            metrics1 = proc1.process_statement(code, annotation=_PERSIST)
            assert metrics1['status'] == CacheStatus.COMPUTED
            backend1.shutdown()

            # Session 2 (New objects, same directory)
            backend2 = FileBackend(cache_dir=temp_dir)
            shell2 = MockShell()
            cash2 = Cash(backend=backend2, register_magic=False)
            magics2 = CashMagics(shell2, cash2)
            proc2 = magics2._statement_processor

            metrics2 = proc2.process_statement(code, annotation=_PERSIST)
            assert metrics2['status'] == CacheStatus.RESTORED
            assert shell2.user_ns['x'] == 42
            
            backend2.shutdown()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

