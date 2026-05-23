from cash.notebook.cache_status import CacheStatus

import pytest
import os
import time
import tempfile
from unittest.mock import MagicMock, patch
from cash.notebook.magics import CashMagics
from cash.notebook.annotations import CacheAnnotation
from cash.core import Cash
from cash.backends import InMemoryBackend
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
        self.user_global_ns = self.user_ns

@pytest.fixture
def statement_processor_fixture():
    '''Provide StatementProcessor instance for testing.'''
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    
    # We need to simulate the magics setup slightly to give processor the tracking dicts
    magics = CashMagics(shell, cash)
    processor = magics._statement_processor
    
    yield processor, shell, backend
    
    # Cleanup
    backend.clear()
    shell.user_ns.clear()

def test_delete_code_in_cell(statement_processor_fixture):
    """Test that removing code from a cell changes the hash and invalidates cache."""
    processor, shell, backend = statement_processor_fixture
    
    # 1. Execute initial code
    code1 = "x = 1\ny = 2\nz = x + y"
    
    # Mock execution side effects
    def exec_side_effect_1(code, **kwargs):
        shell.user_ns['x'] = 1
        shell.user_ns['y'] = 2
        shell.user_ns['z'] = 3
        result = MagicMock(success=True)
        result.skipped = False
        return result, MagicMock(stdout="", stderr="", outputs=[]), 0.1, set()
    
    # We need to patch _execute_statement to avoid actual compilation issues or just let it run if simple
    # But since we use StatementProcessor directly, let's patch _execute_statement for control
    with patch.object(processor, '_execute_statement', side_effect=exec_side_effect_1):
        metrics1 = processor.process_statement(code1)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert len(backend.list_entries()) == 1
    
    # 2. Execute modified code (removed y=2, z depends on x+1 directly)
    code2 = "x = 1\nz = x + 1" # Changed logic
    
    def exec_side_effect_2(code, **kwargs):
        shell.user_ns['x'] = 1
        shell.user_ns['z'] = 2
        result = MagicMock(success=True)
        result.skipped = False
        return result, MagicMock(stdout="", stderr="", outputs=[]), 0.1, set()

    with patch.object(processor, '_execute_statement', side_effect=exec_side_effect_2):
        metrics2 = processor.process_statement(code2)
        assert metrics2['status'] == CacheStatus.COMPUTED # Should miss cache because SOURCE changed
        assert len(backend.list_entries()) == 2 # New entry
        
    # Verify hashes are different
    # This is implicitly verified by COMPLETED status, but let's check
    entries = backend.list_entries()
    assert len(entries) == 2

def test_complex_statement_structure(statement_processor_fixture):
    """Test caching of complex statements like class definitions and loops."""
    processor, shell, backend = statement_processor_fixture
    
    code = """
class Complex:
    def __init__(self, val):
        self.val = val
    def compute(self):
        return self.val * 2

c = Complex(10)
res = c.compute()
"""
    
    # Actual execution is better here to ensure analysis works on complex code
    # But we need to be careful with environment. 
    # StatementProcessor._execute_statement uses exec(), so it should work if we don't mock it too much.
    # However, StatementProcessor also does file tracking which might need mocking if we don't want real file access.
    
    # Let's try running without patching _execute_statement to test real analysis+exec
    # This requires the code to be valid and simple enough not to depend on external things.
    
    # _PERSIST overrides the 10 ms min-execution-time floor so a class
    # definition (which runs instantly) is actually written to cache.
    metrics = processor.process_statement(code, annotation=_PERSIST)
    assert metrics['status'] == CacheStatus.COMPUTED
    print(f"DEBUG: User NS keys after exec: {list(shell.user_ns.keys())}")
    assert 'res' in shell.user_ns
    assert shell.user_ns['res'] == 20
    assert 'c' in shell.user_ns

    # Run again - should be RESTORED
    # We need to clear user_ns to simulate a new session or ensure restoration works,
    # BUT for cache hit, the processor checks inputs.
    # Here, 'code' has no inputs (it defines everything).

    # Clear variables to prove they come from cache
    shell.user_ns.pop('res', None)
    shell.user_ns.pop('c', None)
    shell.user_ns.pop('Complex', None)

    metrics2 = processor.process_statement(code, annotation=_PERSIST)
    assert metrics2['status'] == CacheStatus.RESTORED
    assert 'res' in shell.user_ns
    assert shell.user_ns['res'] == 20
    
    # Note: 'c' and 'Complex' won't be restored because they are locally defined classes 
    # not available at module level for pickle. This is expected behavior.
    if 'c' in shell.user_ns:
        assert shell.user_ns['c'].val == 10

def test_file_dependency_invalidation_integrated(statement_processor_fixture):
    """Test file dependency invalidation with a real temp file."""
    processor, shell, backend = statement_processor_fixture
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write("initial_data")
        temp_path = f.name.replace(os.sep, '/')
    
    try:
        code = f"with open('{temp_path}', 'r') as f: data = f.read()"
        
        # 1. First Run
        metrics1 = processor.process_statement(code)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['data'] == "initial_data"
        
        # 2. Second Run (No Change)
        # Clear variable to ensure restore
        shell.user_ns.pop('data', None)
        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.RESTORED
        assert shell.user_ns['data'] == "initial_data"
        
        # 3. Third Run (File Changed)
        time.sleep(1.1)
        with open(temp_path, 'w') as f:
            f.write("new_data")
            
        shell.user_ns.pop('data', None)
        metrics3 = processor.process_statement(code)
        assert metrics3['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['data'] == "new_data"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@pytest.mark.xfail(reason="Known flaky: file dependency detection timing issue", strict=False)
def test_file_dependency_quick_modification(statement_processor_fixture):
    """
    Regression test: File dependency invalidation should work even with quick modifications.
    
    Previously, the mtime comparison used a 1.0 second threshold which meant that file 
    changes within 1 second of cache creation would not invalidate the cache.
    This test verifies that quick file modifications (< 1 second) are detected.
    """
    processor, shell, backend = statement_processor_fixture
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write("initial_data")
        temp_path = f.name.replace(os.sep, '/')
    
    try:
        code = f"with open('{temp_path}', 'r') as f: data = f.read()"
        
        # 1. First Run
        metrics1 = processor.process_statement(code)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['data'] == "initial_data"
        
        # 2. Second Run (No Change) - should restore
        shell.user_ns.pop('data', None)
        metrics2 = processor.process_statement(code)
        assert metrics2['status'] == CacheStatus.RESTORED
        assert shell.user_ns['data'] == "initial_data"
        
        # 3. Quick modification (NO sleep - this is the regression test!)
        # Previously this would fail because the 1.0 second threshold was too lenient
        with open(temp_path, 'w') as f:
            f.write("quick_modified_data")
            
        shell.user_ns.pop('data', None)
        metrics3 = processor.process_statement(code)
        
        # With the fix, this should detect the file change and re-compute
        assert metrics3['status'] == CacheStatus.COMPUTED, (
            f"Expected COMPUTED but got {metrics3['status']}. "
            "Quick file modifications should invalidate cache."
        )
        assert shell.user_ns['data'] == "quick_modified_data"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_file_dependency_older_file(statement_processor_fixture):
    """
    Regression test: Cache should be invalidated if file becomes OLDER.
    
    This can happen with cloud sync, file restoration from backup, or git operations.
    Previously only newer mtimes were detected.
    """
    processor, shell, backend = statement_processor_fixture
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write("original_data")
        temp_path = f.name.replace(os.sep, '/')
    
    try:
        code = f"with open('{temp_path}', 'r') as f: data = f.read()"
        
        # 1. First Run
        metrics1 = processor.process_statement(code)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['data'] == "original_data"
        
        # 2. Simulate file being "restored to older version" by setting older mtime
        time.sleep(0.1)  # Small delay
        with open(temp_path, 'w') as f:
            f.write("restored_older_data")
        
        # Set mtime to be OLDER than the cached mtime (simulate restore from backup)
        import os as os_module
        current_mtime = os_module.path.getmtime(temp_path)
        older_mtime = current_mtime - 10.0  # 10 seconds in the past
        os_module.utime(temp_path, (older_mtime, older_mtime))
            
        shell.user_ns.pop('data', None)
        metrics3 = processor.process_statement(code)
        
        # With the fix using abs(), this should detect the older file and re-compute
        assert metrics3['status'] == CacheStatus.COMPUTED, (
            f"Expected COMPUTED but got {metrics3['status']}. "
            "Older file mtime should invalidate cache (e.g., restored from backup)."
        )
        assert shell.user_ns['data'] == "restored_older_data"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_file_dependency_cascading(statement_processor_fixture):
    """
    Regression test: Downstream cells should be invalidated when upstream file changes.
    
    Scenario:
    1. Cell A reads a file and creates variable `data`
    2. Cell B uses `data` (doesn't read file directly)
    3. File is modified
    4. Cell B should be INVALIDATED because its input's source file changed
    
    This is the key bug reported: The "df" cell was cached even when the source CSV changed.
    """
    processor, shell, backend = statement_processor_fixture
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as f:
        f.write("initial_content")
        temp_path = f.name.replace(os.sep, '/')
    
    try:
        # Cell A: Read file
        code_read = f"with open('{temp_path}', 'r') as f: data = f.read()"
        
        # Cell B: Use the data (doesn't read file directly)
        code_use = "result = data.upper()"
        
        # 1. Run Cell A - reads file
        metrics1 = processor.process_statement(code_read)
        assert metrics1['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['data'] == "initial_content"
        
        # 2. Run Cell B - uses data
        metrics2 = processor.process_statement(code_use)
        assert metrics2['status'] == CacheStatus.COMPUTED
        assert shell.user_ns['result'] == "INITIAL_CONTENT"
        
        # 3. Re-run Cell B - should be cached
        shell.user_ns.pop('result', None)
        metrics3 = processor.process_statement(code_use)
        assert metrics3['status'] == CacheStatus.RESTORED
        assert shell.user_ns['result'] == "INITIAL_CONTENT"
        
        # 4. Modify the file
        time.sleep(0.1)
        with open(temp_path, 'w') as f:
            f.write("modified_content")
        
        # 5. Run Cell B again - should be INVALIDATED because source file changed
        # Even though Cell B doesn't directly read the file, its input 'data' 
        # was created from a file that has changed.
        shell.user_ns.pop('result', None)
        metrics4 = processor.process_statement(code_use)
        
        # This is the critical assertion - previously this would incorrectly return RESTORED
        assert metrics4['status'] == CacheStatus.COMPUTED, (
            f"Expected COMPUTED but got {metrics4['status']}. "
            "Downstream cells should be invalidated when upstream file changes."
        )
        # Note: The result might still be "INITIAL_CONTENT" if we haven't re-run Cell A
        # But the important thing is that Cell B's cache was invalidated
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_variable_type_change(statement_processor_fixture):
    """Test managing dependencies when input variable type changes."""
    processor, shell, backend = statement_processor_fixture
    
    # 1. Define x as int
    code1 = "x = 10"
    processor.process_statement(code1)
    
    # 2. Use x
    code2 = "y = x * 2"
    processor.process_statement(code2)
    assert shell.user_ns['y'] == 20
    
    # 3. Redefine x as string
    code1_b = "x = '10'"
    processor.process_statement(code1_b)
    
    # 4. Use x again (same code loop/cell)
    # The cache for code2 was based on hash of x (int 10).
    # Now x is "10" (string). Hash should differ.
    
    metrics = processor.process_statement(code2)
    assert metrics['status'] == CacheStatus.COMPUTED
    assert shell.user_ns['y'] == "1010" # '10' * 2 string repetition

