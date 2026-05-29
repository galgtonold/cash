"""
Pytest configuration and fixtures for test isolation.

This module provides comprehensive fixtures for testing Cash functionality
with proper isolation between tests.
"""
import pytest
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock
from io import StringIO
import sys

from cash import Cash
from cash.backends import InMemoryBackend, FileBackend
from cash.notebook.ipython.magics import CashMagics
from traitlets.config import Configurable


# ============================================================================
# Backend Fixtures
# ============================================================================

@pytest.fixture
def clean_backend():
    """Provide a fresh InMemoryBackend for each test."""
    backend = InMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a unique temporary directory for file-based caches."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    yield str(cache_dir)
    # Cleanup with retry for Windows file locking
    max_retries = 3
    for i in range(max_retries):
        try:
            shutil.rmtree(cache_dir, ignore_errors=False)
            break
        except (PermissionError, OSError):
            if i < max_retries - 1:
                time.sleep(0.1)


@pytest.fixture
def file_backend(temp_cache_dir):
    """Provide a FileBackend with unique temp directory."""
    backend = FileBackend(cache_dir=temp_cache_dir)
    yield backend
    backend.clear()


# ============================================================================
# Cash Instance Fixtures
# ============================================================================

@pytest.fixture
def cash_instance(clean_backend):
    """Provide a fresh Cash instance with InMemoryBackend."""
    cash = Cash(backend=clean_backend, register_magic=False)
    yield cash
    cash.backend.clear()


@pytest.fixture
def cash_with_file_backend(file_backend):
    """Provide a Cash instance with FileBackend."""
    cash = Cash(backend=file_backend, register_magic=False)
    yield cash
    cash.backend.clear()


# ============================================================================
# IPython Shell Mock Fixtures
# ============================================================================

class MockShell(Configurable):
    """Mock IPython shell with all required attributes."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_ns = {}
        self.user_ns['_ih'] = []  # Input history
        self.events = MagicMock()
        self.events.register = MagicMock(return_value=None)
        self.ast_transformers = []
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()
        self.magics_manager = MagicMock()
        self.magics_manager.magics = {'cell': {}, 'line': {}}
        
    def reset(self):
        """Reset shell state."""
        self.user_ns.clear()
        self.user_ns['_ih'] = []


@pytest.fixture
def mock_shell():
    """Provide a mock IPython shell."""
    shell = MockShell()
    
    # Configure run_cell to execute code in user_ns
    def run_cell_impl(cell):
        try:
            exec(cell, {}, shell.user_ns)
            result = MagicMock()
            result.success = True
            return result
        except Exception as e:
            result = MagicMock()
            result.success = False
            result.error_in_exec = e
            return result
    
    shell.run_cell.side_effect = run_cell_impl
    yield shell
    shell.reset()


@pytest.fixture
def simple_mock_shell():
    """Provide a simple mock shell without exec functionality."""
    shell = MagicMock()
    shell.user_ns = {}
    shell.events = MagicMock()
    shell.ast_transformers = []
    shell.input_transformers_cleanup = []
    shell.run_cell = MagicMock()
    return shell


# ============================================================================
# CashMagics Fixtures
# ============================================================================

@pytest.fixture
def cash_magics(mock_shell, cash_instance):
    """Provide CashMagics instance with mock shell and clean backend."""
    magics = CashMagics(mock_shell, cash_instance)
    yield magics
    # Cleanup
    cash_instance.backend.clear()
    mock_shell.reset()


# ============================================================================
# Isolation and Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """
    Automatically isolate each test from side effects.
    Runs before every test automatically.
    """
    # Store original sys.modules to detect imports
    original_modules = set(sys.modules.keys())
    
    yield
    
    # Cleanup: Remove any new modules that were imported during test
    # This prevents module-level state from leaking between tests
    new_modules = set(sys.modules.keys()) - original_modules
    for module in new_modules:
        if module.startswith('test_') or 'cash' not in module:
            # Don't remove test modules or non-cash modules
            continue


@pytest.fixture(autouse=True)
def disable_auto_magic_registration(monkeypatch):
    """
    Disable automatic magic registration during tests.
    This prevents Cash() from trying to access get_ipython() and register magics,
    which can cause TraitErrors or warnings when running tests.
    Tests that need magics should register them manually or use the cash_magics fixture.
    """
    from cash.core import Cash
    monkeypatch.setattr(Cash, 'register_magic', lambda self: None)


@pytest.fixture
def isolated_test(monkeypatch, tmp_path):
    """
    Provide complete test isolation:
    - Clean working directory
    - Isolated environment
    """
    original_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    monkeypatch.chdir(original_cwd)


# ============================================================================
# Output Capture Fixtures
# ============================================================================

@pytest.fixture
def captured_output():
    """Capture stdout and stderr during test execution."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    yield stdout_capture, stderr_capture
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Provide a sample pandas DataFrame for testing."""
    try:
        import pandas as pd
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
    except ImportError:
        pytest.skip("pandas not installed")


@pytest.fixture
def sample_data():
    """Provide sample data for caching tests."""
    return {
        'integers': [1, 2, 3, 4, 5],
        'strings': ['a', 'b', 'c'],
        'nested': {'key1': 'value1', 'key2': [1, 2, 3]}
    }


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ipython: marks tests that require IPython"
    )
    # CashImpurityWarning fires on the call-counter pattern that
    # virtually every test uses (`n['calls'] += 1` to count invocations
    # — a real scope mutation). Treating it as noise for the broad
    # suite; dedicated purity tests opt back in with their own filter.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::cash.CashImpurityWarning",
    )


@pytest.fixture(autouse=True)
def cleanup_sys_modules_between_tests():
    """
    Clean up IPython mocks from sys.modules between tests.

    Many unit test files mock IPython modules at import time, polluting
    sys.modules. This fixture removes those mocks after each test to
    prevent cross-test contamination.
    """
    yield  # Run the test

    import sys
    from unittest.mock import MagicMock
    import types

    ipython_modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('IPython'):
            module = sys.modules[module_name]
            if isinstance(module, MagicMock) or isinstance(module, types.ModuleType) and (not hasattr(module, '__file__') or module.__file__ is None):
                ipython_modules_to_remove.append(module_name)

    for module_name in ipython_modules_to_remove:
        del sys.modules[module_name]




