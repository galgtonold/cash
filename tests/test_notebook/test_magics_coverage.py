"""
Tests for CashMagics methods that need additional coverage.

Targets: cash_badge, cash_status, _capture_cell_id, _compute_hash,
         _calculate_memory_size, _recursive_getsizeof
"""
import pytest
import sys
from unittest.mock import MagicMock

from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend
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
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


# ============================================================================
# cash_badge magic
# ============================================================================

class TestCashBadge:
    """Test %cash_badge magic command."""

    def test_set_badge_html(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics.cash_badge("html")
        assert magics._badge_mode == "html"
        captured = capsys.readouterr()
        assert "Badge mode set to: html" in captured.out

    def test_set_badge_print(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics.cash_badge("print")
        assert magics._badge_mode == "print"
        captured = capsys.readouterr()
        assert "Badge mode set to: print" in captured.out

    def test_set_badge_off(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics.cash_badge("off")
        assert magics._badge_mode == "off"
        captured = capsys.readouterr()
        assert "Badge mode set to: off" in captured.out

    def test_badge_invalid_shows_current(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._badge_mode = "print"
        magics.cash_badge("invalid_mode")
        captured = capsys.readouterr()
        assert "Current badge mode: print" in captured.out
        assert "Usage:" in captured.out

    def test_badge_empty_shows_current(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._badge_mode = "html"
        magics.cash_badge("")
        captured = capsys.readouterr()
        assert "Current badge mode: html" in captured.out


# ============================================================================
# cash_status magic
# ============================================================================

class TestCashStatus:
    """Test %cash_status magic command."""

    def test_status_print_mode(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        result = magics.cash_status("")
        assert isinstance(result, dict)
        assert 'lineage' in result
        assert 'auto_cache_enabled' in result
        assert 'debug_enabled' in result
        captured = capsys.readouterr()
        assert captured.out.strip()  # Should print something

    def test_status_dict_mode(self, magics_fixture):
        magics, _, _ = magics_fixture
        result = magics.cash_status("dict")
        assert isinstance(result, dict)
        assert 'last_cell' in result
        assert 'lineage' in result
        assert 'cache_stats' in result

    def test_status_json_mode(self, magics_fixture):
        magics, _, _ = magics_fixture
        result = magics.cash_status("json")
        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert 'lineage' in parsed

    def test_status_reflects_execution(self, magics_fixture):
        """After executing a statement, status should reflect it."""
        magics, shell, _ = magics_fixture
        processor = magics._statement_processor
        processor.process_statement("x = 42")
        result = magics.cash_status("dict")
        assert 'x' in result['executed_codes']


# ============================================================================
# _capture_cell_id
# ============================================================================

class TestCaptureCellId:
    """Test _capture_cell_id method."""

    def test_capture_from_info_cell_id(self, magics_fixture):
        magics, _, _ = magics_fixture
        info = MagicMock()
        info.cell_id = "test-cell-123"
        magics._capture_cell_id(info)
        assert magics._current_cell_id == "test-cell-123"

    def test_capture_from_vscode_metadata(self, magics_fixture):
        magics, shell, _ = magics_fixture
        info = MagicMock(spec=[])  # No cell_id attribute
        # Simulate VS Code parent header
        shell.get_parent = MagicMock(return_value={
            'metadata': {
                'vscode': {'cellId': 'vscode-cell-456'}
            }
        })
        magics._capture_cell_id(info)
        assert magics._current_cell_id == "vscode-cell-456"

    def test_capture_from_parent_metadata_cellId(self, magics_fixture):
        magics, shell, _ = magics_fixture
        info = MagicMock(spec=[])
        shell.get_parent = MagicMock(return_value={
            'metadata': {'cellId': 'parent-cell-789'}
        })
        magics._capture_cell_id(info)
        assert magics._current_cell_id == "parent-cell-789"

    def test_capture_no_cell_id_available(self, magics_fixture):
        magics, _, _ = magics_fixture
        info = MagicMock(spec=[])
        magics._capture_cell_id(info)
        assert magics._current_cell_id is None

    def test_capture_exception_handled(self, magics_fixture):
        """Exceptions in capture_cell_id should not propagate."""
        magics, shell, _ = magics_fixture
        # Create an info object where accessing cell_id raises
        info = MagicMock(spec=[])  # No cell_id
        # Make shell.get_parent raise an exception
        shell.get_parent = MagicMock(side_effect=RuntimeError("test error"))
        # This should not raise
        magics._capture_cell_id(info)
        assert magics._current_cell_id is None

    def test_capture_debug_output(self, magics_fixture, capsys):
        magics, _, _ = magics_fixture
        magics._debug = True
        info = MagicMock()
        info.cell_id = "debug-cell"
        magics._capture_cell_id(info)
        captured = capsys.readouterr()
        assert "[CELL_ID]" in captured.out
        assert "debug-cell" in captured.out


# ============================================================================
# _compute_hash
# ============================================================================

class TestComputeHash:
    """Test _compute_hash method."""

    def test_hash_simple_objects(self, magics_fixture):
        magics, _, _ = magics_fixture
        h1 = magics._compute_hash(42)
        h2 = magics._compute_hash(42)
        h3 = magics._compute_hash(43)
        assert h1 == h2  # Same value, same hash
        assert h1 != h3  # Different value, different hash

    def test_hash_string(self, magics_fixture):
        magics, _, _ = magics_fixture
        h = magics._compute_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_list(self, magics_fixture):
        magics, _, _ = magics_fixture
        h = magics._compute_hash([1, 2, 3])
        assert isinstance(h, str)

    def test_hash_dict(self, magics_fixture):
        magics, _, _ = magics_fixture
        h = magics._compute_hash({"a": 1, "b": 2})
        assert isinstance(h, str)

    def test_hash_dataframe(self, magics_fixture):
        """DataFrame should use fast shape+dtypes hashing."""
        magics, _, _ = magics_fixture
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        h = magics._compute_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_numpy_array(self, magics_fixture):
        """ndarray should use shape+dtype hashing."""
        magics, _, _ = magics_fixture
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.array([1, 2, 3, 4, 5])
        h = magics._compute_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_series(self, magics_fixture):
        magics, _, _ = magics_fixture
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        s = pd.Series([1, 2, 3], name='test')
        h = magics._compute_hash(s)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_unpicklable_object(self, magics_fixture):
        """Unpicklable objects should fall back to id-based hash."""
        magics, _, _ = magics_fixture
        import threading
        lock = threading.Lock()
        h = magics._compute_hash(lock)
        assert isinstance(h, str)


# ============================================================================
# _calculate_memory_size
# ============================================================================

class TestCalculateMemorySize:
    """Test _calculate_memory_size method."""

    def test_simple_types(self, magics_fixture):
        magics, _, _ = magics_fixture
        size = magics._calculate_memory_size({'x': 42, 'y': 'hello'})
        assert size > 0

    def test_dataframe_memory(self, magics_fixture):
        magics, _, _ = magics_fixture
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({'a': range(1000), 'b': range(1000)})
        size = magics._calculate_memory_size({'df': df})
        assert size > 1000  # Should be at least a few KB

    def test_numpy_array_memory(self, magics_fixture):
        magics, _, _ = magics_fixture
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.zeros(10000)
        size = magics._calculate_memory_size({'arr': arr})
        assert size >= 80000  # 10000 * 8 bytes per float64

    def test_series_memory(self, magics_fixture):
        magics, _, _ = magics_fixture
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        s = pd.Series(range(1000))
        size = magics._calculate_memory_size({'s': s})
        assert size > 0

    def test_empty_dict(self, magics_fixture):
        magics, _, _ = magics_fixture
        size = magics._calculate_memory_size({})
        assert size == 0

    def test_nested_containers(self, magics_fixture):
        magics, _, _ = magics_fixture
        data = {'nested': {'a': [1, 2, 3], 'b': {'c': [4, 5]}}}
        size = magics._calculate_memory_size(data)
        assert size > 0


# ============================================================================
# _recursive_getsizeof
# ============================================================================

class TestRecursiveGetsizeof:
    """Test _recursive_getsizeof method."""

    def test_simple_int(self, magics_fixture):
        magics, _, _ = magics_fixture
        size = magics._recursive_getsizeof(42)
        assert size > 0

    def test_list_includes_elements(self, magics_fixture):
        magics, _, _ = magics_fixture
        list_size = magics._recursive_getsizeof([1, 2, 3])
        int_size = magics._recursive_getsizeof(42)
        assert list_size > int_size  # List should be bigger than one element

    def test_dict_includes_keys_and_values(self, magics_fixture):
        magics, _, _ = magics_fixture
        size = magics._recursive_getsizeof({'key': 'value', 'key2': 'value2'})
        assert size > sys.getsizeof({})

    def test_handles_circular_reference(self, magics_fixture):
        """Should handle circular references without infinite recursion."""
        magics, _, _ = magics_fixture
        lst = [1, 2, 3]
        lst.append(lst)  # Circular reference
        size = magics._recursive_getsizeof(lst)
        assert size > 0  # Should not hang or crash
