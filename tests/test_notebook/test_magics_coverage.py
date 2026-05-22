"""
Tests for CashMagics methods that need additional coverage.

Targets: cash_badge, cash_status, _capture_cell_id.

Object-hashing helpers (`compute_hash`, `calculate_memory_size`,
`_recursive_getsizeof`) moved to `cash.notebook.object_hashing`;
their tests live in `test_object_hashing.py`.
"""
import pytest
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
