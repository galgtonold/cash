"""
Tests for %cash_diff and %cash_export --json magic commands.
"""

import json
import pickle
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
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    yield magics, shell, backend
    backend.clear()


class TestCashDiff:
    """Tests for the %cash_diff magic command."""

    def test_diff_no_args(self, magics_fixture, capsys):
        """Test diff with no arguments shows usage."""
        magics, shell, backend = magics_fixture
        magics.cash_diff("")
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_diff_file_not_found(self, magics_fixture, capsys):
        """Test diff with nonexistent file."""
        magics, shell, backend = magics_fixture
        magics.cash_diff("/nonexistent/file.cache")
        output = capsys.readouterr().out
        assert "File not found" in output

    def test_diff_invalid_json(self, magics_fixture, capsys, tmp_path):
        """Test diff with invalid JSON file."""
        magics, shell, backend = magics_fixture
        bad_file = tmp_path / "bad.cache"
        bad_file.write_text("not json {{{")
        magics.cash_diff(str(bad_file))
        output = capsys.readouterr().out
        assert "Invalid cache file" in output

    def test_diff_empty_vs_empty(self, magics_fixture, capsys, tmp_path):
        """Test diff when both sessions are empty."""
        magics, shell, backend = magics_fixture
        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {},
            "cell_codes": {},
            "entries": {}
        }))

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Only in current session: 0" in output
        assert "Identical:               0" in output

    def test_diff_current_has_more(self, magics_fixture, capsys, tmp_path):
        """Test diff when current session has more variables."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x'
        magics._tracking_state.variable_lineage['y'] = 'hash_y'

        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {"x": "hash_x"},
            "cell_codes": {}
        }))

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Only in current session: 1" in output
        assert "Identical:               1" in output

    def test_diff_other_has_more(self, magics_fixture, capsys, tmp_path):
        """Test diff when other file has more variables."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x'

        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {"x": "hash_x", "y": "hash_y", "z": "hash_z"},
            "cell_codes": {}
        }))

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Only in current session: 0" in output
        assert "Only in" in output  # Other file has 2 more

    def test_diff_changed_lineage(self, magics_fixture, capsys, tmp_path):
        """Test diff detects changed variables."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x_v2'  # Changed
        magics._tracking_state.variable_lineage['y'] = 'hash_y'      # Same

        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {"x": "hash_x_v1", "y": "hash_y"},
            "cell_codes": {}
        }))

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Changed (diff lineage):  1" in output
        assert "Identical:               1" in output

    def test_diff_with_vars_flag(self, magics_fixture, capsys, tmp_path):
        """Test diff with --vars flag shows variable names."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['alpha'] = 'hash_a'
        magics._tracking_state.variable_lineage['beta'] = 'hash_b_v2'

        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {"beta": "hash_b_v1", "gamma": "hash_g"},
            "cell_codes": {}
        }))

        magics.cash_diff(f"{str(cache_file)} --vars")
        output = capsys.readouterr().out
        assert "alpha" in output
        assert "gamma" in output
        assert "beta" in output
        assert "Changed" in output

    def test_diff_all_identical(self, magics_fixture, capsys, tmp_path):
        """Test diff when everything is identical."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'h1'
        magics._tracking_state.variable_lineage['y'] = 'h2'

        cache_file = tmp_path / "other.cache"
        cache_file.write_text(json.dumps({
            "lineage": {"x": "h1", "y": "h2"},
            "cell_codes": {}
        }))

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Identical:               2" in output
        assert "Changed (diff lineage):  0" in output

    def test_diff_pickle_format(self, magics_fixture, capsys, tmp_path):
        """Test diff with pickle-format cache file (from %cash_export)."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x'
        magics._tracking_state.variable_lineage['y'] = 'hash_y'

        cache_file = tmp_path / "other.cache"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                "version": 1,
                "lineage": {"x": "hash_x", "z": "hash_z"},
                "cell_codes": {},
                "entries": []
            }, f)

        magics.cash_diff(str(cache_file))
        output = capsys.readouterr().out
        assert "Only in current session: 1" in output  # y
        assert "Identical:               1" in output    # x

    def test_diff_invalid_format(self, magics_fixture, capsys, tmp_path):
        """Test diff with completely invalid file (not JSON or pickle)."""
        magics, shell, backend = magics_fixture
        bad_file = tmp_path / "bad.bin"
        bad_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        magics.cash_diff(str(bad_file))
        output = capsys.readouterr().out
        assert "Invalid cache file" in output


class TestCashExportJson:
    """Tests for %cash_export --json."""

    def test_export_json_basic(self, magics_fixture, capsys, tmp_path):
        """Test JSON export creates valid JSON with lineage."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x'
        magics._tracking_state.variable_lineage['y'] = 'hash_y'
        magics._tracking_state.executed_cell_codes['x'] = 'x = 1'
        magics._tracking_state.executed_cell_codes['y'] = 'y = 2'

        out_file = tmp_path / "export.json"
        magics.cash_export(f"{str(out_file)} --json")
        output = capsys.readouterr().out
        assert "Exported lineage for 2 variables" in output
        assert "JSON" in output

        # Verify file is valid JSON
        with open(out_file) as f:
            data = json.load(f)
        assert data['lineage']['x'] == 'hash_x'
        assert data['lineage']['y'] == 'hash_y'
        assert data['cell_codes']['x'] == 'x = 1'

    def test_export_json_with_vars(self, magics_fixture, capsys, tmp_path):
        """Test JSON export with --vars filter."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['x'] = 'hash_x'
        magics._tracking_state.variable_lineage['y'] = 'hash_y'
        magics._tracking_state.variable_lineage['z'] = 'hash_z'
        magics._tracking_state.executed_cell_codes['x'] = 'x = 1'

        out_file = tmp_path / "export.json"
        magics.cash_export(f"{str(out_file)} --json --vars x,z")
        capsys.readouterr()

        with open(out_file) as f:
            data = json.load(f)
        assert 'x' in data['lineage']
        assert 'z' in data['lineage']
        assert 'y' not in data['lineage']

    def test_export_json_then_diff(self, magics_fixture, capsys, tmp_path):
        """Test full round-trip: export JSON, then diff."""
        magics, shell, backend = magics_fixture
        magics._tracking_state.variable_lineage['a'] = 'h1'
        magics._tracking_state.variable_lineage['b'] = 'h2'

        out_file = tmp_path / "export.json"
        magics.cash_export(f"{str(out_file)} --json")
        capsys.readouterr()

        # Modify lineage (simulate change)
        magics._tracking_state.variable_lineage['b'] = 'h2_changed'
        magics._tracking_state.variable_lineage['c'] = 'h3'

        magics.cash_diff(str(out_file))
        output = capsys.readouterr().out
        assert "Only in current session: 1" in output  # c
        assert "Changed (diff lineage):  1" in output    # b
        assert "Identical:               1" in output     # a
