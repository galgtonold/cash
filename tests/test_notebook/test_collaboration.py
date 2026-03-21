"""Tests for %cash_stats, %cash_export, %cash_import magics."""

from pathlib import Path


class TestCashStats:
    """Test %cash_stats magic."""

    def test_stats_initial_state(self, cash_magics):
        assert cash_magics._session.stats['cells_executed'] == 0
        assert cash_magics._session.stats['statements_computed'] == 0
        assert cash_magics._session.stats['statements_restored'] == 0

    def test_stats_after_compute(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        cash_magics.cash_stats("")
        captured = capsys.readouterr()
        assert "Session Statistics" in captured.out

    def test_stats_json_format(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        # Clear any previous output
        capsys.readouterr()
        cash_magics.cash_stats("json")
        captured = capsys.readouterr()
        import json
        # The output should be valid JSON
        data = json.loads(captured.out.strip())
        assert 'hit_rate_percent' in data
        assert 'cache_entries' in data

    def test_stats_reset(self, cash_magics, capsys):
        cash_magics.cash("", "x = 42")
        cash_magics.cash_stats("reset")
        assert cash_magics._session.stats['cells_executed'] == 0
        assert cash_magics._session.stats['statements_computed'] == 0

    def test_stats_display_all_fields(self, cash_magics, capsys):
        cash_magics.cash_stats("")
        captured = capsys.readouterr()
        assert "Cells executed:" in captured.out
        assert "Cache hit rate:" in captured.out
        assert "Time saved:" in captured.out
        assert "Cache entries:" in captured.out
        assert "Tracked variables:" in captured.out


class TestCashExport:
    """Test %cash_export magic."""

    def test_export_creates_file(self, cash_magics, tmp_path):
        cash_magics.cash("", "x = 42")
        export_path = str(tmp_path / "test.cache")
        cash_magics.cash_export(export_path)
        assert Path(export_path).exists()

    def test_export_no_args(self, cash_magics, capsys):
        cash_magics.cash_export("")
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_export_success_message(self, cash_magics, tmp_path, capsys):
        cash_magics.cash("", "x = 42")
        export_path = str(tmp_path / "test.cache")
        cash_magics.cash_export(export_path)
        captured = capsys.readouterr()
        assert "Exported" in captured.out

    def test_export_import_roundtrip(self, cash_magics, cash_instance, tmp_path):
        cash_magics.cash("", "x = 42")
        export_path = str(tmp_path / "test.cache")
        cash_magics.cash_export(export_path)
        
        # Clear cache
        cash_instance.backend.clear()
        entries_before = cash_instance.backend.list_entries()
        assert len(entries_before) == 0
        
        # Import
        cash_magics.cash_import(export_path)
        
        entries_after = cash_instance.backend.list_entries()
        assert len(entries_after) > 0


class TestCashImport:
    """Test %cash_import magic."""

    def test_import_nonexistent(self, cash_magics, capsys):
        cash_magics.cash_import("/nonexistent/file.cache")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_import_no_args(self, cash_magics, capsys):
        cash_magics.cash_import("")
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_import_success_message(self, cash_magics, tmp_path, capsys):
        cash_magics.cash("", "x = 42")
        export_path = str(tmp_path / "test.cache")
        cash_magics.cash_export(export_path)
        
        # Clear and reimport
        cash_magics._cash_instance.backend.clear()
        cash_magics.cash_import(export_path)
        captured = capsys.readouterr()
        assert "Imported" in captured.out

    def test_import_merge_mode(self, cash_magics, tmp_path, capsys):
        cash_magics.cash("", "x = 42")
        export_path = str(tmp_path / "test.cache")
        cash_magics.cash_export(export_path)
        
        # Import with merge (should skip existing)
        cash_magics.cash_import(f"{export_path} --merge")
        captured = capsys.readouterr()
        # Should complete without error
        assert "Imported" in captured.out or "skipped" in captured.out
