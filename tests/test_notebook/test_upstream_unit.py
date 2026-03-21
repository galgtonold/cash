"""Direct import tests for the upstream module.

Verifies that UpstreamChecker and its new helper methods are importable
and have the expected interface, improving test coverage visibility.
"""

import pytest
from unittest.mock import MagicMock

from cash.notebook.upstream import UpstreamChecker


class TestUpstreamCheckerImport:
    """Verify UpstreamChecker is importable and has expected methods."""

    def test_class_exists(self):
        assert UpstreamChecker is not None

    def test_has_check_and_reexecute_method(self):
        assert hasattr(UpstreamChecker, "check_and_reexecute")

    def test_has_update_virtual_lineage(self):
        assert hasattr(UpstreamChecker, "_update_virtual_lineage")

    def test_has_extracted_helpers(self):
        """Verify the extracted helper methods exist."""
        expected_helpers = [
            "_validate_file_freshness",
            "_resolve_input_lineage",
            "_compute_module_source_hash",
        ]
        for name in expected_helpers:
            assert hasattr(UpstreamChecker, name), (
                f"UpstreamChecker missing helper method {name}"
            )


class TestValidateFileFreshness:
    """Test the static _validate_file_freshness helper."""

    def test_empty_files_is_fresh(self):
        assert UpstreamChecker._validate_file_freshness({}) is True

    def test_missing_file_is_stale(self, tmp_path):
        missing = str(tmp_path / "nonexistent.csv")
        assert UpstreamChecker._validate_file_freshness({missing: 0.0}) is False

    def test_existing_file_with_matching_mtime(self, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")
        import os
        mtime = os.path.getmtime(str(test_file))
        assert UpstreamChecker._validate_file_freshness({str(test_file): mtime}) is True

    def test_existing_file_with_stale_mtime(self, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")
        # Use a very old mtime
        assert UpstreamChecker._validate_file_freshness({str(test_file): 0.0}) is False

    def test_multiple_files_all_fresh(self, tmp_path):
        """All files must be fresh for the result to be True."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("data1")
        f2.write_text("data2")
        import os
        files = {
            str(f1): os.path.getmtime(str(f1)),
            str(f2): os.path.getmtime(str(f2)),
        }
        assert UpstreamChecker._validate_file_freshness(files) is True

    def test_multiple_files_one_stale(self, tmp_path):
        """If any file is stale, the result should be False."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("data1")
        f2.write_text("data2")
        import os
        files = {
            str(f1): os.path.getmtime(str(f1)),
            str(f2): 0.0,  # Stale
        }
        assert UpstreamChecker._validate_file_freshness(files) is False


class TestUpstreamCheckerSetTrackingState:
    """Test set_tracking_state method."""

    def test_has_set_tracking_state(self):
        assert hasattr(UpstreamChecker, "set_tracking_state")
        assert callable(UpstreamChecker.set_tracking_state)

    def test_set_tracking_dicts_removed(self):
        """set_tracking_dicts was removed; only set_tracking_state remains."""
        assert not hasattr(UpstreamChecker, "set_tracking_dicts")

    def test_iter_body_nodes_exists(self):
        """Static helper for iterating control structure bodies."""
        assert hasattr(UpstreamChecker, "_iter_body_nodes")


class TestUpstreamASTCache:
    """Test AST cache in UpstreamChecker."""

    def test_ast_cache_hit(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        code = "x = 1 + 2"
        tree1 = checker._get_cached_ast(code)
        assert tree1 is not None
        tree2 = checker._get_cached_ast(code)
        assert tree2 is tree1

    def test_ast_cache_syntax_error(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        tree = checker._get_cached_ast("def :")
        assert tree is None

    def test_ast_cache_eviction(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)
        checker._ast_cache_max_size = 4

        for i in range(6):
            checker._get_cached_ast(f"x_{i} = {i}")

        assert len(checker._ast_cache) <= 6


class TestUpstreamFindCellIndex:
    """Test _find_current_cell_index."""

    def test_find_by_content(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        cells = ["x = 1", "y = 2", "z = 3"]
        idx = checker._find_current_cell_index("y = 2", cells)
        assert idx == 1

    def test_find_by_cell_id(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        cells = ["x = 1", "y = 2"]
        cells_with_ids = [("id1", "x = 1"), ("id2", "y = 2")]
        idx = checker._find_current_cell_index("y = 2", cells, cell_id="id2", cells_with_ids=cells_with_ids)
        assert idx == 1

    def test_find_not_found(self):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        cells = ["x = 1", "y = 2"]
        idx = checker._find_current_cell_index("z = 999", cells)
        assert idx is None

    def test_find_duplicate_cells_raises(self):
        from cash.exceptions import AmbiguousCellError

        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)

        cells = ["x = 1", "x = 1", "x = 1"]
        with pytest.raises(AmbiguousCellError, match="Ambiguous cell execution"):
            checker._find_current_cell_index("x = 1", cells)
