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


class TestUpdateTrackingAfterRestoreFileDeps:
    """Test that _update_tracking_after_restore propagates file dependencies."""

    def _make_checker(self, tmp_path):
        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)
        checker.variable_lineage = {}
        checker.executed_cell_codes = {}
        checker.executed_cell_hashes = {}
        checker.executed_input_lineages = {}
        checker.executed_file_deps = {}
        return checker

    def test_file_deps_propagated_from_metadata(self, tmp_path):
        """File deps in cache metadata should be propagated to executed_file_deps."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")
        csv_path = str(csv_file)

        checker = self._make_checker(tmp_path)
        metadata = {
            'output_lineages': {'df': 'abc123'},
            'code': 'df = pd.read_csv("data.csv")',
            'source_hash': 'hash1',
            'file_dependencies': {csv_path: csv_file.stat().st_mtime},
        }
        checker._update_tracking_after_restore({'df'}, metadata, {'data_path': 'lin1'})

        assert 'df' in checker.executed_file_deps
        assert csv_path in checker.executed_file_deps['df']

    def test_file_deps_empty_when_no_file_deps_in_metadata(self, tmp_path):
        """No file deps should be propagated when metadata lacks file_dependencies."""
        checker = self._make_checker(tmp_path)
        metadata = {
            'output_lineages': {'x': 'abc123'},
            'code': 'x = 42',
            'source_hash': 'hash1',
        }
        checker._update_tracking_after_restore({'x'}, metadata, {})

        assert 'x' not in checker.executed_file_deps

    def test_file_deps_resolved_via_fallback(self, tmp_path):
        """File deps with stale paths should resolve via resolve_file_dep_path."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        import os
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            # Metadata has a non-existent absolute path; fallback resolves by basename in CWD
            stale_path = "/nonexistent/old/path/data.csv"
            checker = self._make_checker(tmp_path)
            metadata = {
                'output_lineages': {'df': 'abc123'},
                'code': 'df = pd.read_csv("data.csv")',
                'source_hash': 'hash1',
                'file_dependencies': {stale_path: 0.0},
            }
            checker._update_tracking_after_restore({'df'}, metadata, {})

            assert 'df' in checker.executed_file_deps
            # The resolved path should be the actual file, not the stale path
            resolved = next(iter(checker.executed_file_deps['df']))
            assert os.path.exists(resolved)
            assert resolved != stale_path
        finally:
            os.chdir(old_cwd)

    def test_file_deps_not_set_when_path_unresolvable(self, tmp_path):
        """Completely unresolvable paths should not pollute executed_file_deps."""
        checker = self._make_checker(tmp_path)
        metadata = {
            'output_lineages': {'df': 'abc123'},
            'code': 'df = pd.read_csv("missing.csv")',
            'source_hash': 'hash1',
            'file_dependencies': {'/no/such/file/ever_unique_xyz.csv': 0.0},
        }
        checker._update_tracking_after_restore({'df'}, metadata, {})

        # No resolved path → nothing added
        assert 'df' not in checker.executed_file_deps or len(checker.executed_file_deps['df']) == 0

    def test_file_deps_propagated_to_multiple_restored_vars(self, tmp_path):
        """When multiple vars are restored, all get the file deps."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")
        csv_path = str(csv_file)

        checker = self._make_checker(tmp_path)
        metadata = {
            'output_lineages': {'df': 'abc1', 'df2': 'abc2'},
            'code': 'df, df2 = load()',
            'source_hash': 'hash1',
            'file_dependencies': {csv_path: csv_file.stat().st_mtime},
        }
        checker._update_tracking_after_restore({'df', 'df2'}, metadata, {})

        assert csv_path in checker.executed_file_deps['df']
        assert csv_path in checker.executed_file_deps['df2']


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


class TestForwardProbePopulatesState:
    """Verify _eliminate_broken_vars_via_current_cell_probe injects
    placeholder values and lineages for resolved broken vars."""

    def _make_checker(self):
        from cash.notebook.upstream import _FORWARD_PROBE_PLACEHOLDER

        mock_shell = MagicMock()
        mock_shell.user_ns = {}
        checker = UpstreamChecker(mock_shell)
        checker.variable_lineage = {}
        checker.executed_cell_codes = {}
        checker.executed_cell_hashes = {}
        checker.executed_input_lineages = {}
        checker.executed_file_deps = {}
        checker.debug = False

        # Provide a mock cash_instance with a backend that returns a hit
        mock_backend = MagicMock()
        mock_cash = MagicMock()
        mock_cash.backend = mock_backend
        checker.cash_instance = mock_cash
        checker.shell = mock_shell

        return checker, mock_shell, mock_backend, _FORWARD_PROBE_PLACEHOLDER

    def test_placeholder_injected_into_user_ns(self):
        """When the forward probe resolves a broken var, a placeholder
        must appear in user_ns so _check_input_lineage_skip passes."""
        checker, shell, backend, PLACEHOLDER = self._make_checker()

        # Simulate a cache hit
        backend.get.return_value = ({'file_dependencies': {}}, {'variables': {'df': 'data'}})

        broken = {'df'}
        virtual_lineage = {'df': 'lineage_hash_abc'}
        cells = ["x = 10", "df['col'] = x * 2"]

        checker._eliminate_broken_vars_via_current_cell_probe(
            broken, cells, 1, virtual_lineage, set(),
        )

        # broken_vars should be empty (resolved)
        assert not broken
        # Placeholder should be in user_ns
        assert shell.user_ns.get('df') is PLACEHOLDER
        # Lineage should be set
        assert checker.variable_lineage.get('df') == 'lineage_hash_abc'

    def test_no_placeholder_when_no_cache_hit(self):
        """When there's no cache hit, nothing should be injected."""
        checker, shell, backend, PLACEHOLDER = self._make_checker()

        # Simulate a cache miss
        backend.get.return_value = (None, None)

        broken = {'df'}
        virtual_lineage = {'df': 'lineage_hash_abc'}
        cells = ["x = 10", "df['col'] = x * 2"]

        checker._eliminate_broken_vars_via_current_cell_probe(
            broken, cells, 1, virtual_lineage, set(),
        )

        # broken_vars should NOT be resolved
        assert 'df' in broken
        # No placeholder should be injected
        assert 'df' not in shell.user_ns
        assert 'df' not in checker.variable_lineage

    def test_existing_value_not_overwritten(self):
        """If user_ns already has the var, don't overwrite with placeholder."""
        checker, shell, backend, PLACEHOLDER = self._make_checker()

        existing_value = [1, 2, 3]
        shell.user_ns['df'] = existing_value

        backend.get.return_value = ({'file_dependencies': {}}, {'variables': {'df': 'data'}})

        broken = {'df'}
        virtual_lineage = {'df': 'lineage_hash_abc'}
        cells = ["x = 10", "df['col'] = x * 2"]

        checker._eliminate_broken_vars_via_current_cell_probe(
            broken, cells, 1, virtual_lineage, set(),
        )

        # broken_vars should still be resolved
        assert not broken
        # Original value should be preserved, not overwritten with placeholder
        assert shell.user_ns['df'] is existing_value
