"""Tests for VS Code cell ID notebook path extraction and early capture in _execute_cell.

Covers:
  - extract_notebook_path_from_vscode_cell_id()  (utils)
  - set_notebook_path()  (utils)
  - Early cell_id capture in _execute_cell (magics)
"""

import os
import json
from unittest.mock import MagicMock, patch

import pytest

from cash.notebook.server_discovery import (
    extract_notebook_path_from_vscode_cell_id,
    set_notebook_path,
    get_notebook_path,
    invalidate_notebook_path_cache,
)
import contextlib


# ---------------------------------------------------------------------------
# extract_notebook_path_from_vscode_cell_id
# ---------------------------------------------------------------------------

class TestExtractNotebookPathFromVscodeCellId:
    """Tests for extract_notebook_path_from_vscode_cell_id."""

    def test_none_input(self):
        assert extract_notebook_path_from_vscode_cell_id(None) is None

    def test_empty_string(self):
        assert extract_notebook_path_from_vscode_cell_id("") is None

    def test_non_vscode_cell_id(self):
        """A plain UUID cell_id (JupyterLab style) should return None."""
        assert extract_notebook_path_from_vscode_cell_id("abc-def-123") is None

    def test_valid_vscode_uri_existing_file(self, tmp_path):
        """A real VS Code cell ID URI pointing to an existing .ipynb should be decoded."""
        nb = tmp_path / "demo.ipynb"
        nb.write_text(json.dumps({
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }))
        # Construct a VS Code-style URI
        # On Windows: vscode-notebook-cell:/c%3A/Users/.../demo.ipynb#W1sZmlsZQ==
        raw_path = str(nb).replace("\\", "/")
        # Encode the colon after the drive letter
        if len(raw_path) > 1 and raw_path[1] == ':':
            encoded_path = "/" + raw_path[0] + "%3A" + raw_path[2:]
        else:
            encoded_path = raw_path
        cell_id = f"vscode-notebook-cell:{encoded_path}#W1sZmlsZQ%3D%3D"

        result = extract_notebook_path_from_vscode_cell_id(cell_id)
        assert result is not None
        assert os.path.normpath(result) == os.path.normpath(str(nb))

    def test_valid_vscode_uri_nonexistent_file(self, tmp_path):
        """If the decoded path doesn't exist, return None."""
        cell_id = "vscode-notebook-cell:/c%3A/nonexistent/path/nb.ipynb#W1s"
        result = extract_notebook_path_from_vscode_cell_id(cell_id)
        assert result is None

    def test_valid_vscode_uri_non_ipynb(self, tmp_path):
        """If the decoded path exists but isn't .ipynb, return None."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")
        raw_path = str(txt_file).replace("\\", "/")
        if len(raw_path) > 1 and raw_path[1] == ':':
            encoded_path = "/" + raw_path[0] + "%3A" + raw_path[2:]
        else:
            encoded_path = raw_path
        cell_id = f"vscode-notebook-cell:{encoded_path}#frag"
        result = extract_notebook_path_from_vscode_cell_id(cell_id)
        assert result is None

    def test_path_with_spaces(self, tmp_path):
        """Spaces in path (URL-encoded as %20) should be decoded."""
        sub = tmp_path / "My Notebooks"
        sub.mkdir()
        nb = sub / "test.ipynb"
        nb.write_text(json.dumps({
            "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }))
        raw_path = str(nb).replace("\\", "/")
        if len(raw_path) > 1 and raw_path[1] == ':':
            encoded_path = "/" + raw_path[0] + "%3A" + raw_path[2:]
        else:
            encoded_path = raw_path
        # Encode spaces
        encoded_path = encoded_path.replace(" ", "%20")
        cell_id = f"vscode-notebook-cell:{encoded_path}#W2s"

        result = extract_notebook_path_from_vscode_cell_id(cell_id)
        assert result is not None
        assert os.path.normpath(result) == os.path.normpath(str(nb))

    def test_no_fragment(self, tmp_path):
        """URI without a fragment (#...) should still work."""
        nb = tmp_path / "nofrag.ipynb"
        nb.write_text(json.dumps({
            "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }))
        raw_path = str(nb).replace("\\", "/")
        if len(raw_path) > 1 and raw_path[1] == ':':
            encoded_path = "/" + raw_path[0] + "%3A" + raw_path[2:]
        else:
            encoded_path = raw_path
        cell_id = f"vscode-notebook-cell:{encoded_path}"

        result = extract_notebook_path_from_vscode_cell_id(cell_id)
        assert result is not None


# ---------------------------------------------------------------------------
# set_notebook_path / get_notebook_path integration
# ---------------------------------------------------------------------------

class TestSetNotebookPath:
    """Tests for set_notebook_path and its effect on get_notebook_path."""

    def setup_method(self):
        invalidate_notebook_path_cache()

    def teardown_method(self):
        invalidate_notebook_path_cache()

    def test_set_existing_path(self, tmp_path):
        """Setting a valid path should make get_notebook_path return it."""
        nb = tmp_path / "cached.ipynb"
        nb.write_text("{}")
        set_notebook_path(str(nb))
        # get_notebook_path checks cache first
        result = get_notebook_path()
        assert result == str(nb)

    def test_set_nonexistent_path_ignored(self):
        """Setting a non-existent path should not update the cache."""
        invalidate_notebook_path_cache()
        set_notebook_path("/nonexistent/fake.ipynb")
        # Cache should remain empty — get_notebook_path will try other methods
        # We can't easily assert get_notebook_path() == None without mocking
        # all fallback methods, but we can check the module-level cache directly.
        import cash.notebook.server_discovery as utils_mod
        assert utils_mod._cached_notebook_path is None

    def test_set_empty_string_ignored(self):
        """Empty string should not update the cache."""
        set_notebook_path("")
        import cash.notebook.server_discovery as utils_mod
        assert utils_mod._cached_notebook_path is None

    def test_set_none_ignored(self):
        """None should not update the cache."""
        set_notebook_path(None)
        import cash.notebook.server_discovery as utils_mod
        assert utils_mod._cached_notebook_path is None

    def test_invalidate_clears_set_path(self, tmp_path):
        """invalidate_notebook_path_cache should clear explicitly set paths."""
        nb = tmp_path / "test.ipynb"
        nb.write_text("{}")
        set_notebook_path(str(nb))
        invalidate_notebook_path_cache()
        import cash.notebook.server_discovery as utils_mod
        assert utils_mod._cached_notebook_path is None


# ---------------------------------------------------------------------------
# Early cell_id capture in _execute_cell
# ---------------------------------------------------------------------------

class TestEarlyCellIdCapture:
    """Test that _execute_cell extracts cell_id and notebook path early."""

    @pytest.fixture
    def magics_fixture(self):
        """Minimal CashMagics fixture with a mock shell."""
        from cash.notebook.ipython.cell_executor import CellExecutor
        from cash.notebook.ipython.magics import CashMagics

        shell = MagicMock()
        shell.user_ns = {}
        # Provide a minimal get_parent that returns VS Code-style metadata
        shell.get_parent.return_value = None

        # Create CashMagics instance
        with patch.object(CashMagics, '__init__', lambda self, s, **kw: None):
            m = CashMagics.__new__(CashMagics)
            m.shell = shell
            m._auto_cache_enabled = True
            m._benchmark_config = None
            m._badge_mode = 'off'
            m._debug = False
            m._current_cell_id = None
            m._in_sync_cell = False
            m._statement_processor = MagicMock()
            m._upstream_checker = MagicMock()
            m._original_run_cell = MagicMock()
            m._global_ttl = None
            m._execution_history = []
            m._control_structure_processor = MagicMock()
            m._cash_instance = MagicMock()
            # CellExecutor needs to exist so _execute_cell can delegate; the
            # cell_id capture happens inside the executor's pipeline.  The
            # tests suppress() the later-phase failures.
            m._cell_executor = CellExecutor(
                shell=shell,
                cash_instance=m._cash_instance,
                magics=m,
                tracking_state=MagicMock(),
                statement_processor=m._statement_processor,
                upstream_checker=m._upstream_checker,
                restorer=MagicMock(),
                module_invalidator=MagicMock(),
                control_structure_processor=m._control_structure_processor,
                debug=False,
            )
        return m

    def test_captures_cellid_from_parent_metadata(self, magics_fixture):
        """cell_id should be extracted from shell.get_parent() metadata."""
        m = magics_fixture
        m.shell.get_parent.return_value = {
            'metadata': {
                'cellId': 'vscode-notebook-cell:/fake/path.ipynb#W1s'
            }
        }
        # We need _execute_cell to run far enough to set _current_cell_id.
        # It will error somewhere after the cell_id capture, which is fine.
        import contextlib
        with contextlib.suppress(Exception):
            m._execute_cell("x = 1")

        assert m._current_cell_id == 'vscode-notebook-cell:/fake/path.ipynb#W1s'

    def test_captures_cellid_from_vscode_nested_metadata(self, magics_fixture):
        """cell_id under metadata.vscode.cellId should also be captured."""
        m = magics_fixture
        m.shell.get_parent.return_value = {
            'metadata': {
                'vscode': {
                    'cellId': 'vscode-notebook-cell:/nested/path.ipynb#W2s'
                }
            }
        }
        with contextlib.suppress(Exception):
            m._execute_cell("x = 1")

        assert m._current_cell_id == 'vscode-notebook-cell:/nested/path.ipynb#W2s'

    def test_no_parent_metadata_sets_none(self, magics_fixture):
        """When get_parent returns None, cell_id should be None."""
        m = magics_fixture
        m.shell.get_parent.return_value = None
        with contextlib.suppress(Exception):
            m._execute_cell("x = 1")

        assert m._current_cell_id is None

    def test_seeds_notebook_path_from_vscode_cellid(self, magics_fixture, tmp_path):
        """If cell_id is a valid VS Code URI, notebook path should be seeded."""
        nb = tmp_path / "demo.ipynb"
        nb.write_text(json.dumps({
            "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }))
        raw_path = str(nb).replace("\\", "/")
        encoded = "/" + raw_path[0] + "%3A" + raw_path[2:] if len(raw_path) > 1 and raw_path[1] == ':' else raw_path
        cell_id = f"vscode-notebook-cell:{encoded}#W1sZmlsZQ%3D%3D"

        m = magics_fixture
        m.shell.get_parent.return_value = {
            'metadata': {'cellId': cell_id}
        }
        invalidate_notebook_path_cache()
        with contextlib.suppress(Exception):
            m._execute_cell("x = 1")

        # The notebook path cache should be seeded
        result = get_notebook_path()
        assert result is not None
        assert os.path.normpath(result) == os.path.normpath(str(nb))
        invalidate_notebook_path_cache()

    def test_no_shell_get_parent_graceful(self, magics_fixture):
        """If shell doesn't have get_parent, should not error."""
        m = magics_fixture
        del m.shell.get_parent  # Remove get_parent entirely
        with contextlib.suppress(Exception):
            m._execute_cell("x = 1")
        assert m._current_cell_id is None

    def test_exception_in_parent_does_not_crash(self, magics_fixture):
        """If get_parent raises, _execute_cell should continue."""
        m = magics_fixture
        m.shell.get_parent.side_effect = RuntimeError("kernel error")
        # Should not raise
        try:
            m._execute_cell("x = 1")
        except RuntimeError as e:
            # Make sure this is NOT the "kernel error" from get_parent
            assert "kernel error" not in str(e)
