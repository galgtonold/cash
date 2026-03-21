
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cash.notebook.magics import CashMagics
from cash.core import Cash
from cash.backends.backend import InMemoryBackend


# Patch locations where get_notebook_cells is used (upstream module only)
PATCH_TARGETS = {
    'upstream_cells': 'cash.notebook.upstream.get_notebook_cells',
    'upstream_cells_ids': 'cash.notebook.upstream.get_notebook_cells_with_ids',
}


class TestVSCodeDuplicates(unittest.TestCase):
    def setUp(self):
        # Dynamically check for real IPython at test time (not import time)
        # to handle sys.modules pollution from other test modules
        try:
            from IPython.core.interactiveshell import InteractiveShell as IS
            has_real_ipython = not isinstance(IS, MagicMock)
        except (ImportError, AttributeError):
            has_real_ipython = False
        
        if has_real_ipython:
            from IPython.core.interactiveshell import InteractiveShell
            self.shell = MagicMock(spec=InteractiveShell)
            try:
                from traitlets.config import Config
                self.shell.config = Config()
            except ImportError:
                pass
        else:
            self.shell = MagicMock()
            self.shell.user_ns = {}
            self.shell.ast_transformers = []
            self.shell.events = MagicMock()
        
        self.cash = MagicMock(spec=Cash)
        self.cash.debug = False
        self.cash.backend = InMemoryBackend()
        self.magics = CashMagics(self.shell, self.cash)
        
        self.magics._original_run_cell = MagicMock()
        self.magics._auto_cache_enabled = True

    @patch(PATCH_TARGETS['upstream_cells_ids'])
    @patch(PATCH_TARGETS['upstream_cells'])
    def test_duplicate_cells_vscode(self, mock_upstream_cells, mock_upstream_ids):
        """Test that duplicate cells raise RuntimeError when in VS Code env."""
        self.shell.user_ns = {'__vsc_ipynb_file__': '/path/to/notebook.ipynb'}
        
        cell_content = "print('hello')"
        cells = ["import os", cell_content, cell_content, "x = 1"]
        mock_upstream_cells.return_value = cells
        # Return cells with IDs (but no matching ID for current cell to force content-based match)
        mock_upstream_ids.return_value = [(f"id_{i}", c) for i, c in enumerate(cells)]
        
        self.magics._execute_cell(cell_content)
            
        # Verify original run_cell was called with code that raises RuntimeError
        self.magics._original_run_cell.assert_called_once()
        args, _ = self.magics._original_run_cell.call_args
        executed_code = args[0]
        self.assertIn("raise AmbiguousCellError", executed_code)
        self.assertIn("Ambiguous cell execution", executed_code)

    @patch(PATCH_TARGETS['upstream_cells_ids'])
    @patch(PATCH_TARGETS['upstream_cells'])
    def test_unique_cell_vscode(self, mock_upstream_cells, mock_upstream_ids):
        """Test that unique cells execute normally in VS Code env."""
        self.shell.user_ns = {'__vsc_ipynb_file__': '/path/to/notebook.ipynb'}
        
        cell_content = "print('unique')"
        cells = ["import os", cell_content, "x = 1"]
        mock_upstream_cells.return_value = cells
        mock_upstream_ids.return_value = [(f"id_{i}", c) for i, c in enumerate(cells)]
        
        # Should NOT raise
        self.magics._execute_cell(cell_content)
        
    @patch(PATCH_TARGETS['upstream_cells_ids'])
    @patch(PATCH_TARGETS['upstream_cells'])
    def test_duplicate_cells_no_vscode(self, mock_upstream_cells, mock_upstream_ids):
        """Test that duplicate cells are handled when not in VS Code (unknown notebook)."""
        self.shell.user_ns = {}
        
        cell_content = "print('hello')"
        mock_upstream_cells.return_value = []
        mock_upstream_ids.return_value = []
        
        self.magics._execute_cell(cell_content)
        # Should not crash - either runs normally or falls back
        self.magics._original_run_cell.assert_called()

    @patch('cash.notebook.magics.CodeAnalyzer.analyze_code_block')
    def test_syntax_error_handling(self, mock_analyze):
        """Test that SyntaxError in user code is delegated to original run_cell."""
        mock_analyze.side_effect = SyntaxError("invalid syntax")
        
        cell_content = "if True \n print('missing colon')"
        
        self.magics._execute_cell(cell_content)
        
        self.magics._original_run_cell.assert_called_with(cell_content)

    @patch(PATCH_TARGETS['upstream_cells_ids'])
    @patch(PATCH_TARGETS['upstream_cells'])
    def test_duplicate_cells_quoting(self, mock_upstream_cells, mock_upstream_ids):
        """Test that duplicate error message uses triple quotes."""
        self.shell.user_ns = {'__vsc_ipynb_file__': '/path/to/notebook.ipynb'}
        cell_content = "print('duplicate')"
        cells = ["import os", cell_content, cell_content]
        mock_upstream_cells.return_value = cells
        mock_upstream_ids.return_value = [(f"id_{i}", c) for i, c in enumerate(cells)]
        
        self.magics._execute_cell(cell_content)
        
        args, _ = self.magics._original_run_cell.call_args
        executed_code = args[0]
        self.assertIn("raise AmbiguousCellError('''", executed_code)
        self.assertIn("''')", executed_code)

if __name__ == '__main__':
    unittest.main()
