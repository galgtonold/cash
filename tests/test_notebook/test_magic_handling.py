"""
Test handling of Jupyter magics in upstream cells
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pytest

from tests._cell_driver import run_cash_cell


class TestMagicHandling(unittest.TestCase):
    """Test that Jupyter magics are correctly ignored in upstream cells."""

    @pytest.fixture(autouse=True)
    def _notebook(self, cash_magics, mock_shell, clean_backend):
        self.magics, self.shell, self.backend = cash_magics, mock_shell, clean_backend

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, "test.ipynb")

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_notebook(self, cells):
        notebook = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [cell]}
                for cell in cells
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(self.notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f)

    def test_upstream_cell_with_magic_is_ignored(self):
        """Test that an upstream cell with magics doesn't cause SyntaxError."""
        print("\n=== TEST: Upstream Cell with Magic ===")

        # Cell 1: Mixed magic and code
        cell1 = "%matplotlib inline\nx = 10"
        # Cell 2: Uses x
        cell2 = "result = x * 2"

        self.create_notebook([cell1, cell2])

        with patch("cash.notebook.upstream.checker.get_notebook_cells") as mock_get_cells:

            def get_cells(_path=None):
                with open(self.notebook_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                return [cell["source"][0] for cell in nb["cells"]]

            mock_get_cells.side_effect = get_cells

            # Simulate initial execution state
            print("--- Step 1: Simulate initial execution of 'x=10' ---")
            # We need to manually populate tracking dicts to simulate that x=10 was executed and cached
            # effectively ignoring the magic part which CashMagics usually skips caching for anyway (unless handled)
            self.magics.tracking_state.executed_cell_codes["x"] = "x = 10"
            # A SET, which is what every production writer stores here
            # (statement/lineage.py, statement/restore.py, upstream/_types.py).
            # A bare string made `.add()` raise AttributeError, cash bailed out
            # of its own pipeline, and the cell then ran UNCACHED through
            # IPython -- so `x == 20` below held for a reason that had nothing
            # to do with the magic being ignored.
            self.magics.tracking_state.executed_cell_hashes["x"] = {"hash_of_x_10"}
            self.magics.tracking_state.lineage.record("x", "lineage_of_x_10")
            self.shell.user_ns["x"] = 10

            print("--- Step 2: Modify notebook to have magic and x=20 ---")
            cell1_v2 = "%matplotlib inline\nx = 20"
            self.create_notebook([cell1_v2, cell2])

            print("--- Step 3: Run Cell 2 ---")
            # This should trigger re-execution of Cell 1
            run_cash_cell(self.magics, cell2)

            print(f"x in memory: {self.shell.user_ns.get('x')}")
            # If re-execution works despite magic, x should be 20
            self.assertEqual(self.shell.user_ns.get("x"), 20)
            print("[OK] Cell 1 re-executed successfully (magic ignored)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
