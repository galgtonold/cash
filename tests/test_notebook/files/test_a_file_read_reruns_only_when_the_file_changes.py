"""A statement that reads a file is skipped while the file is unchanged and runs
again once it changes.
"""

import time

from tests._cell_driver import run_cash_cell


class TestSkipWithoutRestore:
    """Test the improved skip-without-restore optimization for file dependencies."""

    def test_file_deps_skipped_when_unchanged(self, cash_magics, mock_shell, tmp_path):
        """Statement with file deps should be SKIPPED if files haven't changed."""
        # Create a test file
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        file_path = str(test_file).replace("\\", "/")

        # Execute a statement that reads a file
        run_cash_cell(cash_magics, "import csv")

        # First execution
        code = f"data = open('{file_path}').read()"
        run_cash_cell(cash_magics, code)
        assert "data" in mock_shell.user_ns
        first_value = mock_shell.user_ns["data"]

        # Second execution — should be SKIPPED (file unchanged)
        run_cash_cell(cash_magics, code)
        assert mock_shell.user_ns["data"] == first_value

    def test_file_deps_reexecuted_when_changed(self, cash_magics, mock_shell, tmp_path):
        """Statement with file deps should be REEXECUTED if files have changed."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        file_path = str(test_file).replace("\\", "/")

        # First execution
        code = f"data = open('{file_path}').read()"
        run_cash_cell(cash_magics, code)
        first_value = mock_shell.user_ns["data"]

        # Modify the file
        time.sleep(0.1)  # Ensure different mtime
        test_file.write_text("a,b\n5,6\n7,8\n", encoding="utf-8")

        # Second execution — should detect file change and re-execute
        run_cash_cell(cash_magics, code)
        # The value should reflect the new file content
        new_value = mock_shell.user_ns["data"]
        assert new_value != first_value, "File changed, statement should have re-executed"
        assert "5,6" in new_value
