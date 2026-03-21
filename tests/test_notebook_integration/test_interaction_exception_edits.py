"""Batch 152 – Exception handling code interaction tests.

Tests where try/except blocks are edited, error paths change,
and caching handles exception-related code modifications.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestTryExceptEdits:
    """Edit try/except blocks."""

    def test_edit_try_body(self, nb_runner):
        """Edit the try body."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "try:\n    result = data[0]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(2)

        # Edit to access out of bounds
        nb_runner.set_cell_source(
            2,
            "try:\n    result = data[10]\nexcept IndexError:\n    result = -1\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

    def test_edit_except_handler(self, nb_runner):
        """Edit the except handler."""
        nb_runner.create_notebook([
            "x = 0",
            "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = 0\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

        # Edit to handle differently
        nb_runner.set_cell_source(
            2,
            "try:\n    result = 100 / x\nexcept ZeroDivisionError:\n    result = -999\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = -999" in nb_runner.get_output(2)

    def test_fix_value_to_avoid_exception(self, nb_runner):
        """Fix the value so exception doesn't trigger."""
        nb_runner.create_notebook([
            "divisor = 0",
            "try:\n    result = 100 / divisor\nexcept ZeroDivisionError:\n    result = -1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -1" in nb_runner.get_output(2)

        # Fix divisor
        nb_runner.set_cell_source(1, "divisor = 4")
        nb_runner.run_all()
        assert "result = 25.0" in nb_runner.get_output(2)


class TestContextManagerEdits:
    """Context manager patterns with edits."""

    def test_edit_context_body(self, nb_runner, tmp_path):
        """Edit code inside context manager."""
        fpath = tmp_path / "test.txt"
        fpath.write_text("hello world")
        fpath_str = str(fpath).replace("\\", "/")

        nb_runner.create_notebook([
            f"path = '{fpath_str}'",
            "with open(path) as f:\n    content = f.read()\nresult = len(content)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        # Edit to get word count instead
        nb_runner.set_cell_source(
            2,
            "with open(path) as f:\n    content = f.read()\nresult = len(content.split())\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 2" in nb_runner.get_output(2)
