"""Batch 192 – Error-then-fix pattern interaction tests.

Tests where cells produce errors, then are fixed, and
the cache correctly handles the recovery.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestErrorThenFix:
    """Tests where errors are introduced then fixed."""

    def test_fix_name_error_recovery(self, nb_runner):
        """Introduce a NameError, then fix it."""
        nb_runner.create_notebook([
            "x = 10  # name error recovery source",
            "result = x + y\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix by defining y
        nb_runner.set_cell_source(
            2, "y = 20\nresult = x + y\nprint(f'result = {result}')"
        )
        nb_runner.run_cell(2)
        assert "result = 30" in nb_runner.get_output(2)

    def test_fix_type_error_recovery(self, nb_runner):
        """Introduce a TypeError, then fix it."""
        nb_runner.create_notebook([
            "a = '10'  # type error recovery source",
            "result = a + 5\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix by converting
        nb_runner.set_cell_source(
            2, "result = int(a) + 5\nprint(f'result = {result}')"
        )
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_fix_import_error_recovery(self, nb_runner):
        """Introduce an import error, then fix it."""
        nb_runner.create_notebook([
            "from math import nonexistent_func  # import error recovery",
            "result = nonexistent_func(42)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(1)

        # Fix the import
        nb_runner.set_cell_source(1, "from math import sqrt  # import fixed recovery")
        nb_runner.set_cell_source(
            2, "result = sqrt(42)\nprint(f'result = {result:.2f}')"
        )
        nb_runner.run_all()
        assert "result = 6.48" in nb_runner.get_output(2)

    def test_fix_then_iterate(self, nb_runner):
        """Fix an error, verify works, then edit again."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]  # fix iterate source",
            "result = data[10]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)

        # Fix index
        nb_runner.set_cell_source(
            2, "result = data[1]\nprint(f'result = {result}')"
        )
        nb_runner.run_cell(2)
        assert "result = 2" in nb_runner.get_output(2)

        # Edit again to use a different index
        nb_runner.set_cell_source(
            2, "result = data[2]\nprint(f'result = {result}')"
        )
        nb_runner.run_cell(2)
        assert "result = 3" in nb_runner.get_output(2)
