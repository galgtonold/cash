"""Batch 130 – Multi-statement cell interaction tests (advanced).

Tests that exercise cells with multiple statements and complex
interactions between statements within the same cell.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestMultiStatementBasic:
    """Multi-statement cells, edit individual parts."""


    def test_edit_middle_statement_in_multi(self, nb_runner):
        """Cell has 3 assignments, edit the middle one."""
        nb_runner.create_notebook([
            "x = 1\ny = 2\nz = 3",
            "result = x * y * z\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 1\ny = 20\nz = 3")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    def test_add_statement_to_cell(self, nb_runner):
        """Add a statement to a cell."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Add an additional computation
        nb_runner.set_cell_source(1, "x = 10\nx = x * 5")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)


class TestMultiStatementWithFunctions:
    """Multi-statement cells with function calls."""

    def test_define_and_use_in_same_cell(self, nb_runner):
        """Define function and use it in same cell, edit the function."""
        nb_runner.create_notebook([
            "def calc(x):\n    return x * 2\nresult = calc(5)",
            "print(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "def calc(x):\n    return x ** 2\nresult = calc(5)"
        )
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)



class TestMultiStatementWithControlFlow:
    """Multi-statement cells with control flow."""

    def test_assignment_then_loop(self, nb_runner):
        """Assignment followed by loop in same cell."""
        nb_runner.create_notebook([
            "n = 5",
            "total = 0\nfor i in range(n):\n    total += i\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+1+2+3+4 = 10
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "n = 10")
        nb_runner.run_all()
        # 0+1+...+9 = 45
        assert "total = 45" in nb_runner.get_output(2)

    def test_complex_multi_statement_edit(self, nb_runner):
        """Complex cell with multiple interacting statements, edit one."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "mean = sum(data) / len(data)\ndevs = [(x - mean) ** 2 for x in data]\nvariance = sum(devs) / len(devs)\nprint(f'variance = {variance}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # mean=3, devs=[4,1,0,1,4], variance=2.0
        assert "variance = 2.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 10, 10, 10, 10]")
        nb_runner.run_all()
        # mean=10, devs=[0,0,0,0,0], variance=0.0
        assert "variance = 0.0" in nb_runner.get_output(2)
