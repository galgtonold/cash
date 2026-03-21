"""Batch 115 – Multi-statement cell interaction tests.

Tests that exercise cells with multiple statements, where edits
modify only some statements within a cell.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestMultiStatementCellEdits:
    """Edit individual statements within multi-statement cells."""

    def test_edit_first_statement_in_cell(self, nb_runner):
        """Cell has two statements, edit the first one."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "total = x + y\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100\ny = 20")
        nb_runner.run_all()
        assert "total = 120" in nb_runner.get_output(2)

    def test_edit_second_statement_in_cell(self, nb_runner):
        """Cell has two statements, edit the second one."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "total = x + y\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 10\ny = 200")
        nb_runner.run_all()
        assert "total = 210" in nb_runner.get_output(2)

    def test_add_statement_to_cell(self, nb_runner):
        """Add a new statement to an existing cell."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Add y to the first cell
        nb_runner.set_cell_source(1, "x = 10\ny = 5")
        nb_runner.set_cell_source(2, "result = x * 2 + y\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_remove_statement_from_cell(self, nb_runner):
        """Remove a statement from a multi-statement cell."""
        nb_runner.create_notebook([
            "x = 10\ny = 20\nz = 30",
            "total = x + y + z\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Remove z
        nb_runner.set_cell_source(1, "x = 10\ny = 20")
        nb_runner.set_cell_source(2, "total = x + y\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

    def test_reorder_statements_in_cell(self, nb_runner):
        """Reorder statements within a cell."""
        nb_runner.create_notebook([
            "a = 1\nb = a + 1",
            "print(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 2" in nb_runner.get_output(2)

        # Swap order and change logic
        nb_runner.set_cell_source(1, "b = 10\na = b + 1")
        nb_runner.set_cell_source(2, "print(f'a = {a}, b = {b}')")
        nb_runner.run_all()
        assert "a = 11, b = 10" in nb_runner.get_output(2)


class TestMultiStatementWithFunction:
    """Multi-statement cells containing function defs."""

    def test_function_and_call_in_same_cell(self, nb_runner):
        """Function definition and call in same cell."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2\nresult = double(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(1)

        # Edit function body
        nb_runner.set_cell_source(
            1,
            "def double(x):\n    return x * 3\nresult = double(5)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(1)

    def test_two_functions_in_one_cell(self, nb_runner):
        """Two functions defined in one cell, used in next cell."""
        nb_runner.create_notebook([
            "def add(a, b):\n    return a + b\ndef mul(a, b):\n    return a * b",
            "r1 = add(3, 4)\nr2 = mul(3, 4)\nprint(f'r1 = {r1}, r2 = {r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1 = 7, r2 = 12" in nb_runner.get_output(2)

        # Edit one function
        nb_runner.set_cell_source(
            1,
            "def add(a, b):\n    return a + b + 100\ndef mul(a, b):\n    return a * b",
        )
        nb_runner.run_all()
        assert "r1 = 107, r2 = 12" in nb_runner.get_output(2)


class TestMultiStatementWithPrint:
    """Multi-statement cells with print statements (side effects)."""

    def test_print_between_assignments(self, nb_runner):
        """Print statement between two assignments."""
        nb_runner.create_notebook([
            "x = 10\nprint(f'x = {x}')\ny = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "x = 10" in output
        assert "y = 20" in output

    def test_edit_multi_statement_with_prints(self, nb_runner):
        """Edit a multi-statement cell that includes prints."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nprint(f'sum = {a + b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 3" in nb_runner.get_output(1)

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nprint(f'sum = {a + b}')")
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(1)


class TestMultiStatementDependencies:
    """Multi-statement cells with internal dependencies."""

    def test_internal_dependency_chain(self, nb_runner):
        """Statements within a cell depend on each other."""
        nb_runner.create_notebook([
            "a = 1\nb = a + 1\nc = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(1)

        nb_runner.set_cell_source(1, "a = 10\nb = a + 1\nc = b + 1\nprint(f'c = {c}')")
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(1)

    def test_cross_cell_multi_statement(self, nb_runner):
        """Multi-statement cells with cross-cell dependencies."""
        nb_runner.create_notebook([
            "x = 1\ny = 2",
            "a = x + y\nb = x * y",
            "result = a + b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10\ny = 20")
        nb_runner.run_all()
        assert "result = 230" in nb_runner.get_output(3)
