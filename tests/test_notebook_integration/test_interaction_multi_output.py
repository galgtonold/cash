"""Batch 154 – Multi-output cell interaction tests.

Tests where cells produce multiple outputs, some used by
different downstream cells, with edits that affect
only some of the outputs.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestMultiOutputEdits:
    """Cells that produce multiple variables."""

    def test_edit_multi_output_cell(self, nb_runner):
        """Edit cell that produces two variables."""
        nb_runner.create_notebook([
            "a = 10\nb = 20",
            "result = a + b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit to change both outputs
        nb_runner.set_cell_source(1, "a = 100\nb = 200")
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(2)

    def test_multi_output_different_consumers(self, nb_runner):
        """Two outputs consumed by different cells."""
        nb_runner.create_notebook([
            "x = 5\ny = 10",
            "rx = x * 2\nprint(f'rx = {rx}')",
            "ry = y * 3\nprint(f'ry = {ry}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rx = 10" in nb_runner.get_output(2)
        assert "ry = 30" in nb_runner.get_output(3)

        # Edit only x
        nb_runner.set_cell_source(1, "x = 50\ny = 10")
        nb_runner.run_all()
        assert "rx = 100" in nb_runner.get_output(2)
        assert "ry = 30" in nb_runner.get_output(3)

    def test_tuple_unpacking_edit(self, nb_runner):
        """Tuple unpacking with edits."""
        nb_runner.create_notebook([
            "a, b, c = 1, 2, 3",
            "total = a + b + c\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a, b, c = 10, 20, 30")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)


class TestMultiStatementCellEdits:
    """Cells with multiple statements, edit individual statements."""

    def test_edit_one_statement_in_multi_stmt_cell(self, nb_runner):
        """Edit one statement in a multi-statement cell."""
        nb_runner.create_notebook([
            "x = 10\ny = x * 2\nz = y + 1",
            "print(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 21" in nb_runner.get_output(2)

        # Edit the middle statement
        nb_runner.set_cell_source(1, "x = 10\ny = x * 10\nz = y + 1")
        nb_runner.run_all()
        assert "z = 101" in nb_runner.get_output(2)

    def test_reorder_statements_in_cell(self, nb_runner):
        """Reorder statements within a cell."""
        nb_runner.create_notebook([
            "a = 5\nb = a + 1",
            "result = b * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

        # Change to different computation order
        nb_runner.set_cell_source(1, "b = 100\na = b - 1")
        nb_runner.set_cell_source(2, "result = a * 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 198" in nb_runner.get_output(2)
