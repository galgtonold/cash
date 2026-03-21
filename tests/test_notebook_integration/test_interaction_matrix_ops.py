"""Batch 215 – Matrix operations interaction tests.

Tests editing cells with nested list matrix operations
(transpose, multiply, etc.) and verifying propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMatrixOpsEdits:
    """Editing matrix operation patterns."""

    def test_edit_matrix_values(self, nb_runner):
        """Edit matrix and check computed result."""
        nb_runner.create_notebook([
            "matrix = [[1, 2], [3, 4]]",
            "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Edit matrix
        nb_runner.set_cell_source(1, "matrix = [[10, 20, 30], [40, 50, 60]]")
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40, 50, 60]" in nb_runner.get_output(2)

    def test_edit_matrix_transpose(self, nb_runner):
        """Edit matrix and verify transpose."""
        nb_runner.create_notebook([
            "m = [[1, 2, 3], [4, 5, 6]]",
            "t = list(map(list, zip(*m)))\nprint(f'transposed = {t}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed = [[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

        # Change matrix
        nb_runner.set_cell_source(1, "m = [[10, 20], [30, 40], [50, 60]]")
        nb_runner.run_all()
        assert "transposed = [[10, 30, 50], [20, 40, 60]]" in nb_runner.get_output(2)

    def test_edit_matrix_sum(self, nb_runner):
        """Edit two matrices and check element-wise sum."""
        nb_runner.create_notebook([
            "a = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]",
            "s = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]\nprint(f'sum = {s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = [[6, 8], [10, 12]]" in nb_runner.get_output(2)

        # Change matrix b
        nb_runner.set_cell_source(1, "a = [[1, 2], [3, 4]]\nb = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "sum = [[11, 22], [33, 44]]" in nb_runner.get_output(2)

    def test_edit_matrix_row_sums(self, nb_runner):
        """Edit matrix and check row sums."""
        nb_runner.create_notebook([
            "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            "row_sums = [sum(row) for row in grid]\nprint(f'row_sums = {row_sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "row_sums = [6, 15, 24]" in nb_runner.get_output(2)

        # Edit grid
        nb_runner.set_cell_source(1, "grid = [[10, 10], [20, 20]]")
        nb_runner.run_all()
        assert "row_sums = [20, 40]" in nb_runner.get_output(2)
