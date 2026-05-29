"""Batch 215 – Matrix operations interaction tests.

Tests editing cells with nested list matrix operations
(transpose, multiply, etc.) and verifying propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMatrixOpsEdits:
    """Editing matrix operation patterns."""



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

