"""matrix/linear algebra with pure Python lists."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMatrixOps:
    """Matrix operations implemented with lists."""

    def test_matrix_propagation(self, nb_runner):
        """Matrix computation with upstream change propagation."""
        nb_runner.create_notebook(
            [
                "scale = 2",
                textwrap.dedent("""\
                def scale_matrix(M, s):
                    return [[x * s for x in row] for row in M]
                M = [[1, 2], [3, 4]]
                scaled = scale_matrix(M, scale)
            """),
                "print(f'scaled={scaled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[[2, 4], [6, 8]]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[[10, 20], [30, 40]]" in nb_runner.get_output(3)
