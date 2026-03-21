"""Batch 96 – matrix/linear algebra with pure Python lists."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMatrixOps:
    """Matrix operations implemented with lists."""

    def test_matrix_multiply(self, nb_runner):
        """Matrix multiplication with nested lists."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def mat_mul(A, B):
                    rows_a, cols_a = len(A), len(A[0])
                    rows_b, cols_b = len(B), len(B[0])
                    assert cols_a == rows_b
                    result = [[0] * cols_b for _ in range(rows_a)]
                    for i in range(rows_a):
                        for j in range(cols_b):
                            for k in range(cols_a):
                                result[i][j] += A[i][k] * B[k][j]
                    return result

                A = [[1, 2], [3, 4]]
                B = [[5, 6], [7, 8]]
                C = mat_mul(A, B)
            """),
            "print(f'C={C}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "[[19, 22], [43, 50]]" in out

    def test_matrix_transpose(self, nb_runner):
        """Transpose a matrix."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def transpose(M):
                    return [list(row) for row in zip(*M)]

                M = [[1, 2, 3], [4, 5, 6]]
                T = transpose(M)
            """),
            "print(f'M={M}')\nprint(f'T={T}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "[[1, 4], [2, 5], [3, 6]]" in out

    def test_identity_and_trace(self, nb_runner):
        """Identity matrix and trace."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def identity(n):
                    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

                def trace(M):
                    return sum(M[i][i] for i in range(len(M)))

                I = identity(4)
                tr = trace(I)

                M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                tr_m = trace(M)
            """),
            "print(f'tr_I={tr} tr_M={tr_m}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "tr_I=4" in out
        assert "tr_M=15" in out

    def test_matrix_propagation(self, nb_runner):
        """Matrix computation with upstream change propagation."""
        nb_runner.create_notebook([
            "scale = 2",
            textwrap.dedent("""\
                def scale_matrix(M, s):
                    return [[x * s for x in row] for row in M]
                M = [[1, 2], [3, 4]]
                scaled = scale_matrix(M, scale)
            """),
            "print(f'scaled={scaled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[[2, 4], [6, 8]]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[[10, 20], [30, 40]]" in nb_runner.get_output(3)

    def test_dot_product(self, nb_runner):
        """Vector dot product."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def dot(a, b):
                    return sum(x * y for x, y in zip(a, b))

                v1 = [1, 2, 3]
                v2 = [4, 5, 6]
                result = dot(v1, v2)
                magnitude = sum(x**2 for x in v1) ** 0.5
            """),
            "print(f'dot={result} mag={magnitude:.4f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "dot=32" in out
        assert "mag=3.7417" in out
