"""Batch 61: Matrix/linear algebra — pure Python matrix operations with cash."""
import textwrap
import pytest


@pytest.mark.stress
class TestPureMatrixOps:
    """Test pure Python matrix operations."""

    def test_matrix_multiply(self, nb_runner):
        """Matrix multiplication across cells."""
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
            """),
            textwrap.dedent("""\
                C = mat_mul(A, B)
                print(f"C={C}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "C=[[19, 22], [43, 50]]" in nb_runner.get_output(2)

    def test_matrix_transpose(self, nb_runner):
        """Matrix transpose across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def transpose(M):
                    return [list(row) for row in zip(*M)]

                M = [[1, 2, 3], [4, 5, 6]]
            """),
            textwrap.dedent("""\
                T = transpose(M)
                print(f"original={M}")
                print(f"transposed={T}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed=[[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

    def test_matrix_determinant(self, nb_runner):
        """2x2 and 3x3 determinant across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def det2x2(M):
                    return M[0][0] * M[1][1] - M[0][1] * M[1][0]

                def det3x3(M):
                    return (M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
                          - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
                          + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]))

                m2 = [[3, 8], [4, 6]]
                m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            """),
            textwrap.dedent("""\
                d2 = det2x2(m2)
                d3 = det3x3(m3)
                print(f"det2={d2} det3={d3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "det2=-14" in nb_runner.get_output(2)
        assert "det3=0" in nb_runner.get_output(2)  # singular matrix

    def test_matrix_change_propagates(self, nb_runner):
        """Changing matrix definition propagates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def dot_product(v1, v2):
                    return sum(a * b for a, b in zip(v1, v2))

                vec_a = [1, 2, 3]
                vec_b = [4, 5, 6]
            """),
            textwrap.dedent("""\
                result = dot_product(vec_a, vec_b)
                print(f"dot={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dot=32" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def dot_product(v1, v2):
                return sum(a * b for a, b in zip(v1, v2))

            vec_a = [10, 20, 30]
            vec_b = [4, 5, 6]
        """))
        nb_runner.run_all()
        assert "dot=320" in nb_runner.get_output(2)


@pytest.mark.stress
class TestLinearAlgebraAlgos:
    """Test linear algebra algorithms."""

    def test_gauss_elimination_simple(self, nb_runner):
        """Simple Gaussian elimination."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def solve_2x2(a11, a12, b1, a21, a22, b2):
                    det = a11 * a22 - a12 * a21
                    if det == 0:
                        return None
                    x = (b1 * a22 - a12 * b2) / det
                    y = (a11 * b2 - b1 * a21) / det
                    return (x, y)
            """),
            textwrap.dedent("""\
                # 2x + 3y = 13, 5x + 4y = 22
                solution = solve_2x2(2, 3, 13, 5, 4, 22)
                print(f"solution={solution}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "2.0" in out  # x=2
        assert "3.0" in out  # y=3

    def test_vector_operations(self, nb_runner):
        """Vector math operations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import math

                class Vec3:
                    def __init__(self, x, y, z):
                        self.x, self.y, self.z = x, y, z
                    def __add__(self, other):
                        return Vec3(self.x+other.x, self.y+other.y, self.z+other.z)
                    def __sub__(self, other):
                        return Vec3(self.x-other.x, self.y-other.y, self.z-other.z)
                    def dot(self, other):
                        return self.x*other.x + self.y*other.y + self.z*other.z
                    def cross(self, other):
                        return Vec3(
                            self.y*other.z - self.z*other.y,
                            self.z*other.x - self.x*other.z,
                            self.x*other.y - self.y*other.x)
                    def length(self):
                        return math.sqrt(self.dot(self))
                    def __repr__(self):
                        return f"Vec3({self.x}, {self.y}, {self.z})"

                v1 = Vec3(1, 0, 0)
                v2 = Vec3(0, 1, 0)
            """),
            textwrap.dedent("""\
                cross = v1.cross(v2)
                dot = v1.dot(v2)
                add = v1 + v2
                print(f"cross={cross} dot={dot} add={add}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "cross=Vec3(0, 0, 1)" in out
        assert "dot=0" in out
        assert "add=Vec3(1, 1, 0)" in out
