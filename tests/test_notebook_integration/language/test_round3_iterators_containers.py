"""Iterator & custom container patterns — __iter__, __getitem__, __contains__."""

import textwrap

import pytest


@pytest.mark.stress
class TestCustomContainers:
    """Test custom container classes."""

    def test_matrix_container(self, nb_runner):
        """Custom matrix with __getitem__ and __setitem__."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Matrix:
                    def __init__(self, rows, cols, fill=0):
                        self.rows = rows
                        self.cols = cols
                        self.data = [[fill] * cols for _ in range(rows)]

                    def __getitem__(self, key):
                        r, c = key
                        return self.data[r][c]

                    def __setitem__(self, key, value):
                        r, c = key
                        self.data[r][c] = value

                    def __repr__(self):
                        return f"Matrix({self.rows}x{self.cols})"

                m = Matrix(3, 3)
                for i in range(3):
                    m[i, i] = 1  # identity
            """),
                textwrap.dedent("""\
                diag = [m[i, i] for i in range(3)]
                off_diag = m[0, 1]
                print(f"m={m} diag={diag} off={off_diag}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Matrix(3x3)" in nb_runner.get_output(2)
        assert "diag=[1, 1, 1]" in nb_runner.get_output(2)
        assert "off=0" in nb_runner.get_output(2)

    def test_default_dict_like(self, nb_runner):
        """Custom defaultdict-like with factory across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class AutoDict(dict):
                    def __init__(self, factory):
                        super().__init__()
                        self.factory = factory

                    def __missing__(self, key):
                        self[key] = self.factory()
                        return self[key]

                word_counts = AutoDict(int)
                words = "the cat sat on the mat the cat".split()
                for w in words:
                    word_counts[w] += 1
            """),
                textwrap.dedent("""\
                sorted_counts = sorted(word_counts.items())
                print(f"counts={sorted_counts}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('the', 3)" in out
        assert "('cat', 2)" in out
