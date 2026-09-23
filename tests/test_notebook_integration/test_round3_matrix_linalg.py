"""Matrix/linear algebra — pure Python matrix operations with cash."""

import textwrap

import pytest


@pytest.mark.stress
class TestPureMatrixOps:
    """Test pure Python matrix operations."""

    def test_matrix_change_propagates(self, nb_runner):
        """Changing matrix definition propagates."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dot=32" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def dot_product(v1, v2):
                return sum(a * b for a, b in zip(v1, v2))

            vec_a = [10, 20, 30]
            vec_b = [4, 5, 6]
        """),
        )
        nb_runner.run_all()
        assert "dot=320" in nb_runner.get_output(2)
