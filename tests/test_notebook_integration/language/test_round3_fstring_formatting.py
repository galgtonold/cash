"""Complex f-strings & string formatting — cash caching with advanced formatting."""

import textwrap

import pytest


@pytest.mark.stress
class TestFStringPatterns:
    """Test complex f-string patterns across cells."""

    def test_format_spec_expressions(self, nb_runner):
        """Format spec with computed width and precision."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                values = [3.14159, 2.71828, 1.41421]
                width = 10
                precision = 3
                formatted = [f"{v:{width}.{precision}f}" for v in values]
                print(f"formatted={formatted}")
            """),
                textwrap.dedent("""\
                joined = ' | '.join(formatted)
                print(f"table={joined}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3.142" in nb_runner.get_output(1)
        assert " | " in nb_runner.get_output(2)

    def test_fstring_propagation(self, nb_runner):
        """F-string result propagates when upstream changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                name = "World"
                greeting = f"Hello, {name}!"
            """),
                textwrap.dedent("""\
                print(f"msg={greeting}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            name = "Python"
            greeting = f"Hello, {name}!"
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "msg=Hello, Python!" in nb_runner.get_output(2)
