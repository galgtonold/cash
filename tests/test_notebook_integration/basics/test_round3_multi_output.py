"""
Multi-output cell patterns, display vs return, print ordering,
and assignment expression (walrus) patterns.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestExpressionVsStatement:
    """Test expression vs statement behavior."""

    def test_bare_expression(self, nb_runner):
        """Bare expression in cell (like Jupyter shows last expression)."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "x",
                "y = x + 8",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "50" in nb_runner.get_output(4)

    def test_semicolon_suppression(self, nb_runner):
        """Semicolons in cells."""
        nb_runner.create_notebook(
            [
                "a = 1; b = 2; c = 3",
                textwrap.dedent("""\
                total = a + b + c
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)
