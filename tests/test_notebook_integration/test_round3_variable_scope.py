"""
Tricky multi-cell variable shadowing, reassignment, deletion,
and scope interactions that stress the lineage tracker.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestConditionalAssignment:
    """Test conditional assignment patterns."""

    def test_ternary_expression(self, nb_runner):
        """Ternary expression across cells."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "score = 75",
                textwrap.dedent("""\
                status = 'pass' if score >= threshold else 'fail'
                print(status)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pass" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "score = 30")
        nb_runner.run_all()
        assert "fail" in nb_runner.get_output(3)

    def test_or_default_pattern(self, nb_runner):
        """x = val or default pattern."""
        nb_runner.create_notebook(
            [
                "user_input = ''",
                "name = user_input or 'Anonymous'",
                "print(name)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Anonymous" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "user_input = 'Alice'")
        nb_runner.run_all()
        assert "Alice" in nb_runner.get_output(3)
