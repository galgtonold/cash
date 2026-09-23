"""
Loop and control structure caching patterns.

Tests how cash handles for loops, while loops, if/else branches, nested
control structures, and their caching/invalidation behavior across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.loops]


class TestForLoopCaching:
    """Test for loop caching across cells."""

    def test_for_loop_data_change(self, nb_runner):
        """Change input data and verify loop output updates."""
        nb_runner.create_notebook(
            [
                "numbers = [1, 2, 3]",
                textwrap.dedent("""\
                squares = []
                for n in numbers:
                    squares.append(n ** 2)
                print(squares)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 4, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "numbers = [10, 20, 30]")
        nb_runner.run_all()
        assert "[100, 400, 900]" in nb_runner.get_output(2)


class TestWhileLoopCaching:
    """Test while loop caching."""

    def test_while_with_break(self, nb_runner):
        """While loop with break condition."""
        nb_runner.create_notebook(
            [
                "data = [1, 3, 5, 7, 2, 4, 6]",
                textwrap.dedent("""\
                first_even = None
                i = 0
                while i < len(data):
                    if data[i] % 2 == 0:
                        first_even = data[i]
                        break
                    i += 1
                print(first_even)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2" in nb_runner.get_output(2)


class TestIfElseCaching:
    """Test if/else branch caching."""

    def test_branch_change_invalidation(self, nb_runner):
        """Changing input changes which branch executes."""
        nb_runner.create_notebook(
            [
                "value = 3",
                textwrap.dedent("""\
                if value % 2 == 0:
                    parity = 'even'
                else:
                    parity = 'odd'
                print(parity)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "odd" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "value = 4")
        nb_runner.run_all()
        assert "even" in nb_runner.get_output(2)

    def test_ternary_expression(self, nb_runner):
        """Ternary (conditional) expression."""
        nb_runner.create_notebook(
            [
                "n = 7",
                textwrap.dedent("""\
                parity = 'even' if n % 2 == 0 else 'odd'
                print(parity)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "odd" in nb_runner.get_output(2)


class TestComplexControlFlow:
    """Test complex control flow patterns."""

    def test_loop_with_conditional(self, nb_runner):
        """Loop with conditional inside."""
        nb_runner.create_notebook(
            [
                "numbers = list(range(1, 11))",
                textwrap.dedent("""\
                evens = []
                odds = []
                for n in numbers:
                    if n % 2 == 0:
                        evens.append(n)
                    else:
                        odds.append(n)
                print(len(evens), len(odds))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5 5" in nb_runner.get_output(2)
