"""Editing a loop's body, range or data and re-running."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.loops]


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.timeout(45)
class TestForLoopEdits:
    """For loop editing patterns."""

    def test_edit_loop_range(self, nb_runner):
        """Edit the range of a for loop."""
        nb_runner.create_notebook(
            [
                "total = 0\nfor i in range(5):\n    total += i",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "total = 0\nfor i in range(10):\n    total += i")
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(2)

    def test_edit_loop_body(self, nb_runner):
        """Edit the body of a for loop."""
        nb_runner.create_notebook(
            [
                "results = []\nfor i in range(5):\n    results.append(i * 2)",
                "total = sum(results)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # [0,2,4,6,8] -> 20
        assert "total = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "results = []\nfor i in range(5):\n    results.append(i ** 2)")
        nb_runner.run_all()
        # [0,1,4,9,16] -> 30
        assert "total = 30" in nb_runner.get_output(2)


@pytest.mark.timeout(45)
class TestWhileLoopEdits:
    """While loop editing patterns."""

    def test_edit_while_condition(self, nb_runner):
        """Edit while loop condition."""
        nb_runner.create_notebook(
            [
                "count = 0\nval = 1\nwhile val < 100:\n    val *= 2\n    count += 1",
                "print(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 7" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "count = 0\nval = 1\nwhile val < 1000:\n    val *= 2\n    count += 1")
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(2)

    def test_edit_while_body(self, nb_runner):
        """Edit while loop body."""
        nb_runner.create_notebook(
            [
                "n = 10\ntotal = 0\nwhile n > 0:\n    total += n\n    n -= 1",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "n = 10\ntotal = 0\nwhile n > 0:\n    total += n * n\n    n -= 1")
        nb_runner.run_all()
        assert "total = 385" in nb_runner.get_output(2)
