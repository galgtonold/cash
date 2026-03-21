"""
Batch 23: Loop and control structure caching patterns.

Tests how cash handles for loops, while loops, if/else branches, nested
control structures, and their caching/invalidation behavior across cells.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.loops]


class TestForLoopCaching:
    """Test for loop caching across cells."""

    def test_simple_for_loop(self, nb_runner):
        """Simple for loop with accumulator."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                total = 0
                for x in data:
                    total += x
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(2)

    def test_for_loop_with_list_building(self, nb_runner):
        """For loop building a list."""
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']",
            textwrap.dedent("""\
                upper_items = []
                for item in items:
                    upper_items.append(item.upper())
                print(upper_items)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "APPLE" in output
        assert "BANANA" in output
        assert "CHERRY" in output

    def test_nested_for_loops(self, nb_runner):
        """Nested for loops."""
        nb_runner.create_notebook([
            "rows = 3\ncols = 4",
            textwrap.dedent("""\
                matrix = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        row.append(i * cols + j)
                    matrix.append(row)
                print(len(matrix), len(matrix[0]))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3 4" in nb_runner.get_output(2)

    def test_for_loop_with_enumerate(self, nb_runner):
        """For loop with enumerate."""
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie']",
            textwrap.dedent("""\
                indexed = {}
                for i, name in enumerate(names):
                    indexed[i] = name
                print(indexed)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "0: 'Alice'" in output

    def test_for_loop_data_change(self, nb_runner):
        """Change input data and verify loop output updates."""
        nb_runner.create_notebook([
            "numbers = [1, 2, 3]",
            textwrap.dedent("""\
                squares = []
                for n in numbers:
                    squares.append(n ** 2)
                print(squares)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 4, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "numbers = [10, 20, 30]")
        nb_runner.run_all()
        assert "[100, 400, 900]" in nb_runner.get_output(2)


class TestWhileLoopCaching:
    """Test while loop caching."""

    def test_simple_while_loop(self, nb_runner):
        """Simple while loop with counter."""
        nb_runner.create_notebook([
            "limit = 5",
            textwrap.dedent("""\
                count = 0
                total = 0
                while count < limit:
                    total += count
                    count += 1
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(2)  # 0+1+2+3+4

    def test_while_with_break(self, nb_runner):
        """While loop with break condition."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2" in nb_runner.get_output(2)


class TestIfElseCaching:
    """Test if/else branch caching."""

    def test_simple_if_else(self, nb_runner):
        """Simple if/else cached correctly."""
        nb_runner.create_notebook([
            "x = 10",
            textwrap.dedent("""\
                if x > 5:
                    result = 'high'
                else:
                    result = 'low'
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "high" in nb_runner.get_output(2)

    def test_if_elif_else(self, nb_runner):
        """If/elif/else chain."""
        nb_runner.create_notebook([
            "score = 75",
            textwrap.dedent("""\
                if score >= 90:
                    grade = 'A'
                elif score >= 80:
                    grade = 'B'
                elif score >= 70:
                    grade = 'C'
                else:
                    grade = 'F'
                print(grade)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "C" in nb_runner.get_output(2)

    def test_branch_change_invalidation(self, nb_runner):
        """Changing input changes which branch executes."""
        nb_runner.create_notebook([
            "value = 3",
            textwrap.dedent("""\
                if value % 2 == 0:
                    parity = 'even'
                else:
                    parity = 'odd'
                print(parity)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "odd" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "value = 4")
        nb_runner.run_all()
        assert "even" in nb_runner.get_output(2)

    def test_nested_if_else(self, nb_runner):
        """Nested if/else blocks."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                x = 15
                y = 25
            """),
            textwrap.dedent("""\
                if x > 10:
                    if y > 20:
                        label = 'both_high'
                    else:
                        label = 'x_high'
                else:
                    label = 'x_low'
                print(label)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "both_high" in nb_runner.get_output(2)

    def test_ternary_expression(self, nb_runner):
        """Ternary (conditional) expression."""
        nb_runner.create_notebook([
            "n = 7",
            textwrap.dedent("""\
                parity = 'even' if n % 2 == 0 else 'odd'
                print(parity)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "odd" in nb_runner.get_output(2)


class TestComplexControlFlow:
    """Test complex control flow patterns."""

    def test_loop_with_conditional(self, nb_runner):
        """Loop with conditional inside."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5 5" in nb_runner.get_output(2)

    def test_loop_with_continue(self, nb_runner):
        """Loop with continue statement."""
        nb_runner.create_notebook([
            "data = [1, -2, 3, -4, 5]",
            textwrap.dedent("""\
                positive_sum = 0
                for x in data:
                    if x < 0:
                        continue
                    positive_sum += x
                print(positive_sum)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "9" in nb_runner.get_output(2)

    def test_try_in_loop(self, nb_runner):
        """Try/except inside a loop."""
        nb_runner.create_notebook([
            "values = ['1', 'abc', '3', 'xyz', '5']",
            textwrap.dedent("""\
                parsed = []
                errors = 0
                for v in values:
                    try:
                        parsed.append(int(v))
                    except ValueError:
                        errors += 1
                print(f"parsed={len(parsed)} errors={errors}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parsed=3 errors=2" in nb_runner.get_output(2)

    def test_list_comprehension_vs_loop(self, nb_runner):
        """Compare list comprehension with explicit loop — both should cache."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                # List comprehension
                lc = [x ** 2 for x in data]
            """),
            textwrap.dedent("""\
                # Explicit loop
                loop_result = []
                for x in data:
                    loop_result.append(x ** 2)
            """),
            textwrap.dedent("""\
                print(lc == loop_result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "True" in nb_runner.get_output(4)

    def test_for_else_pattern(self, nb_runner):
        """For/else pattern (else runs if no break)."""
        nb_runner.create_notebook([
            "haystack = [1, 3, 5, 7, 9]",
            textwrap.dedent("""\
                found = False
                for x in haystack:
                    if x == 4:
                        found = True
                        break
                else:
                    found = False
                print(found)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "False" in nb_runner.get_output(2)
