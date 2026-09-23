"""Walrus operator (:=) & complex comprehensions — cash caching."""

import textwrap

import pytest


@pytest.mark.stress
class TestWalrusOperator:
    """Test walrus operator patterns across cells."""

    def test_walrus_propagation(self, nb_runner):
        """Walrus operator result propagation on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                numbers = [2, 8, 3, 15, 7, 20, 1]
                big = [y for x in numbers if (y := x * 2) > 10]
                print(f"big={big}")
            """),
                textwrap.dedent("""\
                count = len(big)
                print(f"count={count}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big=[16, 30, 14, 40]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)

        # Change threshold by changing input
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            numbers = [2, 8, 3, 15, 7, 20, 1]
            big = [y for x in numbers if (y := x * 3) > 10]
            print(f"big={big}")
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "big=[24, 45, 21, 60]" in nb_runner.get_output(1)
        assert "count=4" in nb_runner.get_output(2)


@pytest.mark.stress
class TestComplexComprehensions:
    """Test complex comprehension patterns."""

    def test_set_comprehension_with_condition(self, nb_runner):
        """Set comprehension with multiple conditions."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
                evens = {val for row in matrix for val in row if val % 2 == 0 if val > 4}
                print(f"evens={sorted(evens)}")
            """),
                textwrap.dedent("""\
                total = sum(evens)
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[6, 8, 10, 12]" in nb_runner.get_output(1)
        assert "total=36" in nb_runner.get_output(2)

    def test_generator_expression_chain(self, nb_runner):
        """Chained generator expressions."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                words = ['Hello World', 'Python Programming', 'Cash Library']
                letters = list(c.lower() for phrase in words for c in phrase if c.isalpha())
                freq = {}
                for ch in letters:
                    freq[ch] = freq.get(ch, 0) + 1
                top3 = sorted(freq.items(), key=lambda x: -x[1])[:3]
                print(f"top3={top3}")
            """),
                textwrap.dedent("""\
                total_letters = len(letters)
                print(f"total={total_letters}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "top3=" in out1
        out2 = nb_runner.get_output(2)
        assert "total=" in out2
