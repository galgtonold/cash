"""
Generator, iterator, and functional patterns — yield, send/throw,
itertools, functools, closures, higher-order functions.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestGeneratorPatterns:
    """Test caching with generator functions and itertools."""

    def test_generator_consumed_to_list(self, nb_runner):
        """Generator consumed to list should cache."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def gen_range(n):
                    for i in range(n):
                        yield i * i
            """),
                textwrap.dedent("""\
                result = list(gen_range(5))
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run - should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_generator_expression(self, nb_runner):
        """Generator expression consumed to sum."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 3, 4, 5]",
                textwrap.dedent("""\
                total = sum(x**2 for x in values)
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "55" in nb_runner.get_output(2)

    def test_generator_function_change_propagation(self, nb_runner):
        """Change generator function → re-run consumer."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def fib(n):
                    a, b = 0, 1
                    for _ in range(n):
                        yield a
                        a, b = b, a + b
            """),
                textwrap.dedent("""\
                result = list(fib(6))
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 1, 2, 3, 5]" in nb_runner.get_output(2)

        # Change to start from 1, 1
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def fib(n):
                a, b = 1, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b
        """),
        )
        nb_runner.run_all()
        assert "[1, 1, 2, 3, 5, 8]" in nb_runner.get_output(2)


class TestClosurePatterns:
    """Test caching with closures and higher-order functions."""

    def test_higher_order_map_filter(self, nb_runner):
        """Higher-order functions: map + filter."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                textwrap.dedent("""\
                evens = list(filter(lambda x: x % 2 == 0, data))
                squared = list(map(lambda x: x**2, evens))
                print(squared)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[4, 16, 36, 64, 100]" in nb_runner.get_output(2)


class TestFunctoolsPatterns:
    """Test caching with functools utilities."""

    def test_functools_lru_cache(self, nb_runner):
        """@lru_cache decorator on a function."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def expensive(n):
                    return sum(range(n))
            """),
                textwrap.dedent("""\
                result = expensive(1000)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "499500" in nb_runner.get_output(3)
