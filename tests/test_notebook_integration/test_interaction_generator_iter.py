"""
Batch 333: generator/iterator protocol patterns with caching.
Tests generator functions, iter/next, StopIteration, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestGeneratorIterator:
    """Test generator and iterator protocol caching."""

    def test_generator_function(self, nb_runner):
        """Generator function producing values with caching."""
        nb_runner.create_notebook([
            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
            "n = 8",
            "result = list(fib(n))\nprint(f'fib={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "fib=[0, 1, 1, 2, 3, 5, 8, 13]" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "fib=[0, 1, 1, 2, 3, 5, 8, 13]" in out2

    def test_generator_edit_count(self, nb_runner):
        """Edit generator count parameter, verify output changes."""
        nb_runner.create_notebook([
            "def squares(n):\n    for i in range(n):\n        yield i ** 2",
            "count = 5",
            "result = list(squares(count))\nprint(f'squares={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "squares=[0, 1, 4, 9, 16]" in out

        nb_runner.set_cell_source(2, "count = 3")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "squares=[0, 1, 4]" in out2

    def test_generator_expression(self, nb_runner):
        """Generator expression with sum, verify caching."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "total = sum(x**2 for x in data)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=55" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "total=55" in out2
