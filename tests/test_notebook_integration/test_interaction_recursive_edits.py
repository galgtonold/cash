"""Batch 169 – Recursive function interaction tests.

Tests editing recursive function definitions, base cases,
and recursive steps, verifying correct recomputation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestRecursiveFunctionEdits:
    """Editing recursive function bodies."""

    def test_edit_base_case(self, nb_runner):
        """Edit the base case of a recursive function."""
        nb_runner.create_notebook([
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "result = factorial(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        # Change base case to return 2
        nb_runner.set_cell_source(
            1,
            "def factorial(n):\n    if n <= 1:\n        return 2\n    return n * factorial(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 240" in nb_runner.get_output(2)

    def test_edit_recursive_step(self, nb_runner):
        """Edit the recursive step."""
        nb_runner.create_notebook([
            "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            "result = fib(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(2)

        # Change to tribonacci
        nb_runner.set_cell_source(
            1,
            "def fib(n):\n    if n <= 1:\n        return n\n    if n == 2:\n        return 1\n    return fib(n-1) + fib(n-2) + fib(n-3)",
        )
        nb_runner.run_all()
        # tribonacci(7) = 24
        assert "result = 24" in nb_runner.get_output(2)

    def test_add_memoization(self, nb_runner):
        """Add memoization to a recursive function."""
        nb_runner.create_notebook([
            "def slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
            "result = slow_sum(100)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

        # Add caching
        nb_runner.set_cell_source(
            1,
            "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

    def test_recursive_with_helper(self, nb_runner):
        """Edit a recursive function that calls a helper."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "def recurse(n):\n    if n <= 0:\n        return 0\n    return double(n) + recurse(n - 1)",
            "result = recurse(4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # double(4)+double(3)+double(2)+double(1) = 8+6+4+2 = 20
        assert "result = 20" in nb_runner.get_output(3)

        # Edit helper
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_all()
        # 12+9+6+3 = 30
        assert "result = 30" in nb_runner.get_output(3)
