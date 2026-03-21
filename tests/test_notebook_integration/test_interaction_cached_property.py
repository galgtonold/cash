"""Batch 220 – Cached property and memoization interaction tests.

Tests editing cells with memoization patterns
and verifying correct cache invalidation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMemoizationEdits:
    """Editing memoization patterns."""

    def test_edit_memoized_function(self, nb_runner):
        """Edit a function that uses manual memoization."""
        nb_runner.create_notebook([
            "def fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]",
            "result = fib(10)\nprint(f'fib(10) = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib(10) = 55" in nb_runner.get_output(2)

        # Change to call with different arg
        nb_runner.set_cell_source(2, "result = fib(15)\nprint(f'fib(15) = {result}')")
        nb_runner.run_all()
        assert "fib(15) = 610" in nb_runner.get_output(2)

    def test_edit_lru_cache_function(self, nb_runner):
        """Edit function using lru_cache."""
        nb_runner.create_notebook([
            "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
            "result = factorial(5)\nprint(f'5! = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5! = 120" in nb_runner.get_output(2)

        # Change call
        nb_runner.set_cell_source(2, "result = factorial(7)\nprint(f'7! = {result}')")
        nb_runner.run_all()
        assert "7! = 5040" in nb_runner.get_output(2)

    def test_edit_lookup_table(self, nb_runner):
        """Edit a precomputed lookup table."""
        nb_runner.create_notebook([
            "squares = {i: i**2 for i in range(10)}",
            "vals = [squares[x] for x in [1, 3, 5, 7]]\nprint(f'vals = {vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 9, 25, 49]" in nb_runner.get_output(2)

        # Change to cubes
        nb_runner.set_cell_source(1, "squares = {i: i**3 for i in range(10)}")
        nb_runner.run_all()
        assert "vals = [1, 27, 125, 343]" in nb_runner.get_output(2)
