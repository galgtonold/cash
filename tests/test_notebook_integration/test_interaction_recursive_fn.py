"""Batch 241 – Recursive function edit propagation.

Tests editing recursive functions and verifying downstream re-evaluation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveFunctionEdit:
    """Edit recursive functions, verify downstream propagation."""

    def test_factorial_edit(self, nb_runner):
        """Edit factorial base case, downstream recalculates."""
        nb_runner.create_notebook([
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "val = factorial(5)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 120" in nb_runner.get_output(2)

        # Change to double factorial base case
        nb_runner.set_cell_source(
            1,
            "def factorial(n):\n    if n <= 1:\n        return 2\n    return n * factorial(n - 1)",
        )
        nb_runner.run_all()
        # 5*4*3*2*2 = 240
        assert "val = 240" in nb_runner.get_output(2)

    def test_fibonacci_edit(self, nb_runner):
        """Edit fibonacci recurrence, downstream updates."""
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

    def test_recursive_sum_edit(self, nb_runner):
        """Edit recursive list sum approach."""
        nb_runner.create_notebook([
            "def rsum(lst):\n    if not lst:\n        return 0\n    return lst[0] + rsum(lst[1:])",
            "data = [10, 20, 30, 40]\nval = rsum(data)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change to product
        nb_runner.set_cell_source(
            1,
            "def rsum(lst):\n    if not lst:\n        return 1\n    return lst[0] * rsum(lst[1:])",
        )
        nb_runner.run_all()
        # 10*20*30*40 = 240000
        assert "val = 240000" in nb_runner.get_output(2)
