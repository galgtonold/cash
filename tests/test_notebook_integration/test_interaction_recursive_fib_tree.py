"""Batch 519: recursive functions fibonacci and tree."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveFibTree:
    def test_fibonacci_memo(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache",
            "@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nresults = [fib(i) for i in range(10)]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]" in nb_runner.get_output(2)

    def test_recursive_flatten(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "def depth(lst):\n    if not isinstance(lst, list): return 0\n    if not lst: return 1\n    return 1 + max(depth(item) for item in lst)\ntree = [1, [2, [3, [4]]]]\nprint(f'depth={depth(tree)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth=4" in nb_runner.get_output(2)

    def test_recursive_edit(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nresult = factorial(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nresult = factorial(7)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=5040" in nb_runner.get_output(2)
