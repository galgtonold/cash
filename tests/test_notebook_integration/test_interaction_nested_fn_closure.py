"""Batch 348: nested function definitions and closures with edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNestedFunctionClosure:
    def test_closure_counter(self, nb_runner):
        nb_runner.create_notebook([
            "def make_counter(start=0):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
            "counter = make_counter(10)\nresults = [counter() for _ in range(3)]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[11, 12, 13]" in nb_runner.get_output(2)

    def test_closure_edit_factory(self, nb_runner):
        nb_runner.create_notebook([
            "def multiplier(factor):\n    def multiply(x):\n        return x * factor\n    return multiply",
            "double = multiplier(2)\nresult = double(7)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=14" in nb_runner.get_output(2)
        # Edit factory
        nb_runner.set_cell_source(1, "def multiplier(factor):\n    def multiply(x):\n        return x * factor + 1\n    return multiply")
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)

    def test_nested_helper(self, nb_runner):
        nb_runner.create_notebook([
            "def process(items):\n    def clean(s):\n        return s.strip().lower()\n    return [clean(i) for i in items]",
            "data = ['  Hello ', ' WORLD ', '  Python  ']\nresult = process(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['hello', 'world', 'python']" in nb_runner.get_output(2)
