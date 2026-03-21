"""Batch 265 – Nested function definition edit patterns.

Tests inner function edits propagating through outer function calls.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNestedFunctionEdits:
    """Nested function definition patterns."""

    def test_inner_function_edit(self, nb_runner):
        """Edit inner function, outer function result changes."""
        nb_runner.create_notebook([
            "def outer(data):\n    def inner(x):\n        return x * 2\n    return [inner(x) for x in data]",
            "result = outer([1, 2, 3, 4])\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def outer(data):\n    def inner(x):\n        return x ** 2\n    return [inner(x) for x in data]",
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_nested_with_accumulator(self, nb_runner):
        """Edit inner accumulator function."""
        nb_runner.create_notebook([
            "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x\n        return total\n    return [accumulate(i) for i in items]",
            "running = process([10, 20, 30])\nprint(f'running = {running}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running = [10, 30, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x * 2\n        return total\n    return [accumulate(i) for i in items]",
        )
        nb_runner.run_all()
        assert "running = [20, 60, 120]" in nb_runner.get_output(2)

    def test_factory_with_inner_edit(self, nb_runner):
        """Edit factory function that produces inner functions."""
        nb_runner.create_notebook([
            "def make_processor(op):\n    def process(x):\n        if op == 'double':\n            return x * 2\n        return x + 1\n    return process",
            "proc = make_processor('double')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "proc = make_processor('other')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')")
        nb_runner.run_all()
        assert "results = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)
