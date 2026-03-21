"""Batch 129 – Lambda & higher-order function interaction tests.

Tests that exercise lambda expressions, map/filter/reduce, and
higher-order function patterns with cell edits.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestLambdaEdits:
    """Lambda expressions + cell edits."""

    def test_edit_lambda(self, nb_runner):
        """Edit a lambda expression."""
        nb_runner.create_notebook([
            "fn = lambda x: x * 2",
            "result = fn(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "fn = lambda x: x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_lambda_with_closure(self, nb_runner):
        """Lambda capturing a variable, edit the captured value."""
        nb_runner.create_notebook([
            "factor = 3",
            "scale = lambda x: x * factor",
            "result = scale(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "factor = 10")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(3)

    def test_lambda_as_argument(self, nb_runner):
        """Lambda used as argument, edit the lambda."""
        nb_runner.create_notebook([
            "data = [3, 1, 4, 1, 5, 9, 2, 6]",
            "sorted_data = sorted(data, key=lambda x: x)",
            "print(f'sorted = {sorted_data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sorted = [1, 1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(3)

        # Sort in reverse
        nb_runner.set_cell_source(
            2, "sorted_data = sorted(data, key=lambda x: -x)"
        )
        nb_runner.run_all()
        assert "sorted = [9, 6, 5, 4, 3, 2, 1, 1]" in nb_runner.get_output(3)


class TestMapFilterReduce:
    """map/filter/reduce + cell edits."""

    def test_map_edit(self, nb_runner):
        """Edit the function passed to map."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "mapped = list(map(lambda x: x * 2, data))",
            "result = sum(mapped)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "mapped = list(map(lambda x: x ** 2, data))"
        )
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(3)

    def test_filter_edit(self, nb_runner):
        """Edit the filter condition."""
        nb_runner.create_notebook([
            "data = list(range(1, 11))",
            "filtered = list(filter(lambda x: x > 5, data))",
            "result = len(filtered)\nprint(f'count = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "filtered = list(filter(lambda x: x % 2 == 0, data))"
        )
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(3)

    def test_chain_map_filter(self, nb_runner):
        """Chain map and filter, edit one."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "step1 = list(map(lambda x: x * 2, data))",
            "step2 = list(filter(lambda x: x > 10, step1))",
            "result = sum(step2)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # map: [2,4,6,8,10,12,14,16,18,20], filter >10: [12,14,16,18,20] = 80
        assert "result = 80" in nb_runner.get_output(4)

        # Change map
        nb_runner.set_cell_source(
            2, "step1 = list(map(lambda x: x * 3, data))"
        )
        nb_runner.run_all()
        # map: [3,6,9,12,15,18,21,24,27,30], filter >10: [12,15,18,21,24,27,30] = 147
        assert "result = 147" in nb_runner.get_output(4)


class TestHigherOrderFunctions:
    """Higher-order function patterns + edits."""

    def test_apply_function(self, nb_runner):
        """Apply a function to data, edit the function."""
        nb_runner.create_notebook([
            "def apply_fn(fn, data):\n    return [fn(x) for x in data]",
            "transform = lambda x: x + 1",
            "data = [1, 2, 3]",
            "result = apply_fn(transform, data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "transform = lambda x: x * 10")
        nb_runner.run_all()
        assert "result = [10, 20, 30]" in nb_runner.get_output(4)

    def test_compose_functions(self, nb_runner):
        """Compose two functions, edit one."""
        nb_runner.create_notebook([
            "def compose(f, g):\n    return lambda x: f(g(x))",
            "double = lambda x: x * 2",
            "add_one = lambda x: x + 1",
            "pipeline = compose(double, add_one)",
            "result = pipeline(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # add_one(5) = 6, double(6) = 12
        assert "result = 12" in nb_runner.get_output(5)

        nb_runner.set_cell_source(2, "double = lambda x: x * 10")
        nb_runner.run_all()
        # add_one(5) = 6, double(6) = 60
        assert "result = 60" in nb_runner.get_output(5)
