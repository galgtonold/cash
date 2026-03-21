"""
Batch 308: Mapping and filtering with functions interaction tests.
Tests that editing filter/map functions or data properly invalidates downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMapFilterInteraction:
    """Test map/filter function patterns with cache invalidation."""

    def test_map_function_edit(self, nb_runner):
        """Editing the mapping function should propagate."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "def transform(x):\n    return x * 2",
            "mapped = list(map(transform, data))",
            "result = sum(mapped)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=30" in out

        nb_runner.set_cell_source(2, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=55" in out

    def test_filter_predicate_edit(self, nb_runner):
        """Editing the filter predicate should propagate."""
        nb_runner.create_notebook([
            "numbers = list(range(1, 11))",
            "def is_valid(x):\n    return x % 2 == 0",
            "filtered = list(filter(is_valid, numbers))",
            "result = ','.join(str(x) for x in filtered)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=2,4,6,8,10" in out

        nb_runner.set_cell_source(2, "def is_valid(x):\n    return x % 3 == 0")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=3,6,9" in out

    def test_reduce_edit(self, nb_runner):
        """Editing the reduce function should propagate."""
        nb_runner.create_notebook([
            "from functools import reduce\ndata = [1, 2, 3, 4]",
            "def combine(acc, x):\n    return acc + x",
            "result = reduce(combine, data, 0)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "def combine(acc, x):\n    return acc * x")
        nb_runner.set_cell_source(3, "result = reduce(combine, data, 1)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=24" in out
