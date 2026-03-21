"""Batch 508: generator expressions and yield from."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGeneratorYieldFrom:
    def test_generator_expression(self, nb_runner):
        nb_runner.create_notebook([
            "data = range(1, 11)",
            "squares_sum = sum(x**2 for x in data)\neven_sum = sum(x for x in data if x % 2 == 0)\nprint(f'squares_sum={squares_sum} even_sum={even_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "squares_sum=385" in out
        assert "even_sum=30" in out

    def test_yield_from(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "def flatten(nested):\n    for item in nested:\n        if isinstance(item, list):\n            yield from flatten(item)\n        else:\n            yield item\ndata = [1, [2, 3], [4, [5, 6]], 7]\nresult = list(flatten(data))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5, 6, 7]" in nb_runner.get_output(2)

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook([
            "n = 5",
            "result = sum(i for i in range(n))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "n = 10")
        nb_runner.run_all()
        assert "result=45" in nb_runner.get_output(2)
