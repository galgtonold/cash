"""Batch 477: functools reduce and operator module."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsReduceOperator:
    def test_reduce_sum(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nimport operator",
            "nums = [1, 2, 3, 4, 5]\nproduct = reduce(operator.mul, nums)\ntotal = reduce(operator.add, nums)\nprint(f'product={product} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "product=120" in out
        assert "total=15" in out

    def test_reduce_nested(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce",
            "lists = [[1, 2], [3, 4], [5]]\nflat = reduce(lambda a, b: a + b, lists)\nprint(f'flat={flat} len={len(flat)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "flat=[1, 2, 3, 4, 5]" in out
        assert "len=5" in out

    def test_reduce_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nimport operator",
            "vals = [2, 3, 4]\nresult = reduce(operator.mul, vals)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=24" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "vals = [10, 20, 30]\nresult = reduce(operator.add, vals)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=60" in nb_runner.get_output(2)
