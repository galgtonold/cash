"""Batch 400: functools.reduce and accumulate patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsReduceAccumulate:
    def test_reduce_sum(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nnums = [1, 2, 3, 4, 5]",
            "total = reduce(lambda a, b: a + b, nums)\nproduct = reduce(lambda a, b: a * b, nums)\nprint(f'total={total} product={product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)
        assert "product=120" in nb_runner.get_output(2)

    def test_reduce_with_initial(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nitems = ['a', 'b', 'c']",
            "result = reduce(lambda acc, x: acc + '-' + x, items, 'start')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=start-a-b-c" in nb_runner.get_output(2)

    def test_itertools_accumulate(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import accumulate\nimport operator\nvals = [1, 2, 3, 4, 5]",
            "running_sum = list(accumulate(vals))\nrunning_product = list(accumulate(vals, operator.mul))\nprint(f'sums={running_sum} products={running_product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sums=[1, 3, 6, 10, 15]" in out
        assert "products=[1, 2, 6, 24, 120]" in out
