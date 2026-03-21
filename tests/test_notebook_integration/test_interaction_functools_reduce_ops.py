"""
Interaction test: functools reduce with various operators.
Tests functools.reduce for accumulation with different operators,
initial values, and cross-cell reduction pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsReduceOps:
    """Test functools.reduce with various operators across cells."""

    def test_reduce_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic reduce operations
            "from functools import reduce\nimport operator\nnums = [1, 2, 3, 4, 5]\nproduct = reduce(operator.mul, nums)\ntotal = reduce(operator.add, nums)\nprint(f'product={product}')\nprint(f'total={total}')",
            # Cell 2: reduce with initial value
            "concatenated = reduce(lambda acc, x: acc + str(x), nums, 'nums:')\nmax_val = reduce(lambda a, b: a if a > b else b, nums)\nprint(f'concat={concatenated}')\nprint(f'max={max_val}')",
            # Cell 3: nested reduce
            "matrix = [[1, 2], [3, 4], [5, 6]]\nflat = reduce(operator.add, matrix)\nflat_sum = reduce(operator.add, flat)\nprint(f'flat={flat}')\nprint(f'flat_sum={flat_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "product=120" in out1
        assert "total=15" in out1
        out2 = nb_runner.get_output(2)
        assert "concat=nums:12345" in out2
        assert "max=5" in out2
        out3 = nb_runner.get_output(3)
        assert "flat=[1, 2, 3, 4, 5, 6]" in out3
        assert "flat_sum=21" in out3

    def test_reduce_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nimport operator\ndata = [2, 3, 4]\nresult = reduce(operator.mul, data)\nprint(f'result={result}')",
            "info = f'product of {len(data)} numbers = {result}'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=product of 3 numbers = 24" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(1, "from functools import reduce\nimport operator\ndata = [2, 3, 4, 5]\nresult = reduce(operator.mul, data)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "info=product of 4 numbers = 120" in nb_runner.get_output(2)

    def test_reduce_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\nwords = ['hello', 'world', 'foo']\nlongest = reduce(lambda a, b: a if len(a) >= len(b) else b, words)\nprint(f'longest={longest}')",
            "length = len(longest)\nprint(f'length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "longest=hello" in nb_runner.get_output(1)
        assert "length=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=5" in nb_runner.get_output(2)
