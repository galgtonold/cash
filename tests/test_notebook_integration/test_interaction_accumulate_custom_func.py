"""
Interaction test: itertools accumulate with custom function.
Tests itertools.accumulate with operator.mul, custom functions,
initial value, and cross-cell running computation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAccumulateCustomFunc:
    """Test itertools.accumulate with custom function across cells."""

    def test_accumulate_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: running sum and product
            "from itertools import accumulate\nimport operator\nnums = [1, 2, 3, 4, 5]\nrunning_sum = list(accumulate(nums))\nrunning_prod = list(accumulate(nums, operator.mul))\nprint(f'sum={running_sum}')\nprint(f'prod={running_prod}')",
            # Cell 2: custom max accumulate
            "running_max = list(accumulate(nums, max))\nprint(f'max={running_max}')",
            # Cell 3: with initial value
            "with_init = list(accumulate(nums, operator.add, initial=100))\nprint(f'with_init={with_init}')\nprint(f'final={with_init[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sum=[1, 3, 6, 10, 15]" in out1
        assert "prod=[1, 2, 6, 24, 120]" in out1
        out2 = nb_runner.get_output(2)
        assert "max=[1, 2, 3, 4, 5]" in out2
        out3 = nb_runner.get_output(3)
        assert "with_init=[100, 101, 103, 106, 110, 115]" in out3
        assert "final=115" in out3

    def test_accumulate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import accumulate\ndata = [10, 20, 30]\nresult = list(accumulate(data))\nprint(f'result={result}')",
            "last = result[-1]\nprint(f'last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[10, 30, 60]" in nb_runner.get_output(1)
        assert "last=60" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(1, "from itertools import accumulate\ndata = [10, 20, 30, 40]\nresult = list(accumulate(data))\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "last=100" in nb_runner.get_output(2)

    def test_accumulate_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import accumulate\nimport operator\nfactors = [2, 3, 5, 7]\nrunning = list(accumulate(factors, operator.mul))\nprint(f'running={running}')",
            "final_prod = running[-1]\nprint(f'final={final_prod}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running=[2, 6, 30, 210]" in nb_runner.get_output(1)
        assert "final=210" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "final=210" in nb_runner.get_output(2)
