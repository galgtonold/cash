"""
Batch 326: functools.reduce and accumulate patterns with caching.
Tests reduce, accumulate, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestReduceAccumulate:
    """Test functools.reduce and itertools.accumulate caching."""

    def test_reduce_sum(self, nb_runner):
        """functools.reduce for summation with caching."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "nums = [1, 2, 3, 4, 5]",
            "total = reduce(lambda a, b: a + b, nums)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=15" in out2

    def test_reduce_edit(self, nb_runner):
        """Edit input, verify reduce result changes."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "nums = [2, 3, 4]",
            "product = reduce(lambda a, b: a * b, nums)",
            "print(f'product={product}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "product=24" in out

        nb_runner.set_cell_source(2, "nums = [2, 3, 4, 5]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "product=120" in out2

    def test_accumulate_pattern(self, nb_runner):
        """itertools.accumulate running totals."""
        nb_runner.create_notebook([
            "from itertools import accumulate",
            "payments = [100, 200, 150, 300]",
            "running = list(accumulate(payments))\nlast = running[-1]",
            "print(f'last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "last=750" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "last=750" in out2
