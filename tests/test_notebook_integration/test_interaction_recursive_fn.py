"""Batch 241 – Recursive function edit propagation.

Tests editing recursive functions and verifying downstream re-evaluation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveFunctionEdit:
    """Edit recursive functions, verify downstream propagation."""



    def test_recursive_sum_edit(self, nb_runner):
        """Edit recursive list sum approach."""
        nb_runner.create_notebook([
            "def rsum(lst):\n    if not lst:\n        return 0\n    return lst[0] + rsum(lst[1:])",
            "data = [10, 20, 30, 40]\nval = rsum(data)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change to product
        nb_runner.set_cell_source(
            1,
            "def rsum(lst):\n    if not lst:\n        return 1\n    return lst[0] * rsum(lst[1:])",
        )
        nb_runner.run_all()
        # 10*20*30*40 = 240000
        assert "val = 240000" in nb_runner.get_output(2)
