"""Batch 275 – Multi-round iterative refinement patterns.

Tests multiple sequential edits to same cell, verifying each round.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiRoundRefinement:
    """Multiple rounds of editing the same cell."""

    def test_five_round_formula_refinement(self, nb_runner):
        """Edit formula cell 5 times sequentially."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Round 1: add 5
        nb_runner.set_cell_source(2, "result = x + 5\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Round 2: multiply
        nb_runner.set_cell_source(2, "result = x * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Round 3: power
        nb_runner.set_cell_source(2, "result = x ** 2\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Round 4: floor div
        nb_runner.set_cell_source(2, "result = x // 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)

        # Round 5: complex expression
        nb_runner.set_cell_source(2, "result = (x + 1) * (x - 1)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 99" in nb_runner.get_output(2)

    def test_alternating_data_source(self, nb_runner):
        """Alternate data sources, verify correct propagation each time."""
        nb_runner.create_notebook([
            "source = [1, 2, 3]",
            "total = sum(source)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        for vals, expected in [
            ([10, 20], 30),
            ([100], 100),
            ([5, 5, 5, 5], 20),
            ([1000, 2000, 3000], 6000),
        ]:
            nb_runner.set_cell_source(1, f"source = {vals}")
            nb_runner.run_all()
            assert f"total = {expected}" in nb_runner.get_output(2)

    def test_refine_function_multiple_times(self, nb_runner):
        """Refine function definition through multiple iterations."""
        nb_runner.create_notebook([
            "def transform(x):\n    return x",
            "vals = [1, 2, 3, 4, 5]\nresult = [transform(v) for v in vals]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)

        # v2: double
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 2")
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # v3: square
        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

        # v4: negate
        nb_runner.set_cell_source(1, "def transform(x):\n    return -x")
        nb_runner.run_all()
        assert "result = [-1, -2, -3, -4, -5]" in nb_runner.get_output(2)
