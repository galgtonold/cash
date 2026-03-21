"""
Batch 336: dict merge and walrus operator patterns with caching.
Tests dict merge (|), walrus operator (:=), and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictMergeWalrus:
    """Test dict merge and walrus operator caching."""

    def test_dict_merge_operator(self, nb_runner):
        """Dict merge with | operator, verify caching."""
        nb_runner.create_notebook([
            "d1 = {'a': 1, 'b': 2}",
            "d2 = {'b': 3, 'c': 4}",
            "merged = d1 | d2\nprint(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'a': 1" in out
        assert "'b': 3" in out
        assert "'c': 4" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'b': 3" in out2

    def test_dict_merge_edit(self, nb_runner):
        """Edit one dict, verify merge result updates."""
        nb_runner.create_notebook([
            "base = {'x': 10, 'y': 20}",
            "override = {'y': 99}",
            "final = base | override\nprint(f'y={final[\"y\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "y=99" in out

        nb_runner.set_cell_source(2, "override = {'y': 42}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "y=42" in out2

    def test_walrus_in_condition(self, nb_runner):
        """Walrus operator in condition with caching."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "evens = [x for x in data if (y := x % 2) == 0]\ncount = len(evens)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=5" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "count=5" in out2
