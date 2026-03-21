"""
Batch 314: Aggregation with groupby interaction tests.
Tests that editing groupby logic or data properly invalidates
aggregated results downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestGroupbyAggregationInteraction:
    """Test groupby/aggregation patterns with cache invalidation."""

    def test_manual_groupby_edit(self, nb_runner):
        """Editing data grouped manually should propagate."""
        nb_runner.create_notebook([
            "data = [('a', 1), ('b', 2), ('a', 3), ('b', 4), ('a', 5)]",
            "groups = {}\nfor key, val in data:\n    groups.setdefault(key, []).append(val)",
            "sums = {k: sum(v) for k, v in sorted(groups.items())}",
            "print(f'sums={sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sums={'a': 9, 'b': 6}" in out

        nb_runner.set_cell_source(1, "data = [('x', 10), ('y', 20), ('x', 30)]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sums={'x': 40, 'y': 20}" in out

    def test_itertools_groupby_edit(self, nb_runner):
        """Editing sorted data for itertools.groupby should propagate."""
        nb_runner.create_notebook([
            "from itertools import groupby\ndata = sorted([('a', 1), ('b', 2), ('a', 3), ('b', 4)])",
            "grouped = {k: [v for _, v in g] for k, g in groupby(data, key=lambda x: x[0])}",
            "counts = {k: len(v) for k, v in sorted(grouped.items())}",
            "print(f'counts={counts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "counts={'a': 2, 'b': 2}" in out

        nb_runner.set_cell_source(1, "from itertools import groupby\ndata = sorted([('a', 1), ('a', 2), ('a', 3), ('b', 4)])")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "counts={'a': 3, 'b': 1}" in out
