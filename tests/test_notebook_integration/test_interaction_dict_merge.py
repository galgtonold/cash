"""Batch 247 – Dict merge/update operator patterns.

Tests dict union (|), update, merge patterns with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictMergePatterns:
    """Dict merge and update patterns with edit propagation."""

    def test_dict_union_operator(self, nb_runner):
        """Edit one dict in union, merged result updates."""
        nb_runner.create_notebook([
            "defaults = {'color': 'blue', 'size': 10}",
            "overrides = {'size': 20, 'shape': 'circle'}",
            "config = defaults | overrides\nprint(f'config = {config}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'color': 'blue'" in out
        assert "'size': 20" in out
        assert "'shape': 'circle'" in out

        nb_runner.set_cell_source(1, "defaults = {'color': 'red', 'size': 10, 'weight': 5}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'color': 'red'" in out2
        assert "'weight': 5" in out2

    def test_dict_comprehension_merge(self, nb_runner):
        """Edit source dict, comprehension downstream updates."""
        nb_runner.create_notebook([
            "prices = {'apple': 1.0, 'banana': 0.5, 'cherry': 2.0}",
            "discount = 0.8",
            "sale = {k: round(v * discount, 2) for k, v in prices.items()}\nprint(f'sale = {sale}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'apple': 0.8" in out
        assert "'banana': 0.4" in out

        nb_runner.set_cell_source(2, "discount = 0.5")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "'apple': 0.5" in out2
        assert "'cherry': 1.0" in out2

    def test_nested_dict_edit(self, nb_runner):
        """Edit nested dict structure, downstream uses nested access."""
        nb_runner.create_notebook([
            "db = {'users': {'alice': 30, 'bob': 25}, 'version': 1}",
            "names = list(db['users'].keys())\nages = list(db['users'].values())\nprint(f'names={names} ages={ages}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['alice', 'bob']" in nb_runner.get_output(2)
        assert "ages=[30, 25]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "db = {'users': {'charlie': 40, 'diana': 35, 'eve': 28}, 'version': 2}",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "charlie" in out2
        assert "40" in out2
