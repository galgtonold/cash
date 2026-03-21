"""Batch 346: dict.setdefault, dict.update, and chained dict ops."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictAdvancedOps:
    def test_setdefault_grouping(self, nb_runner):
        nb_runner.create_notebook([
            "items = [('fruit', 'apple'), ('veg', 'carrot'), ('fruit', 'banana'), ('veg', 'pea')]",
            "groups = {}\nfor cat, item in items:\n    groups.setdefault(cat, []).append(item)\nprint(f'groups={dict(sorted(groups.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fruit" in nb_runner.get_output(2)
        assert "'apple', 'banana'" in nb_runner.get_output(2)

    def test_dict_update_edit(self, nb_runner):
        nb_runner.create_notebook([
            "base = {'a': 1, 'b': 2}",
            "overlay = {'b': 20, 'c': 30}\nmerged = {**base, **overlay}\nprint(f'merged={dict(sorted(merged.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged={'a': 1, 'b': 20, 'c': 30}" in nb_runner.get_output(2)
        # Edit base
        nb_runner.set_cell_source(1, "base = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        assert "merged={'a': 100, 'b': 20, 'c': 30}" in nb_runner.get_output(2)

    def test_dict_pop_get(self, nb_runner):
        nb_runner.create_notebook([
            "d = {'x': 10, 'y': 20, 'z': 30}",
            "val = d.get('w', -1)\nprint(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=-1" in nb_runner.get_output(2)
