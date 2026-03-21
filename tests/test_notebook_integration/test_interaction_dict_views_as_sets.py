"""Batch 449: dict views (keys, values, items) as sets."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictViewsAsSets:
    def test_keys_intersection(self, nb_runner):
        nb_runner.create_notebook([
            "d1 = {'a': 1, 'b': 2, 'c': 3}\nd2 = {'b': 20, 'c': 30, 'd': 40}",
            "common = sorted(d1.keys() & d2.keys())\nonly_d1 = sorted(d1.keys() - d2.keys())\nprint(f'common={common} only_d1={only_d1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common=['b', 'c']" in nb_runner.get_output(2)
        assert "only_d1=['a']" in nb_runner.get_output(2)

    def test_items_as_set(self, nb_runner):
        nb_runner.create_notebook([
            "d1 = {'a': 1, 'b': 2}\nd2 = {'a': 1, 'b': 3}",
            "common_items = sorted(d1.items() & d2.items())\nprint(f'common_items={common_items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "common_items=[('a', 1)]" in nb_runner.get_output(2)

    def test_dict_views_edit(self, nb_runner):
        nb_runner.create_notebook([
            "d = {'x': 10, 'y': 20, 'z': 30}",
            "vals = sorted(d.values())\nkeys = sorted(d.keys())\nprint(f'keys={keys} vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y', 'z']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "d = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        assert "keys=['a', 'b']" in nb_runner.get_output(2)
        assert "vals=[100, 200]" in nb_runner.get_output(2)
