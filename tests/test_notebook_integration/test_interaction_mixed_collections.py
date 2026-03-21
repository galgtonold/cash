"""Batch 200 – Mixed-type collection operation interaction tests.

Tests editing operations on collections containing mixed types
(lists of dicts, dicts of lists, nested structures).
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMixedCollectionEdits:
    """Editing mixed-type collection operations."""

    def test_edit_list_of_dicts(self, nb_runner):
        """Edit operations on a list of dicts."""
        nb_runner.create_notebook([
            "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}]",
            "names = [r['name'] for r in records]\nprint(f'names = {names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(2)

        # Change to extract scores
        nb_runner.set_cell_source(
            2, "scores = [r['score'] for r in records]\nprint(f'scores = {scores}')"
        )
        nb_runner.run_all()
        assert "scores = [90, 85]" in nb_runner.get_output(2)

    def test_edit_dict_of_lists(self, nb_runner):
        """Edit operations on a dict of lists."""
        nb_runner.create_notebook([
            "data = {'fruits': ['apple', 'banana'], 'vegs': ['carrot', 'pea']}",
            "all_items = []\nfor k in data:\n    all_items.extend(data[k])\nprint(f'count = {len(all_items)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(
            1,
            "data = {'fruits': ['apple'], 'vegs': ['carrot', 'pea', 'bean'], 'grains': ['rice']}",
        )
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(2)

    def test_edit_nested_structure_access(self, nb_runner):
        """Edit access patterns on deeply nested structures."""
        nb_runner.create_notebook([
            "tree = {'a': {'b': {'c': 42}}}  # nested tree",
            "val = tree['a']['b']['c']\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change tree
        nb_runner.set_cell_source(
            1, "tree = {'a': {'b': {'c': 99, 'd': 100}}}  # nested tree v2"
        )
        nb_runner.set_cell_source(
            2, "val = tree['a']['b']['c'] + tree['a']['b']['d']\nprint(f'val = {val}')"
        )
        nb_runner.run_all()
        assert "val = 199" in nb_runner.get_output(2)

    def test_edit_zip_combination(self, nb_runner):
        """Edit zip-based combination of collections."""
        nb_runner.create_notebook([
            "keys = ['x', 'y', 'z']  # zip source keys",
            "vals = [10, 20, 30]  # zip source vals",
            "combined = dict(zip(keys, vals))\nprint(f'combined = {combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'x': 10" in nb_runner.get_output(3)

        # Change values
        nb_runner.set_cell_source(2, "vals = [100, 200, 300]  # zip source vals v2")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'x': 100" in out
        assert "'z': 300" in out
