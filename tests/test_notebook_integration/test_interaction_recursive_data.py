"""Batch 217 – Recursive data structure interaction tests.

Tests editing cells with recursive data structures
(trees, linked lists) and verifying propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestRecursiveDataEdits:
    """Editing recursive data structure patterns."""

    def test_edit_tree_data(self, nb_runner):
        """Edit a nested dict tree and check traversal."""
        nb_runner.create_notebook([
            "tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}",
            "def collect(node):\n    if node is None:\n        return []\n    return [node['val']] + collect(node['left']) + collect(node['right'])\nvals = collect(tree)\nprint(f'vals = {vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 2, 3]" in nb_runner.get_output(2)

        # Edit tree
        nb_runner.set_cell_source(1, "tree = {'val': 10, 'left': {'val': 20, 'left': None, 'right': None}, 'right': None}")
        nb_runner.run_all()
        assert "vals = [10, 20]" in nb_runner.get_output(2)

    def test_edit_nested_dict_depth(self, nb_runner):
        """Edit depth of nested dictionary."""
        nb_runner.create_notebook([
            "data = {'a': {'b': {'c': 42}}}",
            "def depth(d, level=0):\n    if not isinstance(d, dict):\n        return level\n    return max(depth(v, level + 1) for v in d.values())\nresult = depth(data)\nprint(f'depth = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth = 3" in nb_runner.get_output(2)

        # Deepen
        nb_runner.set_cell_source(1, "data = {'a': {'b': {'c': {'d': {'e': 99}}}}}")
        nb_runner.run_all()
        assert "depth = 5" in nb_runner.get_output(2)

    def test_edit_flat_list_to_tree(self, nb_runner):
        """Edit flat list that gets converted to nested pairs."""
        nb_runner.create_notebook([
            "items = [1, 2, 3, 4]",
            "def nest(lst):\n    if len(lst) <= 1:\n        return lst[0] if lst else None\n    mid = len(lst) // 2\n    return (nest(lst[:mid]), nest(lst[mid:]))\nresult = nest(items)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ((1, 2), (3, 4))" in nb_runner.get_output(2)

        # Change items
        nb_runner.set_cell_source(1, "items = [10, 20]")
        nb_runner.run_all()
        assert "result = (10, 20)" in nb_runner.get_output(2)
