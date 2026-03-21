"""Batch 382: recursive data structures (trees, nested dicts)."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveDataStruct:
    def test_tree_traversal(self, nb_runner):
        nb_runner.create_notebook([
            "tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}",
            "def inorder(node):\n    if node is None:\n        return []\n    return inorder(node['left']) + [node['val']] + inorder(node['right'])\nresult = inorder(tree)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 1, 3]" in nb_runner.get_output(2)

    def test_nested_dict_flatten(self, nb_runner):
        nb_runner.create_notebook([
            "nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}",
            "def flatten_dict(d, prefix=''):\n    items = {}\n    for k, v in d.items():\n        new_key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            items.update(flatten_dict(v, new_key))\n        else:\n            items[new_key] = v\n    return items\nresult = flatten_dict(nested)\nprint(f'result={dict(sorted(result.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b.c': 2" in nb_runner.get_output(2)
        assert "'b.d.e': 3" in nb_runner.get_output(2)

    def test_recursive_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def depth(obj):\n    if not isinstance(obj, (list, dict)):\n        return 0\n    if isinstance(obj, list):\n        return 1 + max((depth(x) for x in obj), default=0)\n    return 1 + max((depth(v) for v in obj.values()), default=0)",
            "data = {'a': [1, [2, [3]]]}",
            "d = depth(data)\nprint(f'd={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=4" in nb_runner.get_output(3)
        # Edit data
        nb_runner.set_cell_source(2, "data = {'a': 1}")
        nb_runner.run_all()
        assert "d=1" in nb_runner.get_output(3)
