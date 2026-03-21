"""
Batch 303: Recursive data structure traversal interaction tests.
Tests tree, linked list, and nested dict traversal with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestRecursiveTraversalInteraction:
    """Test recursive traversal patterns with cache invalidation."""

    def test_tree_sum_edit(self, nb_runner):
        """Editing a tree structure should propagate to traversal results."""
        nb_runner.create_notebook([
            (
                "class TNode:\n"
                "    def __init__(self, val, children=None):\n"
                "        self.val = val\n"
                "        self.children = children or []"
            ),
            "tree = TNode(1, [TNode(2, [TNode(4)]), TNode(3, [TNode(5)])])",
            (
                "def tree_sum(node):\n"
                "    return node.val + sum(tree_sum(c) for c in node.children)\n"
                "total = tree_sum(tree)"
            ),
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        nb_runner.set_cell_source(2, "tree = TNode(10, [TNode(20, [TNode(40)]), TNode(30, [TNode(50)])])")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=150" in out

    def test_linked_list_sum_edit(self, nb_runner):
        """Editing a linked list should propagate to sum computation."""
        nb_runner.create_notebook([
            (
                "class LNode:\n"
                "    def __init__(self, val, nxt=None):\n"
                "        self.val = val\n"
                "        self.nxt = nxt"
            ),
            "head = LNode(1, LNode(2, LNode(3)))",
            (
                "def lsum(node):\n"
                "    t = 0\n"
                "    while node:\n"
                "        t += node.val\n"
                "        node = node.nxt\n"
                "    return t\n"
                "s = lsum(head)"
            ),
            "print(f'sum={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sum=6" in out

        nb_runner.set_cell_source(2, "head = LNode(10, LNode(20, LNode(30, LNode(40))))")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sum=100" in out

    def test_nested_dict_flatten_edit(self, nb_runner):
        """Editing nested dict should propagate through recursive flattening."""
        nb_runner.create_notebook([
            "nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}",
            (
                "def flatten(d, prefix=''):\n"
                "    result = {}\n"
                "    for k, v in d.items():\n"
                "        key = f'{prefix}.{k}' if prefix else k\n"
                "        if isinstance(v, dict):\n"
                "            result.update(flatten(v, key))\n"
                "        else:\n"
                "            result[key] = v\n"
                "    return result\n"
                "flat = flatten(nested)"
            ),
            "keys = sorted(flat.keys())\nvals = [flat[k] for k in keys]",
            "print(f'keys={keys},vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "keys=['a', 'b.c', 'b.d.e']" in out
        assert "vals=[1, 2, 3]" in out

        nb_runner.set_cell_source(1, "nested = {'x': 10, 'y': {'z': 20}}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "keys=['x', 'y.z']" in out
        assert "vals=[10, 20]" in out
