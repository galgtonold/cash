"""
Batch 329: OrderedDict patterns with caching.
Tests OrderedDict operations, move_to_end, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestOrderedDictOps:
    """Test OrderedDict operation caching."""

    def test_ordered_dict_basic(self, nb_runner):
        """OrderedDict preserves insertion order with caching."""
        nb_runner.create_notebook([
            "from collections import OrderedDict",
            "od = OrderedDict([('b', 2), ('a', 1), ('c', 3)])",
            "keys = list(od.keys())\nprint(f'keys={keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "keys=['b', 'a', 'c']" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "keys=['b', 'a', 'c']" in out2

    def test_ordered_dict_edit(self, nb_runner):
        """Edit OrderedDict, verify order update."""
        nb_runner.create_notebook([
            "from collections import OrderedDict",
            "items = [('x', 10), ('y', 20), ('z', 30)]",
            "od = OrderedDict(items)\nfirst = list(od.keys())[0]",
            "print(f'first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "first=x" in out

        nb_runner.set_cell_source(2, "items = [('z', 30), ('x', 10), ('y', 20)]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "first=z" in out2
