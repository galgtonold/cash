"""
Interaction test: collections.OrderedDict move_to_end and popitem.
Tests OrderedDict ordering operations with move_to_end(last=False),
popitem(last=True/False), and cross-cell state tracking.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOrderedDictMovePopitem:
    """Test OrderedDict move_to_end and popitem across cells."""

    def test_ordereddict_operations(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create ordered dict
            "from collections import OrderedDict\nod = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])\nprint(f'keys={list(od.keys())}')",
            # Cell 2: move_to_end operations
            "od.move_to_end('a')  # move to end\nod.move_to_end('d', last=False)  # move to beginning\nordered = list(od.keys())\nprint(f'after_move={ordered}')",
            # Cell 3: popitem operations
            "last_item = od.popitem(last=True)\nfirst_item = od.popitem(last=False)\nremaining = list(od.keys())\nprint(f'popped_last={last_item}')\nprint(f'popped_first={first_item}')\nprint(f'remaining={remaining}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "keys=['a', 'b', 'c', 'd']" in out1
        out2 = nb_runner.get_output(2)
        assert "after_move=['d', 'b', 'c', 'a']" in out2
        out3 = nb_runner.get_output(3)
        assert "popped_last=('a', 1)" in out3
        assert "popped_first=('d', 4)" in out3
        assert "remaining=['b', 'c']" in out3

    def test_ordereddict_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30)])\nprint(f'keys={list(od.keys())}')",
            "od.move_to_end('x')\nresult = list(od.keys())\nprint(f'order={result}')",
            "vals = list(od.values())\nprint(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "order=['y', 'z', 'x']" in nb_runner.get_output(2)
        assert "vals=[20, 30, 10]" in nb_runner.get_output(3)

        # Edit initial dict
        nb_runner.set_cell_source(1, "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30), ('w', 40)])\nprint(f'keys={list(od.keys())}')")
        nb_runner.run_cells([1, 2, 3])
        assert "order=['y', 'z', 'w', 'x']" in nb_runner.get_output(2)
        assert "vals=[20, 30, 40, 10]" in nb_runner.get_output(3)

    def test_ordereddict_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nod = OrderedDict(alpha=1, beta=2, gamma=3)\nprint(f'count={len(od)}')",
            "reversed_keys = list(reversed(od))\nprint(f'reversed={reversed_keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "reversed=['gamma', 'beta', 'alpha']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "reversed=['gamma', 'beta', 'alpha']" in nb_runner.get_output(2)
