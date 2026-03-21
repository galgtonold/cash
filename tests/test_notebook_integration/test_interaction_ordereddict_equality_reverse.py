"""
Interaction test: OrderedDict equality and reversal.
Tests OrderedDict order-sensitive equality, reversed() iteration,
and cross-cell dict rebuilding.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOrderedDictEqualityReverse:
    """Test OrderedDict equality and reversal across cells."""

    def test_ordereddict_equality(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: order-sensitive equality
            "from collections import OrderedDict\nod1 = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nod2 = OrderedDict([('c', 3), ('b', 2), ('a', 1)])\nod3 = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nprint(f'eq_diff_order={od1 == od2}')\nprint(f'eq_same_order={od1 == od3}')",
            # Cell 2: reversed iteration
            "rev_keys = list(reversed(od1))\nrev_items = list(reversed(od1.items()))\nprint(f'rev_keys={rev_keys}')\nprint(f'rev_last_item={rev_items[0]}')",
            # Cell 3: rebuild from reversed
            "rebuilt = OrderedDict(reversed(list(od1.items())))\nprint(f'rebuilt_keys={list(rebuilt.keys())}')\nprint(f'rebuilt_eq_od2={rebuilt == od2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "eq_diff_order=False" in out1
        assert "eq_same_order=True" in out1
        out2 = nb_runner.get_output(2)
        assert "rev_keys=['c', 'b', 'a']" in out2
        assert "rev_last_item=('c', 3)" in out2
        out3 = nb_runner.get_output(3)
        assert "rebuilt_keys=['c', 'b', 'a']" in out3
        assert "rebuilt_eq_od2=True" in out3

    def test_ordereddict_eq_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\ndata = OrderedDict(alpha=1, beta=2, gamma=3)\nprint(f'keys={list(data.keys())}')",
            "summary = '-'.join(data.keys())\nprint(f'summary={summary}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=alpha-beta-gamma" in nb_runner.get_output(2)

        # Edit
        nb_runner.set_cell_source(1, "from collections import OrderedDict\ndata = OrderedDict(alpha=1, beta=2, gamma=3, delta=4)\nprint(f'keys={list(data.keys())}')")
        nb_runner.run_cells([1, 2])
        assert "summary=alpha-beta-gamma-delta" in nb_runner.get_output(2)

    def test_ordereddict_eq_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nscores = OrderedDict(math=95, science=88, english=92)\nprint(f'count={len(scores)}')",
            "avg = sum(scores.values()) / len(scores)\nprint(f'avg={avg:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=91.7" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "avg=91.7" in nb_runner.get_output(2)
