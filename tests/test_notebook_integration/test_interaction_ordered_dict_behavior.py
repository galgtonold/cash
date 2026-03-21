"""Batch 401: collections.OrderedDict behavior."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOrderedDictBehavior:
    def test_ordered_dict_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nod = OrderedDict([('b', 2), ('a', 1), ('c', 3)])",
            "keys = list(od.keys())\nvals = list(od.values())\nprint(f'keys={keys} vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['b', 'a', 'c']" in nb_runner.get_output(2)
        assert "vals=[2, 1, 3]" in nb_runner.get_output(2)

    def test_ordered_dict_move_to_end(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nod = OrderedDict([('x', 10), ('y', 20), ('z', 30)])",
            "od.move_to_end('x')\nresult = list(od.keys())\nprint(f'after_move={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "after_move=['y', 'z', 'x']" in nb_runner.get_output(2)

    def test_ordered_dict_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict\nod = OrderedDict([('a', 1), ('b', 2)])",
            "first = next(iter(od))\nlast = next(reversed(od))\nprint(f'first={first} last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=a last=b" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import OrderedDict\nod = OrderedDict([('z', 99), ('m', 50)])")
        nb_runner.run_all()
        assert "first=z last=m" in nb_runner.get_output(2)
