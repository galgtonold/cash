"""Batch 406: collections.ChainMap usage patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainMapPatterns:
    def test_chainmap_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 10}\noverrides = {'color': 'blue'}",
            "cm = ChainMap(overrides, defaults)\ncolor = cm['color']\nsize = cm['size']\nprint(f'color={color} size={size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=blue" in nb_runner.get_output(2)
        assert "size=10" in nb_runner.get_output(2)

    def test_chainmap_new_child(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\nlayer = {'b': 20}",
            "cm = ChainMap(layer, base)\nchild = cm.new_child({'c': 30})\nresult = dict(child)\nprint(f'a={child[\"a\"]} b={child[\"b\"]} c={child[\"c\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a=1" in out
        assert "b=20" in out
        assert "c=30" in out

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\nd1 = {'x': 10}\nd2 = {'y': 20}",
            "cm = ChainMap(d1, d2)\nkeys = sorted(cm.keys())\nprint(f'keys={keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import ChainMap\nd1 = {'a': 1}\nd2 = {'b': 2, 'c': 3}")
        nb_runner.run_all()
        assert "keys=['a', 'b', 'c']" in nb_runner.get_output(2)
