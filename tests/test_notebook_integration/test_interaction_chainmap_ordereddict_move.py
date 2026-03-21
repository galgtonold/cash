"""Batch 478: collections ChainMap and OrderedDict move."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainMapOrderedDict:
    def test_chainmap_lookup(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap",
            "cmd_args = {'debug': True}\nenv_vars = {'debug': False, 'verbose': True}\ndefaults = {'debug': False, 'verbose': False, 'log': 'info'}\nconfig = ChainMap(cmd_args, env_vars, defaults)\nprint(f'debug={config[\"debug\"]} verbose={config[\"verbose\"]} log={config[\"log\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "debug=True" in out
        assert "verbose=True" in out
        assert "log=info" in out

    def test_ordereddict_move(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import OrderedDict",
            "od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])\nod.move_to_end('a')\norder1 = list(od.keys())\nod.move_to_end('c', last=False)\norder2 = list(od.keys())\nprint(f'order1={order1} order2={order2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "order1=['b', 'c', 'a']" in out
        assert "order2=['c', 'b', 'a']" in out

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap",
            "a = {'x': 1}\nb = {'y': 2}\ncm = ChainMap(a, b)\nprint(f'x={cm[\"x\"]} y={cm[\"y\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "a = {'x': 99}\nb = {'y': 2}\ncm = ChainMap(a, b)\nprint(f'x={cm[\"x\"]} y={cm[\"y\"]}')")
        nb_runner.run_all()
        assert "x=99" in nb_runner.get_output(2)
