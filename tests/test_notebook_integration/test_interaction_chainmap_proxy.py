"""Batch 355: collections.ChainMap and MappingProxyType patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainMapProxy:
    def test_chainmap_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 'M'}\nuser = {'color': 'blue'}",
            "config = ChainMap(user, defaults)\nresult = dict(config)\nprint(f'result={dict(sorted(result.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result={'color': 'blue', 'size': 'M'}" in nb_runner.get_output(2)

    def test_chainmap_edit_user(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\ndefaults = {'a': 1, 'b': 2}\noverrides = {'b': 20}",
            "cm = ChainMap(overrides, defaults)\nval_a = cm['a']\nval_b = cm['b']\nprint(f'a={val_a} b={val_b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=20" in nb_runner.get_output(2)
        # Edit overrides
        nb_runner.set_cell_source(1, "from collections import ChainMap\ndefaults = {'a': 1, 'b': 2}\noverrides = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)

    def test_mapping_proxy(self, nb_runner):
        nb_runner.create_notebook([
            "from types import MappingProxyType\ndata = {'x': 10, 'y': 20}\nproxy = MappingProxyType(data)",
            "vals = [proxy['x'], proxy['y']]\nprint(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[10, 20]" in nb_runner.get_output(2)
