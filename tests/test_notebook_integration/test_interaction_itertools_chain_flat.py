"""Batch 427: itertools.chain and chain.from_iterable."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsChainFlat:
    def test_chain_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\na = [1, 2]\nb = [3, 4]\nc = [5, 6]",
            "result = list(chain(a, b, c))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)

    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\nnested = [[1, 2], [3, 4], [5]]",
            "flat = list(chain.from_iterable(nested))\nprint(f'flat={flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\nx = ['a', 'b']\ny = ['c']",
            "combined = list(chain(x, y))\nprint(f'combined={combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=['a', 'b', 'c']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import chain\nx = ['x', 'y', 'z']\ny = ['w']")
        nb_runner.run_all()
        assert "combined=['x', 'y', 'z', 'w']" in nb_runner.get_output(2)
