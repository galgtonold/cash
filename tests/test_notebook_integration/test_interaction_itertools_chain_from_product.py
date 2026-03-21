"""Batch 467: itertools chain from iterable and product repeat."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsChainFromProduct:
    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools",
            "nested = [[1, 2], [3, 4], [5]]\nflat = list(itertools.chain.from_iterable(nested))\nprint(f'flat={flat} sum={sum(flat)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "flat=[1, 2, 3, 4, 5]" in out
        assert "sum=15" in out

    def test_product_repeat(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools",
            "bits = list(itertools.product([0, 1], repeat=3))\nprint(f'combos={len(bits)} first={bits[0]} last={bits[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "combos=8" in out
        assert "(0, 0, 0)" in out
        assert "(1, 1, 1)" in out

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools",
            "data = list(itertools.chain.from_iterable([[1], [2]]))\nprint(f'data={data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[1, 2]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "data = list(itertools.chain.from_iterable([[10, 20], [30]]))\nprint(f'data={data}')")
        nb_runner.run_all()
        assert "data=[10, 20, 30]" in nb_runner.get_output(2)
