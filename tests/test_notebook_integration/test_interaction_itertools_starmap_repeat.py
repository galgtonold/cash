"""Batch 455: itertools.starmap and repeat patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsStarmapRepeat:
    def test_starmap(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap\npairs = [(2, 3), (4, 5), (6, 7)]",
            "products = list(starmap(lambda a, b: a * b, pairs))\nprint(f'products={products}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "products=[6, 20, 42]" in nb_runner.get_output(2)

    def test_repeat(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import repeat\nvals = list(repeat('x', 5))",
            "print(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=['x', 'x', 'x', 'x', 'x']" in nb_runner.get_output(2)

    def test_starmap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap\ndata = [(1, 10), (2, 20), (3, 30)]",
            "sums = list(starmap(lambda a, b: a + b, data))\nprint(f'sums={sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums=[11, 22, 33]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import starmap\ndata = [(5, 5), (10, 10)]")
        nb_runner.run_all()
        assert "sums=[10, 20]" in nb_runner.get_output(2)
