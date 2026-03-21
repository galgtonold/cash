"""Batch 439: itertools.islice and takewhile/dropwhile."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsSliceTakewhile:
    def test_islice(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import islice, count\nnatural = count(1)",
            "first10 = list(islice(natural, 10))\nprint(f'first10={first10}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first10=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" in nb_runner.get_output(2)

    def test_takewhile_dropwhile(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import takewhile, dropwhile\nnums = [1, 3, 5, 7, 2, 4, 6]",
            "taken = list(takewhile(lambda x: x < 6, nums))\ndropped = list(dropwhile(lambda x: x < 6, nums))\nprint(f'taken={taken} dropped={dropped}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "taken=[1, 3, 5]" in nb_runner.get_output(2)
        assert "dropped=[7, 2, 4, 6]" in nb_runner.get_output(2)

    def test_islice_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import islice\ndata = list(range(100))",
            "chunk = list(islice(data, 5, 10))\nprint(f'chunk={chunk}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chunk=[5, 6, 7, 8, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "chunk = list(islice(data, 90, 95))\nprint(f'chunk={chunk}')")
        nb_runner.run_all()
        assert "chunk=[90, 91, 92, 93, 94]" in nb_runner.get_output(2)
