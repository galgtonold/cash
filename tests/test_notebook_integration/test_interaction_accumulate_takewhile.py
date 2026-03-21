"""Batch 497: itertools accumulate and takewhile."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsAccumulateTakewhile:
    def test_accumulate_running_sum(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools",
            "data = [1, 2, 3, 4, 5]\nrunning = list(itertools.accumulate(data))\nprint(f'running={running}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running=[1, 3, 6, 10, 15]" in nb_runner.get_output(2)

    def test_takewhile_dropwhile(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools",
            "data = [1, 3, 5, 2, 4, 6]\ntaken = list(itertools.takewhile(lambda x: x < 4, data))\ndropped = list(itertools.dropwhile(lambda x: x < 4, data))\nprint(f'taken={taken} dropped={dropped}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "taken=[1, 3]" in out
        assert "dropped=[5, 2, 4, 6]" in out

    def test_accumulate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools\nimport operator",
            "vals = [2, 3, 4]\nprods = list(itertools.accumulate(vals, operator.mul))\nprint(f'prods={prods}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "prods=[2, 6, 24]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "vals = [1, 2, 3, 4]\nprods = list(itertools.accumulate(vals, operator.mul))\nprint(f'prods={prods}')")
        nb_runner.run_all()
        assert "prods=[1, 2, 6, 24]" in nb_runner.get_output(2)
