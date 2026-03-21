"""Batch 491: copy deepcopy nested mutable objects."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCopyDeepcopyNested:
    def test_shallow_vs_deep(self, nb_runner):
        nb_runner.create_notebook([
            "import copy",
            "original = [[1, 2], [3, 4]]\nshallow = copy.copy(original)\ndeep = copy.deepcopy(original)\noriginal[0].append(99)\nprint(f'orig={original}')\nprint(f'shallow={shallow}')\nprint(f'deep={deep}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "orig=[[1, 2, 99], [3, 4]]" in out
        assert "shallow=[[1, 2, 99], [3, 4]]" in out
        assert "deep=[[1, 2], [3, 4]]" in out

    def test_deepcopy_dict(self, nb_runner):
        nb_runner.create_notebook([
            "import copy",
            "d = {'a': [1, 2], 'b': {'c': 3}}\nd2 = copy.deepcopy(d)\nd['a'].append(99)\nd['b']['c'] = 999\nprint(f'd={d}')\nprint(f'd2={d2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d={'a': [1, 2, 99], 'b': {'c': 999}}" in out
        assert "d2={'a': [1, 2], 'b': {'c': 3}}" in out

    def test_copy_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import copy",
            "data = [1, 2, 3]\nclone = copy.deepcopy(data)\nprint(f'eq={data == clone} same={data is clone}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "eq=True" in out
        assert "same=False" in out
        nb_runner.set_cell_source(2, "data = {'x': [1]}\nclone = copy.deepcopy(data)\ndata['x'].append(2)\nprint(f'data={data} clone={clone}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "data={'x': [1, 2]}" in out2
        assert "clone={'x': [1]}" in out2
