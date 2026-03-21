"""Batch 447: copy.deepcopy vs shallow copy behaviors."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDeepVsShallowCopy:
    def test_shallow_vs_deep(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\noriginal = [[1, 2], [3, 4]]",
            "shallow = copy.copy(original)\ndeep = copy.deepcopy(original)\noriginal[0][0] = 99\nshallow_changed = shallow[0][0]\ndeep_unchanged = deep[0][0]\nprint(f'shallow={shallow_changed} deep={deep_unchanged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "shallow=99" in nb_runner.get_output(2)
        assert "deep=1" in nb_runner.get_output(2)

    def test_deepcopy_dict(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\nd = {'a': [1, 2], 'b': {'c': 3}}",
            "d2 = copy.deepcopy(d)\nd['a'].append(99)\nd['b']['c'] = 999\nprint(f'orig_a={d[\"a\"]} copy_a={d2[\"a\"]} copy_c={d2[\"b\"][\"c\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "copy_a=[1, 2]" in out
        assert "copy_c=3" in out

    def test_copy_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\ndata = [1, [2, 3]]",
            "cloned = copy.deepcopy(data)\ndata[1].append(4)\nprint(f'orig={data} clone={cloned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "clone=[1, [2, 3]]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import copy\ndata = [10, [20, 30]]")
        nb_runner.run_all()
        assert "clone=[10, [20, 30]]" in nb_runner.get_output(2)
