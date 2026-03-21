"""Batch 370: object copying (copy, deepcopy) with modifications."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCopyDeepModify:
    def test_shallow_copy(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\noriginal = {'a': [1, 2], 'b': [3, 4]}",
            "shallow = copy.copy(original)\nshallow['a'].append(99)\nprint(f'original_a={original[\"a\"]}')\nprint(f'shallow_a={shallow[\"a\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # shallow copy shares inner lists
        assert "original_a=[1, 2, 99]" in out
        assert "shallow_a=[1, 2, 99]" in out

    def test_deep_copy(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\noriginal = {'a': [1, 2], 'b': [3, 4]}",
            "deep = copy.deepcopy(original)\ndeep['a'].append(99)\nprint(f'original_a={original[\"a\"]}')\nprint(f'deep_a={deep[\"a\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "original_a=[1, 2]" in out
        assert "deep_a=[1, 2, 99]" in out

    def test_copy_edit_source(self, nb_runner):
        nb_runner.create_notebook([
            "import copy\ndata = [[1, 2], [3, 4]]",
            "cloned = copy.deepcopy(data)\ntotal = sum(sum(row) for row in cloned)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=10" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import copy\ndata = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "total=100" in nb_runner.get_output(2)
