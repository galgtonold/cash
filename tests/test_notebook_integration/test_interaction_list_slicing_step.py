"""Batch 365: list slicing with step and negative indices."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestListSlicingStep:
    def test_step_slice(self, nb_runner):
        nb_runner.create_notebook([
            "data = list(range(20))",
            "evens = data[::2]\nodds = data[1::2]\nreversed_data = data[::-1]\nprint(f'evens={evens[:5]}')\nprint(f'odds={odds[:5]}')\nprint(f'last3={reversed_data[:3]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "odds=[1, 3, 5, 7, 9]" in out
        assert "last3=[19, 18, 17]" in out

    def test_slice_edit(self, nb_runner):
        nb_runner.create_notebook([
            "items = ['a', 'b', 'c', 'd', 'e', 'f']",
            "middle = items[1:-1]\nprint(f'middle={middle}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "middle=['b', 'c', 'd', 'e']" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "items = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "middle=[20, 30, 40]" in nb_runner.get_output(2)

    def test_negative_index(self, nb_runner):
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]",
            "last = data[-1]\nsecond_last = data[-2]\nslice_neg = data[-3:]\nprint(f'last={last} second_last={second_last} slice={slice_neg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "last=50" in out
        assert "second_last=40" in out
        assert "slice=[30, 40, 50]" in out
