"""Batch 442: list slicing advanced patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestListSlicingAdvanced:
    def test_slice_step(self, nb_runner):
        nb_runner.create_notebook([
            "data = list(range(20))",
            "evens = data[::2]\nrev = data[::-1]\nchunk = data[5:15:3]\nprint(f'evens={evens[:5]} rev5={rev[:5]} chunk={chunk}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "rev5=[19, 18, 17, 16, 15]" in out
        assert "chunk=[5, 8, 11, 14]" in out

    def test_slice_assignment(self, nb_runner):
        nb_runner.create_notebook([
            "items = [1, 2, 3, 4, 5]",
            "items[1:3] = [20, 30]\nprint(f'items={items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items=[1, 20, 30, 4, 5]" in nb_runner.get_output(2)

    def test_slice_edit(self, nb_runner):
        nb_runner.create_notebook([
            "seq = list(range(10))",
            "last3 = seq[-3:]\nfirst3 = seq[:3]\nprint(f'last3={last3} first3={first3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "last3=[7, 8, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "seq = list(range(5))")
        nb_runner.run_all()
        assert "last3=[2, 3, 4]" in nb_runner.get_output(2)
        assert "first3=[0, 1, 2]" in nb_runner.get_output(2)
