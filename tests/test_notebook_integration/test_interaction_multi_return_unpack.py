"""Batch 430: multiple return values and tuple unpacking."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiReturnTupleUnpack:
    def test_multi_return(self, nb_runner):
        nb_runner.create_notebook([
            "def stats(nums):\n    return min(nums), max(nums), sum(nums) / len(nums)",
            "lo, hi, avg = stats([10, 20, 30, 40, 50])\nprint(f'lo={lo} hi={hi} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10" in nb_runner.get_output(2)
        assert "hi=50" in nb_runner.get_output(2)
        assert "avg=30.0" in nb_runner.get_output(2)

    def test_star_unpack(self, nb_runner):
        nb_runner.create_notebook([
            "items = [1, 2, 3, 4, 5]",
            "first, *middle, last = items\nprint(f'first={first} middle={middle} last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1" in nb_runner.get_output(2)
        assert "middle=[2, 3, 4]" in nb_runner.get_output(2)
        assert "last=5" in nb_runner.get_output(2)

    def test_multi_return_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def divmod_custom(a, b):\n    return a // b, a % b",
            "q, r = divmod_custom(17, 5)\nprint(f'q={q} r={r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q=3 r=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "q, r = divmod_custom(100, 7)\nprint(f'q={q} r={r}')")
        nb_runner.run_all()
        assert "q=14 r=2" in nb_runner.get_output(2)
