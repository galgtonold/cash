"""Batch 501: list slicing and slice object usage."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestListSlicingObject:
    def test_slice_patterns(self, nb_runner):
        nb_runner.create_notebook([
            "data = list(range(10))",
            "first3 = data[:3]\nlast3 = data[-3:]\nevens = data[::2]\nreversed_list = data[::-1]\nprint(f'first3={first3} last3={last3}')\nprint(f'evens={evens}')\nprint(f'rev={reversed_list}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first3=[0, 1, 2]" in out
        assert "last3=[7, 8, 9]" in out
        assert "evens=[0, 2, 4, 6, 8]" in out
        assert "rev=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]" in out

    def test_slice_object(self, nb_runner):
        nb_runner.create_notebook([
            "data = list(range(20))",
            "s = slice(2, 10, 3)\nresult = data[s]\nprint(f'result={result} start={s.start} stop={s.stop} step={s.step}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result=[2, 5, 8]" in out
        assert "start=2" in out
        assert "stop=10" in out

