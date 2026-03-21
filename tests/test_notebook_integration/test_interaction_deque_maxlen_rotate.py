"""Batch 495: collections deque maxlen and rotate."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDequeMaxlenRotate:
    def test_deque_maxlen(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque",
            "d = deque(maxlen=3)\nfor i in range(5):\n    d.append(i)\nprint(f'd={list(d)} len={len(d)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=[2, 3, 4]" in out
        assert "len=3" in out

    def test_deque_rotate(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque",
            "d = deque([1, 2, 3, 4, 5])\nd.rotate(2)\nr1 = list(d)\nd.rotate(-3)\nr2 = list(d)\nprint(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=[4, 5, 1, 2, 3]" in out
        assert "r2=[2, 3, 4, 5, 1]" in out

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque",
            "d = deque([10, 20, 30])\nd.appendleft(5)\nprint(f'd={list(d)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=[5, 10, 20, 30]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = deque([10, 20, 30])\nd.extendleft([3, 2, 1])\nprint(f'd={list(d)}')")
        nb_runner.run_all()
        assert "d=[1, 2, 3, 10, 20, 30]" in nb_runner.get_output(2)
