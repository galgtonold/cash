"""Batch 454: collections.deque as ring buffer pattern."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDequeRingBuffer:
    def test_deque_maxlen(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nbuf = deque(maxlen=3)",
            "for i in range(5):\n    buf.append(i)\nresult = list(buf)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 3, 4]" in nb_runner.get_output(2)

    def test_deque_rotate(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nd = deque([1, 2, 3, 4, 5])",
            "d.rotate(2)\nright = list(d)\nd.rotate(-4)\nleft = list(d)\nprint(f'right={right} left={left}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "right=[4, 5, 1, 2, 3]" in out

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nd = deque([10, 20, 30])",
            "d.appendleft(5)\nd.append(35)\nresult = list(d)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[5, 10, 20, 30, 35]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import deque\nd = deque([100, 200])")
        nb_runner.run_all()
        assert "result=[5, 100, 200, 35]" in nb_runner.get_output(2)
