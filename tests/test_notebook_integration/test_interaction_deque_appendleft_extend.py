"""
Interaction test: deque appendleft and extendleft operations.
Tests collections.deque with appendleft, extendleft, popleft,
and cross-cell deque state management.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDequeAppendleftExtend:
    """Test deque appendleft and extendleft across cells."""

    def test_deque_left_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: build deque
            "from collections import deque\nd = deque([3, 4, 5])\nd.appendleft(2)\nd.appendleft(1)\nprint(f'deque={list(d)}')",
            # Cell 2: extendleft (note: order is reversed)
            "d2 = deque([4, 5, 6])\nd2.extendleft([3, 2, 1])  # 1 goes first, then 2, then 3\nprint(f'd2={list(d2)}')",
            # Cell 3: combine
            "merged = list(d) + list(d2)\nprint(f'merged_len={len(merged)}')\nprint(f'sorted={sorted(set(merged))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "deque=[1, 2, 3, 4, 5]" in out1
        out2 = nb_runner.get_output(2)
        assert "d2=[1, 2, 3, 4, 5, 6]" in out2
        out3 = nb_runner.get_output(3)
        assert "merged_len=11" in out3

    def test_deque_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nd = deque([1, 2, 3], maxlen=5)\nd.appendleft(0)\nprint(f'd={list(d)}')",
            "total = sum(d)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=[0, 1, 2, 3]" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Edit cell 2 to also change
        nb_runner.set_cell_source(1, "from collections import deque\nd = deque([1, 2, 3], maxlen=5)\nd.appendleft(0)\nd.appendleft(-1)\nd.appendleft(-2)  # triggers drop from right\nprint(f'd={list(d)}')")
        nb_runner.set_cell_source(2, "total = sum(d)  # recomputed\nprint(f'total={total}')")
        nb_runner.run_cells([1, 2])
        assert "d=[-2, -1, 0, 1, 2]" in nb_runner.get_output(1)
        assert "total=0" in nb_runner.get_output(2)

    def test_deque_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nd = deque(range(5))\nleft = d.popleft()\nright = d.pop()\nprint(f'left={left} right={right}')\nprint(f'remaining={list(d)}')",
            "mid = d[len(d)//2]\nprint(f'mid={mid}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "left=0 right=4" in nb_runner.get_output(1)
        assert "mid=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "mid=2" in nb_runner.get_output(2)
