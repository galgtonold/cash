"""Batch 377: collections.deque as stack/queue with maxlen."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDequeStackQueue:
    def test_deque_as_queue(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nq = deque()\nfor i in range(5):\n    q.append(i)",
            "items = []\nwhile q:\n    items.append(q.popleft())\nprint(f'items={items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items=[0, 1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_deque_maxlen_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nbuf = deque(maxlen=3)\nfor i in [10, 20, 30, 40, 50]:\n    buf.append(i)",
            "result = list(buf)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 40, 50]" in nb_runner.get_output(2)
        # Edit maxlen
        nb_runner.set_cell_source(1, "from collections import deque\nbuf = deque(maxlen=2)\nfor i in [10, 20, 30, 40, 50]:\n    buf.append(i)")
        nb_runner.run_all()
        assert "result=[40, 50]" in nb_runner.get_output(2)

    def test_deque_extend(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import deque\nd = deque([1, 2, 3])\nd.extendleft([10, 20])",
            "result = list(d)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[20, 10, 1, 2, 3]" in nb_runner.get_output(2)
