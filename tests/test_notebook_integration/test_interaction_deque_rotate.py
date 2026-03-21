"""
Batch 318: collections.deque rotation and operation patterns with caching.
Tests deque creation, rotate, appendleft, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDequeRotate:
    """Test collections.deque operation caching."""

    def test_deque_rotate_basic(self, nb_runner):
        """Create deque, rotate, verify caching."""
        nb_runner.create_notebook([
            "from collections import deque",
            "d = deque([1, 2, 3, 4, 5])\nd.rotate(2)",
            "result = list(d)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "result=[4, 5, 1, 2, 3]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "result=[4, 5, 1, 2, 3]" in out2

    def test_deque_maxlen_edit(self, nb_runner):
        """Deque with maxlen, edit propagation."""
        nb_runner.create_notebook([
            "from collections import deque",
            "size = 3",
            "d = deque(range(10), maxlen=size)\nresult = list(d)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[7, 8, 9]" in out

        nb_runner.set_cell_source(2, "size = 5")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=[5, 6, 7, 8, 9]" in out2

    def test_deque_appendleft_pattern(self, nb_runner):
        """Build deque via appendleft in same cell."""
        nb_runner.create_notebook([
            "from collections import deque",
            "items = [10, 20, 30]",
            "d = deque()\nfor item in items:\n    d.appendleft(item)\nresult = list(d)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[30, 20, 10]" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=[30, 20, 10]" in out2
