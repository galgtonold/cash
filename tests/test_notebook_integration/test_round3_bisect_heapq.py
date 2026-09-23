"""Bisect, heapq & algorithm patterns — cash caching with stdlib algorithms."""

import textwrap

import pytest


@pytest.mark.stress
class TestHeapqPatterns:
    """Test heapq patterns across cells."""

    def test_priority_queue(self, nb_runner):
        """Priority queue with heapq across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import heapq

                tasks = []
                heapq.heappush(tasks, (3, 'low priority'))
                heapq.heappush(tasks, (1, 'urgent'))
                heapq.heappush(tasks, (2, 'medium'))
                heapq.heappush(tasks, (1, 'also urgent'))
                print(f"size={len(tasks)}")
            """),
                textwrap.dedent("""\
                import heapq
                order = []
                while tasks:
                    priority, name = heapq.heappop(tasks)
                    order.append(f"[{priority}]{name}")
                print(f"order={order}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=4" in nb_runner.get_output(1)
        out = nb_runner.get_output(2)
        assert "[1]" in out
        # urgents come first
        assert out.index("[1]") < out.index("[3]")
