"""Batch 67: Bisect, heapq & algorithm patterns — cash caching with stdlib algorithms."""
import textwrap
import pytest


@pytest.mark.stress
class TestBisectPatterns:
    """Test bisect module patterns across cells."""

    def test_sorted_insert(self, nb_runner):
        """Maintain sorted list with bisect across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import bisect

                sorted_list = []
                for val in [5, 1, 8, 3, 9, 2, 7]:
                    bisect.insort(sorted_list, val)
                print(f"sorted={sorted_list}")
            """),
            textwrap.dedent("""\
                import bisect
                pos = bisect.bisect_left(sorted_list, 5)
                print(f"pos_of_5={pos}")
                count_le_5 = bisect.bisect_right(sorted_list, 5)
                print(f"count_le_5={count_le_5}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sorted=[1, 2, 3, 5, 7, 8, 9]" in nb_runner.get_output(1)
        assert "pos_of_5=3" in nb_runner.get_output(2)
        assert "count_le_5=4" in nb_runner.get_output(2)

    def test_grade_lookup(self, nb_runner):
        """Grade lookup using bisect across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import bisect

                def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
                    idx = bisect.bisect(breakpoints, score)
                    return grades[idx]

                scores = [33, 99, 77, 70, 89, 90, 100]
                student_grades = [(s, grade(s)) for s in scores]
                print(f"grades={student_grades}")
            """),
            textwrap.dedent("""\
                from collections import Counter
                distribution = Counter(g for _, g in student_grades)
                print(f"dist={dict(sorted(distribution.items()))}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "(33, 'F')" in out1
        assert "(99, 'A')" in out1
        out2 = nb_runner.get_output(2)
        assert "A" in out2


@pytest.mark.stress
class TestHeapqPatterns:
    """Test heapq patterns across cells."""

    def test_priority_queue(self, nb_runner):
        """Priority queue with heapq across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=4" in nb_runner.get_output(1)
        out = nb_runner.get_output(2)
        assert "[1]" in out
        # urgents come first
        assert out.index("[1]") < out.index("[3]")

    def test_nlargest_nsmallest(self, nb_runner):
        """heapq.nlargest and nsmallest across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import heapq

                data = [
                    {'name': 'AAPL', 'price': 150},
                    {'name': 'GOOGL', 'price': 2800},
                    {'name': 'MSFT', 'price': 300},
                    {'name': 'AMZN', 'price': 3400},
                    {'name': 'META', 'price': 330},
                ]
                top2 = heapq.nlargest(2, data, key=lambda x: x['price'])
                bottom2 = heapq.nsmallest(2, data, key=lambda x: x['price'])
            """),
            textwrap.dedent("""\
                top_names = [d['name'] for d in top2]
                bottom_names = [d['name'] for d in bottom2]
                print(f"top={top_names} bottom={bottom_names}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "AMZN" in out
        assert "GOOGL" in out
        assert "AAPL" in out

    def test_merge_sorted_streams(self, nb_runner):
        """heapq.merge sorted streams across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import heapq

                stream_a = [1, 5, 9, 13]
                stream_b = [2, 6, 10, 14]
                stream_c = [3, 7, 11, 15]
                merged = list(heapq.merge(stream_a, stream_b, stream_c))
                print(f"merged={merged}")
            """),
            textwrap.dedent("""\
                is_sorted = all(merged[i] <= merged[i+1] for i in range(len(merged)-1))
                print(f"sorted={is_sorted} len={len(merged)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "merged=" in out1
        assert "1," in out1
        assert "sorted=True" in nb_runner.get_output(2)
        assert "len=12" in nb_runner.get_output(2)
