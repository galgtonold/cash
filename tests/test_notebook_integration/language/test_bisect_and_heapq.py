"""bisect and heapq across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBisectInsortSortedKey:
    """bisect insort and sorted key functions."""

    def test_bisect_search(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect",
                "grades = [60, 70, 80, 90]\ncutoffs = [60, 70, 80, 90]\nletters = ['F', 'D', 'C', 'B', 'A']\nresults = []\nfor g in [55, 65, 75, 85, 95]:\n    idx = bisect.bisect(cutoffs, g)\n    results.append(letters[idx])\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)

    def test_sorted_key(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [('banana', 3), ('apple', 1), ('cherry', 2)]",
                "by_name = sorted(items, key=lambda x: x[0])\nby_count = sorted(items, key=lambda x: x[1])\nprint(f'by_name={[n for n,c in by_name]}')\nprint(f'by_count={[n for n,c in by_count]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_name=['apple', 'banana', 'cherry']" in out
        assert "by_count=['apple', 'cherry', 'banana']" in out

    def test_bisect_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect",
                "data = [1, 3, 5, 7]\nbisect.insort(data, 4)\nprint(f'data={data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[1, 3, 4, 5, 7]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "data = [10, 20, 30]\nbisect.insort(data, 25)\nprint(f'data={data}')")
        nb_runner.run_all()
        assert "data=[10, 20, 25, 30]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBisectBinarySearch:
    """bisect insert and binary search patterns."""

    def test_bisect_insort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\ndata = [10, 20, 30, 40, 50]",
                "pos = bisect.bisect_left(data, 25)\nbisect.insort(data, 25)\nprint(f'pos={pos} data={data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=2" in nb_runner.get_output(2)
        assert "data=[10, 20, 25, 30, 40, 50]" in nb_runner.get_output(2)

    def test_bisect_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\nsorted_list = [1, 3, 5, 7, 9]",
                "idx = bisect.bisect_right(sorted_list, 5)\nprint(f'idx={idx}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=3" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import bisect\nsorted_list = [2, 4, 6, 8, 10]")
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)

    def test_grade_lookup(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\nbreakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'",
                "scores = [55, 65, 75, 85, 95]\nresult = [grades[bisect.bisect(breakpoints, s)] for s in scores]\nprint(f'grades={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grades=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)


# Interaction test: bisect for sorted sequence operations.
# Tests bisect_left, bisect_right, insort, and cross-cell
# sorted data maintenance and grade lookup patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBisectSortedOps:
    """Test bisect operations on sorted sequences across cells."""

    def test_bisect_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: bisect_left and bisect_right
                "import bisect\nsorted_list = [10, 20, 30, 30, 40, 50]\nleft_pos = bisect.bisect_left(sorted_list, 30)\nright_pos = bisect.bisect_right(sorted_list, 30)\nprint(f'left_pos={left_pos}')\nprint(f'right_pos={right_pos}')\nprint(f'count_30={right_pos - left_pos}')",
                # Cell 2: insort
                "data = [1, 3, 5, 7, 9]\nbisect.insort(data, 4)\nbisect.insort(data, 6)\nprint(f'data={data}')",
                # Cell 3: grade lookup using bisect
                "breakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'\ndef grade_lookup(score):\n    i = bisect.bisect(breakpoints, score)\n    return grades[i]\n\nresults = [(s, grade_lookup(s)) for s in [55, 65, 75, 85, 95]]\nfor score, g in results:\n    print(f'{score}={g}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "left_pos=2" in out1
        assert "right_pos=4" in out1
        assert "count_30=2" in out1
        out2 = nb_runner.get_output(2)
        assert "data=[1, 3, 4, 5, 6, 7, 9]" in out2
        out3 = nb_runner.get_output(3)
        assert "55=F" in out3
        assert "75=C" in out3
        assert "95=A" in out3

    def test_bisect_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\nthresholds = [10, 20, 30]\nlabels = ['low', 'medium', 'high', 'very_high']\nval = 25\nidx = bisect.bisect(thresholds, val)\nlabel = labels[idx]\nprint(f'label={label}')",
                "report = f'Value {val} is {label}'\nprint(f'report={report}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=high" in nb_runner.get_output(1)
        assert "report=Value 25 is high" in nb_runner.get_output(2)

        # Edit threshold
        nb_runner.set_cell_source(
            1,
            "import bisect\nthresholds = [10, 20, 30]\nlabels = ['low', 'medium', 'high', 'very_high']\nval = 35\nidx = bisect.bisect(thresholds, val)\nlabel = labels[idx]\nprint(f'label={label}')",
        )
        nb_runner.run_cells([1, 2])
        assert "label=very_high" in nb_runner.get_output(1)
        assert "report=Value 35 is very_high" in nb_runner.get_output(2)

    def test_bisect_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\nsorted_vals = list(range(0, 100, 5))\npos = bisect.bisect_left(sorted_vals, 37)\nprint(f'pos={pos}')\nprint(f'nearest={sorted_vals[pos]}')",
                "is_exact = sorted_vals[pos] == 37\nprint(f'exact_match={is_exact}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=8" in nb_runner.get_output(1)
        assert "nearest=40" in nb_runner.get_output(1)
        assert "exact_match=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "exact_match=False" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBisectSortedSeq:
    """bisect module for sorted sequence operations."""

    def test_bisect_grade(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\nbreakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'\nscores = [33, 65, 77, 89, 95]",
                "results = [grades[bisect.bisect(breakpoints, s)] for s in scores]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=['F', 'D', 'C', 'B', 'A']" in nb_runner.get_output(2)

    def test_bisect_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import bisect\ndata = [1, 3, 5, 7, 9]",
                "idx = bisect.bisect_left(data, 5)\nprint(f'idx={idx}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import bisect\ndata = [2, 4, 6, 8, 10]")
        nb_runner.set_cell_source(2, "idx = bisect.bisect_left(data, 6)\nprint(f'idx={idx}')")
        nb_runner.run_all()
        assert "idx=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHeapqPriority:
    """heapq priority queue operations and edits."""

    def test_heapq_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\ndata = [5, 1, 8, 3, 2]\nheapq.heapify(data)\nsmallest = heapq.nsmallest(3, data)\nprint(f'smallest={smallest}')",
                "heapq.heappush(data, 0)\ntop = heapq.heappop(data)\nprint(f'top={top}')",
                "merged = list(heapq.merge([1,4,7], [2,5,8], [3,6,9]))\nprint(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=[1, 2, 3]" in nb_runner.get_output(1)
        assert "top=0" in nb_runner.get_output(2)
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(3)

    def test_heapq_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\ndata = [5, 1, 8, 3, 2]",
                "heapq.heapify(data)\ntop3 = heapq.nsmallest(3, data)\nprint(f'top3={top3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[1, 2, 3]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import heapq\ndata = [50, 10, 80, 30, 20]")
        nb_runner.run_all()
        assert "top3=[10, 20, 30]" in nb_runner.get_output(2)

    def test_heapq_nlargest(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\nscores = [88, 92, 75, 100, 63, 95]",
                "top2 = heapq.nlargest(2, scores)\nprint(f'top2={top2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top2=[100, 95]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHeapqPriorityQueue:
    """heapq operations for priority queue patterns."""

    def test_heapq_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\ndata = [5, 1, 8, 3, 2]",
                "heapq.heapify(data)\nsmallest = heapq.heappop(data)\nnext_smallest = heapq.heappop(data)\nprint(f'smallest={smallest} next={next_smallest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=1" in nb_runner.get_output(2)
        assert "next=2" in nb_runner.get_output(2)

    def test_nlargest_nsmallest(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\nnums = [10, 1, 8, 3, 5, 7, 2, 9, 4, 6]",
                "top3 = heapq.nlargest(3, nums)\nbot3 = heapq.nsmallest(3, nums)\nprint(f'top3={top3} bot3={bot3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[10, 9, 8]" in nb_runner.get_output(2)
        assert "bot3=[1, 2, 3]" in nb_runner.get_output(2)

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\nvals = [7, 3, 9, 1]",
                "top2 = heapq.nlargest(2, vals)\nprint(f'top2={top2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top2=[9, 7]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import heapq\nvals = [100, 50, 200, 75]")
        nb_runner.run_all()
        assert "top2=[200, 100]" in nb_runner.get_output(2)


# Interaction test: heapq nlargest nsmallest with key function.
# Tests heapq.nlargest, heapq.nsmallest with key parameter,
# heapify, heappush/heappop, and cross-cell priority queue patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHeapqNlargestKey:
    """Test heapq nlargest/nsmallest with key across cells."""

    def test_heapq_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: nlargest/nsmallest
                "import heapq\nscores = [('Alice', 92), ('Bob', 85), ('Charlie', 98), ('Diana', 88), ('Eve', 95)]\ntop2 = heapq.nlargest(2, scores, key=lambda x: x[1])\nbottom2 = heapq.nsmallest(2, scores, key=lambda x: x[1])\nprint(f'top2={top2}')\nprint(f'bottom2={bottom2}')",
                # Cell 2: heap as priority queue
                "heap = []\nfor name, score in scores:\n    heapq.heappush(heap, (-score, name))  # negative for max-heap\nbest_name = heapq.heappop(heap)[1]\nsecond_name = heapq.heappop(heap)[1]\nprint(f'best={best_name}')\nprint(f'second={second_name}')",
                # Cell 3: merge sorted sequences
                "seq1 = [1, 3, 5, 7]\nseq2 = [2, 4, 6, 8]\nmerged = list(heapq.merge(seq1, seq2))\nprint(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Charlie" in out1 and "Eve" in out1
        assert "Bob" in out1
        out2 = nb_runner.get_output(2)
        assert "best=Charlie" in out2
        assert "second=Eve" in out2
        out3 = nb_runner.get_output(3)
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8]" in out3

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\nnums = [5, 1, 8, 3, 9, 2]\ntop3 = heapq.nlargest(3, nums)\nprint(f'top3={top3}')",
                "top_sum = sum(top3)\nprint(f'top_sum={top_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3=[9, 8, 5]" in nb_runner.get_output(1)
        assert "top_sum=22" in nb_runner.get_output(2)

        # Edit to get top 2 instead
        nb_runner.set_cell_source(
            1, "import heapq\nnums = [5, 1, 8, 3, 9, 2]\ntop3 = heapq.nlargest(2, nums)\nprint(f'top3={top3}')"
        )
        nb_runner.run_cells([1, 2])
        assert "top3=[9, 8]" in nb_runner.get_output(1)
        assert "top_sum=17" in nb_runner.get_output(2)

    def test_heapq_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq\ndata = [15, 3, 22, 7, 19, 11]\nsmallest = heapq.nsmallest(3, data)\nprint(f'smallest={smallest}')",
                "small_sum = sum(smallest)\nprint(f'small_sum={small_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "smallest=[3, 7, 11]" in nb_runner.get_output(1)
        assert "small_sum=21" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "small_sum=21" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHeapqNlargestMerge:
    """heapq nlargest nsmallest merge."""

    def test_nlargest_nsmallest(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq",
                "data = [15, 3, 8, 22, 1, 17, 9]\ntop3 = heapq.nlargest(3, data)\nbot3 = heapq.nsmallest(3, data)\nprint(f'top3={top3} bot3={bot3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "top3=[22, 17, 15]" in out
        assert "bot3=[1, 3, 8]" in out

    def test_merge_sorted(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq",
                "a = [1, 4, 7]\nb = [2, 5, 8]\nc = [3, 6, 9]\nmerged = list(heapq.merge(a, b, c))\nprint(f'merged={merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(2)

    def test_heapq_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import heapq",
                "vals = [10, 20, 30]\nresult = heapq.nlargest(2, vals)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 20]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "vals = [100, 200, 300, 400]\nresult = heapq.nlargest(2, vals)\nprint(f'result={result}')"
        )
        nb_runner.run_all()
        assert "result=[400, 300]" in nb_runner.get_output(2)


# Bisect, heapq & algorithm patterns — cash caching with stdlib algorithms.
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


# Bisect/heapq interaction tests.
# Tests that editing sorted data or heap structures properly invalidates
# downstream lookups and extractions.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBisectHeapqInteraction:
    """Test bisect/heapq patterns with cache invalidation."""

    def test_bisect_insert_point_edit(self, nb_runner):
        """Editing sorted data should invalidate bisect lookup."""
        nb_runner.create_notebook(
            [
                "import bisect\nsorted_data = [10, 20, 30, 40, 50]",
                "target = 25",
                "pos = bisect.bisect_left(sorted_data, target)",
                "print(f'pos={pos}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "pos=2" in out

        nb_runner.set_cell_source(2, "target = 45")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "pos=4" in out

    def test_heapq_nsmallest_edit(self, nb_runner):
        """Editing data should invalidate heapq nsmallest results."""
        nb_runner.create_notebook(
            [
                "import heapq\nvalues = [5, 1, 8, 3, 9, 2]",
                "n = 3",
                "smallest = heapq.nsmallest(n, values)",
                "result = ','.join(str(x) for x in smallest)",
                "print(f'smallest={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "smallest=1,2,3" in out

        nb_runner.set_cell_source(1, "import heapq\nvalues = [50, 10, 80, 30, 90, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "smallest=10,20,30" in out

    def test_heapq_merge_edit(self, nb_runner):
        """Editing one of the sorted lists to merge should propagate."""
        nb_runner.create_notebook(
            [
                "import heapq\nlist_a = [1, 4, 7]\nlist_b = [2, 5, 8]",
                "merged = list(heapq.merge(list_a, list_b))",
                "result = ','.join(str(x) for x in merged)",
                "print(f'merged={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "merged=1,2,4,5,7,8" in out

        nb_runner.set_cell_source(1, "import heapq\nlist_a = [10, 40, 70]\nlist_b = [2, 5, 8]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "merged=2,5,8,10,40,70" in out
