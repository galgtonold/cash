"""
Interaction test: bisect for sorted sequence operations.
Tests bisect_left, bisect_right, insort, and cross-cell
sorted data maintenance and grade lookup patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBisectSortedOps:
    """Test bisect operations on sorted sequences across cells."""

    def test_bisect_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: bisect_left and bisect_right
            "import bisect\nsorted_list = [10, 20, 30, 30, 40, 50]\nleft_pos = bisect.bisect_left(sorted_list, 30)\nright_pos = bisect.bisect_right(sorted_list, 30)\nprint(f'left_pos={left_pos}')\nprint(f'right_pos={right_pos}')\nprint(f'count_30={right_pos - left_pos}')",
            # Cell 2: insort
            "data = [1, 3, 5, 7, 9]\nbisect.insort(data, 4)\nbisect.insort(data, 6)\nprint(f'data={data}')",
            # Cell 3: grade lookup using bisect
            "breakpoints = [60, 70, 80, 90]\ngrades = 'FDCBA'\ndef grade_lookup(score):\n    i = bisect.bisect(breakpoints, score)\n    return grades[i]\n\nresults = [(s, grade_lookup(s)) for s in [55, 65, 75, 85, 95]]\nfor score, g in results:\n    print(f'{score}={g}')",
        ])
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
        nb_runner.create_notebook([
            "import bisect\nthresholds = [10, 20, 30]\nlabels = ['low', 'medium', 'high', 'very_high']\nval = 25\nidx = bisect.bisect(thresholds, val)\nlabel = labels[idx]\nprint(f'label={label}')",
            "report = f'Value {val} is {label}'\nprint(f'report={report}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=high" in nb_runner.get_output(1)
        assert "report=Value 25 is high" in nb_runner.get_output(2)

        # Edit threshold
        nb_runner.set_cell_source(1, "import bisect\nthresholds = [10, 20, 30]\nlabels = ['low', 'medium', 'high', 'very_high']\nval = 35\nidx = bisect.bisect(thresholds, val)\nlabel = labels[idx]\nprint(f'label={label}')")
        nb_runner.run_cells([1, 2])
        assert "label=very_high" in nb_runner.get_output(1)
        assert "report=Value 35 is very_high" in nb_runner.get_output(2)

    def test_bisect_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import bisect\nsorted_vals = list(range(0, 100, 5))\npos = bisect.bisect_left(sorted_vals, 37)\nprint(f'pos={pos}')\nprint(f'nearest={sorted_vals[pos]}')",
            "is_exact = sorted_vals[pos] == 37\nprint(f'exact_match={is_exact}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=8" in nb_runner.get_output(1)
        assert "nearest=40" in nb_runner.get_output(1)
        assert "exact_match=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "exact_match=False" in nb_runner.get_output(2)
