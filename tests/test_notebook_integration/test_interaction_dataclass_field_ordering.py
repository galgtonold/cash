"""
Interaction test: dataclass field ordering with total_ordering comparisons.
Tests @dataclass(order=True) with field(compare=False), sorted() on dataclasses,
cross-cell min/max operations, and cache invalidation on value changes.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassFieldOrdering:
    """Test dataclass field-based ordering across cells."""

    def test_ordered_dataclass(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create ordered dataclass
            "from dataclasses import dataclass, field\n@dataclass(order=True)\nclass Student:\n    gpa: float\n    name: str = field(compare=False)\n\ns1 = Student(3.5, 'Alice')\ns2 = Student(3.9, 'Bob')\ns3 = Student(3.2, 'Charlie')\nstudents = [s1, s2, s3]\nsorted_names = [s.name for s in sorted(students)]\nprint(f'sorted={sorted_names}')",
            # Cell 2: comparisons
            "is_less = s1 < s2\nis_greater = s1 > s3\nprint(f'alice_lt_bob={is_less}')\nprint(f'alice_gt_charlie={is_greater}')",
            # Cell 3: min/max
            "best = max(students)\nworst = min(students)\nprint(f'best={best.name}')\nprint(f'worst={worst.name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sorted=['Charlie', 'Alice', 'Bob']" in out1
        out2 = nb_runner.get_output(2)
        assert "alice_lt_bob=True" in out2
        assert "alice_gt_charlie=True" in out2
        out3 = nb_runner.get_output(3)
        assert "best=Bob" in out3
        assert "worst=Charlie" in out3

    def test_ordered_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int\n\nscores = [Score(80), Score(95), Score(70)]\nbest = max(scores)\nprint(f'best={best.value}')",
            "spread = max(scores).value - min(scores).value\nprint(f'spread={spread}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=95" in nb_runner.get_output(1)
        assert "spread=25" in nb_runner.get_output(2)

        # Edit scores
        nb_runner.set_cell_source(1, "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int\n\nscores = [Score(50), Score(100), Score(75)]\nbest = max(scores)\nprint(f'best={best.value}')")
        nb_runner.run_cells([1, 2])
        assert "best=100" in nb_runner.get_output(1)
        assert "spread=50" in nb_runner.get_output(2)

    def test_ordered_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(order=True)\nclass Temp:\n    celsius: float\n\ntemps = [Temp(20.0), Temp(35.5), Temp(10.2)]\nhot = max(temps)\nprint(f'hot={hot.celsius}')",
            "is_hot = hot.celsius > 30\nprint(f'is_hot={is_hot}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hot=35.5" in nb_runner.get_output(1)
        assert "is_hot=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_hot=True" in nb_runner.get_output(2)
