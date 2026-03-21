"""
Interaction test: functools.total_ordering with rich comparison.
Tests @total_ordering decorator for auto-generating comparison methods,
sorting, and cross-cell usage with min/max.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsTotalOrdering:
    """Test functools.total_ordering across cells."""

    def test_total_ordering_comparisons(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define ordered class
            "from functools import total_ordering\n@total_ordering\nclass Temperature:\n    def __init__(self, celsius):\n        self.celsius = celsius\n    def __eq__(self, other):\n        return self.celsius == other.celsius\n    def __lt__(self, other):\n        return self.celsius < other.celsius\n    def __repr__(self):\n        return f'T({self.celsius})'\nprint('Temperature defined')",
            # Cell 2: create and compare
            "t1 = Temperature(20)\nt2 = Temperature(30)\nt3 = Temperature(20)\nprint(f'lt={t1 < t2}')\nprint(f'gt={t2 > t1}')\nprint(f'eq={t1 == t3}')\nprint(f'le={t1 <= t3}')\nprint(f'ge={t2 >= t1}')\nprint(f'ne={t1 != t2}')",
            # Cell 3: sort and min/max
            "temps = [Temperature(25), Temperature(10), Temperature(35), Temperature(15)]\nsorted_temps = sorted(temps)\nprint(f'sorted={sorted_temps}')\nprint(f'min={min(temps)}')\nprint(f'max={max(temps)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "lt=True" in out2
        assert "gt=True" in out2
        assert "eq=True" in out2
        assert "le=True" in out2
        assert "ge=True" in out2
        assert "ne=True" in out2
        out3 = nb_runner.get_output(3)
        assert "sorted=[T(10), T(15), T(25), T(35)]" in out3
        assert "min=T(10)" in out3
        assert "max=T(35)" in out3

    def test_total_ordering_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import total_ordering\n@total_ordering\nclass Score:\n    def __init__(self, val):\n        self.val = val\n    def __eq__(self, other):\n        return self.val == other.val\n    def __lt__(self, other):\n        return self.val < other.val\n    def __repr__(self):\n        return f'S({self.val})'\nprint('Score defined')",
            "scores = [Score(85), Score(92), Score(78)]\nbest = max(scores)\nprint(f'best={best}')",
            "ranking = sorted(scores, reverse=True)\nprint(f'rank={ranking}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=S(92)" in nb_runner.get_output(2)
        assert "rank=[S(92), S(85), S(78)]" in nb_runner.get_output(3)

        # Add a new score
        nb_runner.set_cell_source(2, "scores = [Score(85), Score(92), Score(78), Score(99)]\nbest = max(scores)\nprint(f'best={best}')")
        nb_runner.run_cells([2, 3])
        assert "best=S(99)" in nb_runner.get_output(2)
        assert "rank=[S(99), S(92), S(85), S(78)]" in nb_runner.get_output(3)

    def test_total_ordering_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import total_ordering\n@total_ordering\nclass Version:\n    def __init__(self, major, minor):\n        self.major = major\n        self.minor = minor\n    def __eq__(self, other):\n        return (self.major, self.minor) == (other.major, other.minor)\n    def __lt__(self, other):\n        return (self.major, self.minor) < (other.major, other.minor)\n    def __repr__(self):\n        return f'v{self.major}.{self.minor}'\nprint('Version defined')",
            "versions = [Version(2, 1), Version(1, 9), Version(2, 0)]\nlatest = max(versions)\nprint(f'latest={latest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "latest=v2.1" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "latest=v2.1" in nb_runner.get_output(2)
