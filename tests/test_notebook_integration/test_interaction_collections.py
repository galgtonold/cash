"""
Batch 291: Collections module interaction tests (Counter, OrderedDict, defaultdict, namedtuple).
Tests that editing data fed into collections types properly invalidates downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestCollectionsInteraction:
    """Test collections module patterns with cache invalidation."""

    def test_counter_edit(self, nb_runner):
        """Editing input to Counter should propagate."""
        nb_runner.create_notebook([
            "from collections import Counter\nwords = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']",
            "counts = Counter(words)",
            "top = counts.most_common(1)[0]",
            "print(f'top={top[0]},count={top[1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "top=apple,count=3" in out

        nb_runner.set_cell_source(1, "from collections import Counter\nwords = ['banana', 'banana', 'banana', 'cherry', 'apple']")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "top=banana,count=3" in out

    def test_defaultdict_edit(self, nb_runner):
        """Editing data that populates a defaultdict should propagate."""
        nb_runner.create_notebook([
            "from collections import defaultdict\npairs = [('a', 1), ('b', 2), ('a', 3)]",
            "dd = defaultdict(list)\nfor k, v in pairs:\n    dd[k].append(v)",
            "result_a = sorted(dd['a'])\nresult_b = sorted(dd['b'])",
            "print(f'a={result_a},b={result_b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=[1, 3]" in out
        assert "b=[2]" in out

        nb_runner.set_cell_source(1, "from collections import defaultdict\npairs = [('a', 10), ('b', 20), ('b', 30), ('a', 40)]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=[10, 40]" in out
        assert "b=[20, 30]" in out

    def test_namedtuple_edit(self, nb_runner):
        """Editing namedtuple instances should propagate."""
        nb_runner.create_notebook([
            "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
            "p1 = Point(1, 2)\np2 = Point(3, 4)",
            "dist_sq = (p2.x - p1.x)**2 + (p2.y - p1.y)**2",
            "print(f'dist_sq={dist_sq}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist_sq=8" in out

        nb_runner.set_cell_source(2, "p1 = Point(0, 0)\np2 = Point(3, 4)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist_sq=25" in out
