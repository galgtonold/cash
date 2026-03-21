"""Batch 417: Counter intersection and union operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterIntersectionUnion:
    def test_counter_intersect(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc1 = Counter(a=3, b=1, c=5)\nc2 = Counter(a=1, b=4, c=2)",
            "inter = c1 & c2\nunion = c1 | c2\nprint(f'inter_a={inter[\"a\"]} inter_b={inter[\"b\"]} union_c={union[\"c\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "inter_a=1" in out
        assert "inter_b=1" in out
        assert "union_c=5" in out

    def test_counter_elements(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc = Counter(x=2, y=3, z=1)",
            "elems = sorted(c.elements())\nprint(f'elems={elems}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elems=['x', 'x', 'y', 'y', 'y', 'z']" in nb_runner.get_output(2)

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\ntext = 'aabbc'",
            "c = Counter(text)\ntotal = c.total()\nprint(f'total={total} a={c[\"a\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=5" in nb_runner.get_output(2)
        assert "a=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import Counter\ntext = 'aaabbbcccdddd'")
        nb_runner.run_all()
        assert "total=13" in nb_runner.get_output(2)
        assert "a=3" in nb_runner.get_output(2)
