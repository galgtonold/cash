"""
Interaction test: collections.Counter elements and most_common.
Tests Counter arithmetic, elements() iteration, most_common filtering,
and cross-cell counter merging with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterElements:
    """Test Counter.elements and arithmetic across cells."""

    def test_counter_elements(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create counter
            "from collections import Counter\nwords = 'apple banana apple cherry banana apple'.split()\nc = Counter(words)\nmost = c.most_common(2)\nprint(f'most={most}')",
            # Cell 2: elements iteration
            "elems = sorted(c.elements())\nprint(f'total={len(elems)}')\nprint(f'first_3={elems[:3]}')",
            # Cell 3: counter arithmetic
            "c2 = Counter({'apple': 1, 'date': 2})\ncombined = c + c2\nprint(f'apple_count={combined[\"apple\"]}')\nprint(f'date_count={combined[\"date\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('apple', 3)" in out1
        out2 = nb_runner.get_output(2)
        assert "total=6" in out2
        out3 = nb_runner.get_output(3)
        assert "apple_count=4" in out3
        assert "date_count=2" in out3

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc = Counter('aabbbcccc')\nprint(f'a={c[\"a\"]}')\nprint(f'b={c[\"b\"]}')\nprint(f'c_count={c[\"c\"]}')",
            "top = c.most_common(1)[0]\nprint(f'top_char={top[0]}')\nprint(f'top_count={top[1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=2" in nb_runner.get_output(1)
        assert "top_char=c" in nb_runner.get_output(2)
        assert "top_count=4" in nb_runner.get_output(2)

        # Edit input string
        nb_runner.set_cell_source(1, "from collections import Counter\nc = Counter('aaaaaabb')\nprint(f'a={c[\"a\"]}')\nprint(f'b={c[\"b\"]}')")
        nb_runner.run_cells([1, 2])
        assert "a=6" in nb_runner.get_output(1)
        assert "top_char=a" in nb_runner.get_output(2)
        assert "top_count=6" in nb_runner.get_output(2)

    def test_counter_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc = Counter([1, 1, 2, 3, 3, 3])\nunique = len(c)\nprint(f'unique={unique}')",
            "total = sum(c.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=3" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=6" in nb_runner.get_output(2)
