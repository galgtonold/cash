"""
Interaction test: collections.Counter with arithmetic and most_common.
Tests Counter addition, subtraction, intersection (&), union (|),
and most_common across cells.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterArithAdvanced:
    """Test Counter arithmetic operations across cells."""

    def test_counter_arithmetic(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create counters
            "from collections import Counter\nc1 = Counter(a=3, b=2, c=1)\nc2 = Counter(a=1, b=3, d=2)\nprint(f'c1={dict(c1)}')\nprint(f'c2={dict(c2)}')",
            # Cell 2: arithmetic operations
            "added = c1 + c2\nsubtracted = c1 - c2  # drops zero/negative\nintersected = c1 & c2  # min of each\nunioned = c1 | c2  # max of each\nprint(f'add={dict(added)}')\nprint(f'sub={dict(subtracted)}')\nprint(f'inter={dict(intersected)}')\nprint(f'union={dict(unioned)}')",
            # Cell 3: most_common and total
            "most = added.most_common(2)\ntotal_val = added.total()\nprint(f'most={most}')\nprint(f'total={total_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'a': 4" in out2
        assert "'b': 5" in out2
        out3 = nb_runner.get_output(3)
        assert "total=" in out3

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\ntext = 'hello world'\nc = Counter(text)\nprint(f'l_count={c[\"l\"]}')",
            "top3 = c.most_common(3)\nprint(f'top3={top3}')",
            "vowels = sum(c[v] for v in 'aeiou')\nprint(f'vowels={vowels}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "l_count=3" in nb_runner.get_output(1)

        # Edit text
        nb_runner.set_cell_source(1, "from collections import Counter\ntext = 'banana split'\nc = Counter(text)\nprint(f'a_count={c[\"a\"]}')")
        nb_runner.run_cells([1, 2, 3])
        assert "a_count=3" in nb_runner.get_output(1)

    def test_counter_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nwords = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the']\nwc = Counter(words)\nprint(f'the_count={wc[\"the\"]}')",
            "unique = len(wc)\nprint(f'unique={unique}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "the_count=3" in nb_runner.get_output(1)
        assert "unique=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "unique=5" in nb_runner.get_output(2)
