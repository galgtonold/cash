"""Batch 390: collections.Counter most_common and arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCounterArithmetic:
    def test_counter_ops(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc1 = Counter('aabbc')\nc2 = Counter('bccdd')",
            "added = c1 + c2\nsubtracted = c1 - c2\ncommon = c1 & c2\nprint(f'added={dict(sorted(added.items()))}')\nprint(f'common={dict(sorted(common.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 3" in out

    def test_counter_most_common_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nwords = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'cat']",
            "counts = Counter(words)\ntop2 = counts.most_common(2)\nprint(f'top2={top2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('the', 3)" in nb_runner.get_output(2)
        assert "('cat', 2)" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from collections import Counter\nwords = ['a', 'b', 'a', 'c', 'b', 'a']")
        nb_runner.run_all()
        assert "('a', 3)" in nb_runner.get_output(2)

    def test_counter_elements(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import Counter\nc = Counter(a=3, b=1)",
            "elements = sorted(c.elements())\nprint(f'elements={elements}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elements=['a', 'a', 'a', 'b']" in nb_runner.get_output(2)
