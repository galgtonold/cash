"""
Batch 328: defaultdict patterns with caching.
Tests defaultdict(list), defaultdict(int), nested defaultdict, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDefaultdictPatterns:
    """Test defaultdict operation caching."""

    def test_defaultdict_list(self, nb_runner):
        """defaultdict(list) grouping pattern with caching."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "data = [('a', 1), ('b', 2), ('a', 3)]",
            "dd = defaultdict(list)\nfor k, v in data:\n    dd[k].append(v)\nresult = dict(sorted(dd.items()))",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result={'a': [1, 3], 'b': [2]}" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result={'a': [1, 3], 'b': [2]}" in out2

    def test_defaultdict_int_edit(self, nb_runner):
        """defaultdict(int) counting with edit."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "words = 'the cat sat on the mat the cat'.split()",
            "counts = defaultdict(int)\nfor w in words:\n    counts[w] += 1\nmost = max(counts, key=counts.get)",
            "print(f'most={most} count={counts[most]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "most=the" in out
        assert "count=3" in out

        nb_runner.set_cell_source(2, "words = 'a b a b a b a'.split()")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "most=a" in out2
        assert "count=4" in out2

    def test_defaultdict_nested(self, nb_runner):
        """Nested defaultdict pattern."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "data = [('US', 'NY', 100), ('US', 'CA', 200), ('UK', 'LN', 150)]",
            "nested = defaultdict(lambda: defaultdict(int))\nfor country, city, val in data:\n    nested[country][city] = val\nus_total = sum(nested['US'].values())",
            "print(f'us_total={us_total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "us_total=300" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "us_total=300" in out2
