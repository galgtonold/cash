"""Batch 214 – Counter and accumulator interaction tests.

Tests editing cells that use Counter, defaultdict,
and accumulator patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestCounterPatternEdits:
    """Editing Counter and accumulator patterns."""

    def test_edit_counter_source(self, nb_runner):
        """Edit source data for Counter."""
        nb_runner.create_notebook([
            "from collections import Counter\nwords = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']",
            "counts = Counter(words)\nprint(f'most = {counts.most_common(2)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('apple', 3)" in nb_runner.get_output(2)

        # Change words
        nb_runner.set_cell_source(1, "from collections import Counter\nwords = ['x', 'y', 'x', 'x', 'y', 'z', 'z', 'z', 'z']")
        nb_runner.run_all()
        assert "('z', 4)" in nb_runner.get_output(2)

    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit defaultdict population."""
        nb_runner.create_notebook([
            "from collections import defaultdict\npairs = [('a', 1), ('b', 2), ('a', 3)]",
            "d = defaultdict(list)\nfor k, v in pairs:\n    d[k].append(v)\nprint(f'a = {d[\"a\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = [1, 3]" in nb_runner.get_output(2)

        # Change pairs
        nb_runner.set_cell_source(1, "from collections import defaultdict\npairs = [('a', 10), ('a', 20), ('b', 5)]")
        nb_runner.run_all()
        assert "a = [10, 20]" in nb_runner.get_output(2)

    def test_edit_running_total(self, nb_runner):
        """Edit accumulation source."""
        nb_runner.create_notebook([
            "transactions = [100, -50, 200, -75]",
            "balance = 0\nfor t in transactions:\n    balance += t\nprint(f'balance = {balance}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "balance = 175" in nb_runner.get_output(2)

        # Change transactions
        nb_runner.set_cell_source(1, "transactions = [500, -100, -200]")
        nb_runner.run_all()
        assert "balance = 200" in nb_runner.get_output(2)

    def test_edit_histogram(self, nb_runner):
        """Edit data for histogram-style grouping."""
        nb_runner.create_notebook([
            "scores = [85, 92, 78, 95, 88, 72, 91]",
            "bins = {'A': 0, 'B': 0, 'C': 0}\nfor s in scores:\n    if s >= 90:\n        bins['A'] += 1\n    elif s >= 80:\n        bins['B'] += 1\n    else:\n        bins['C'] += 1\nprint(f'bins = {bins}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'A': 3" in nb_runner.get_output(2)

        # Change scores
        nb_runner.set_cell_source(1, "scores = [60, 65, 70, 75]")
        nb_runner.run_all()
        assert "'A': 0" in nb_runner.get_output(2)
        assert "'C': 4" in nb_runner.get_output(2)
