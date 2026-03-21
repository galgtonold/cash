"""Batch 244 – Walrus operator and assignment expression patterns.

Tests := operator in various contexts with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWalrusOperator:
    """Walrus operator patterns with edit propagation."""

    def test_walrus_in_while_condition(self, nb_runner):
        """Walrus op in while, edit threshold."""
        nb_runner.create_notebook([
            "data = [5, 3, 8, 1, 9, 2]",
            "results = []\ni = 0\nwhile i < len(data) and (val := data[i]) > 0:\n    results.append(val * 2)\n    i += 1\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [10, 6, 16, 2, 18, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [5, 3, 8, 1, 9, 2, 7, 4]")
        nb_runner.run_all()
        assert "results = [10, 6, 16, 2, 18, 4, 14, 8]" in nb_runner.get_output(2)

    def test_walrus_in_list_comp(self, nb_runner):
        """Walrus op in list comprehension, edit data."""
        nb_runner.create_notebook([
            "values = [1, 4, 9, 16, 25, 36]",
            "import math\nfiltered = [(y := math.sqrt(x), x) for x in values if (y := math.sqrt(x)) > 3]\nprint(f'filtered = {filtered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(4.0, 16)" in out
        assert "(5.0, 25)" in out

        nb_runner.set_cell_source(1, "values = [49, 64, 81, 100]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "(7.0, 49)" in out2
        assert "(10.0, 100)" in out2

    def test_walrus_in_if_chain(self, nb_runner):
        """Walrus in conditional, edit input data."""
        nb_runner.create_notebook([
            "text = 'Hello World Python'",
            "words = text.split()\nif (count := len(words)) > 2:\n    label = f'{count} words'\nelse:\n    label = 'short'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = 3 words" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "text = 'Hi'")
        nb_runner.run_all()
        assert "label = short" in nb_runner.get_output(2)
