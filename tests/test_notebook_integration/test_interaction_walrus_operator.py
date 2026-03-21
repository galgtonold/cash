"""Batch 210 – Walrus operator interaction tests.

Tests editing cells that use the walrus operator (:=)
in various contexts and verifying cache propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestWalrusOperatorEdits:
    """Editing walrus operator patterns."""

    def test_edit_walrus_in_while(self, nb_runner):
        """Edit a walrus operator used in accumulation."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "total = 0\nresults = []\nfor x in data:\n    total += x\n    results.append(total)\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [1, 3, 6, 10, 15]" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "results = [10, 30, 60]" in nb_runner.get_output(2)

    def test_edit_walrus_in_comprehension(self, nb_runner):
        """Edit list used in filtered comprehension."""
        nb_runner.create_notebook([
            "nums = [1, 5, 3, 8, 2, 9, 4]",
            "big = [x for x in nums if x > 4]\nprint(f'big = {big}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big = [5, 8, 9]" in nb_runner.get_output(2)

        # Edit threshold by changing the filter
        nb_runner.set_cell_source(2, "big = [x for x in nums if x > 3]\nprint(f'big = {big}')")
        nb_runner.run_all()
        assert "big = [5, 8, 9, 4]" in nb_runner.get_output(2)

    def test_edit_walrus_assignment(self, nb_runner):
        """Edit source data that feeds conditional logic."""
        nb_runner.create_notebook([
            "values = [2, 4, 6, 8, 10]",
            "even_count = sum(1 for v in values if v % 2 == 0)\nprint(f'count = {even_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(2)

        # Add odd numbers
        nb_runner.set_cell_source(1, "values = [1, 2, 3, 4, 5, 6]")
        nb_runner.run_all()
        assert "count = 3" in nb_runner.get_output(2)
