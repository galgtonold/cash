"""Batch 272 – List accumulation and cross-cell aggregation.

Tests patterns where data is built across multiple cells then aggregated.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAccumulationAggregation:
    """Cross-cell accumulation and aggregation patterns."""

    def test_build_list_across_cells(self, nb_runner):
        """Build list in separate cells, aggregate in final."""
        nb_runner.create_notebook([
            "part1 = [1, 2, 3]",
            "part2 = [4, 5, 6]",
            "part3 = [7, 8, 9]",
            "combined = part1 + part2 + part3\ntotal = sum(combined)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "part2 = [40, 50, 60]")
        nb_runner.run_all()
        # 1+2+3+40+50+60+7+8+9 = 180
        assert "total = 180" in nb_runner.get_output(4)

    def test_dict_merge_across_cells(self, nb_runner):
        """Build dict across cells, query in final."""
        nb_runner.create_notebook([
            "user_data = {'name': 'Alice', 'age': 30}",
            "settings = {'theme': 'dark', 'lang': 'en'}",
            "profile = {**user_data, **settings}\nprint(f'profile = {sorted(profile.items())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "('name', 'Alice')" in out
        assert "('theme', 'dark')" in out

        nb_runner.set_cell_source(2, "settings = {'theme': 'light', 'lang': 'de'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "('theme', 'light')" in out2
        assert "('lang', 'de')" in out2

    def test_reduce_across_cells(self, nb_runner):
        """Multiple reduction steps across cells."""
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]",
            "filtered = [x for x in data if x >= 20]",
            "squared = [x**2 for x in filtered]",
            "avg = sum(squared) / len(squared)\nprint(f'avg = {avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # filtered=[20,30,40,50], squared=[400,900,1600,2500], avg=5400/4=1350
        assert "avg = 1350.0" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]")
        nb_runner.run_all()
        # filtered=[20..100], squared, avg
        out = nb_runner.get_output(4)
        assert "avg =" in out
