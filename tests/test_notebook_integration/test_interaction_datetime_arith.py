"""
Batch 298: Datetime arithmetic and formatting interaction tests.
Tests editing datetime operations (different from batch 267's basic datetime test).
Focus on timedelta chains and date range generation.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestDatetimeArithmeticInteraction:
    """Test datetime arithmetic patterns with cache invalidation."""

    def test_timedelta_chain_edit(self, nb_runner):
        """Editing chained timedelta operations should propagate."""
        nb_runner.create_notebook([
            "from datetime import datetime, timedelta\nbase = datetime(2024, 1, 1)",
            "d1 = timedelta(days=10)\nd2 = timedelta(hours=5)",
            "result_dt = base + d1 + d2",
            "print(f'result={result_dt.strftime(\"%Y-%m-%d %H:%M\")}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=2024-01-11 05:00" in out

        nb_runner.set_cell_source(2, "d1 = timedelta(days=100)\nd2 = timedelta(hours=12)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=2024-04-10 12:00" in out

    def test_date_range_count_edit(self, nb_runner):
        """Editing number of days in a date range should propagate."""
        nb_runner.create_notebook([
            "from datetime import date, timedelta\nstart = date(2024, 3, 1)",
            "n_days = 3",
            "dates = [start + timedelta(days=i) for i in range(n_days)]",
            "result = len(dates)",
            "print(f'count={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=3" in out

        nb_runner.set_cell_source(2, "n_days = 7")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=7" in out
