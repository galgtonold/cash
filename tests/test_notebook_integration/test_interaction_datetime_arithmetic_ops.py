"""Batch 399: datetime arithmetic and timedelta operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDatetimeArithmeticOps:
    def test_timedelta_add(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import date, timedelta\nstart = date(2024, 1, 1)",
            "end = start + timedelta(days=30)\ndiff = (end - start).days\nprint(f'end={end.isoformat()} diff={diff}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "end=2024-01-31" in nb_runner.get_output(2)
        assert "diff=30" in nb_runner.get_output(2)

    def test_datetime_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import date\nd = date(2024, 3, 15)",
            "weekday = d.strftime('%A')\niso = d.isoformat()\nprint(f'weekday={weekday} iso={iso}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "weekday=Friday" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from datetime import date\nd = date(2024, 12, 25)")
        nb_runner.run_all()
        assert "weekday=Wednesday" in nb_runner.get_output(2)
        assert "iso=2024-12-25" in nb_runner.get_output(2)

    def test_date_range(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import date, timedelta\nstart = date(2024, 1, 1)\nend = date(2024, 1, 5)",
            "dates = []\ncurrent = start\nwhile current <= end:\n    dates.append(current.day)\n    current += timedelta(days=1)\nprint(f'dates={dates}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dates=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
