"""
Interaction test: calendar module month and week operations.
Tests calendar.monthcalendar, weekday calculation,
isleap checks, and cross-cell date analysis.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCalendarMonthWeek:
    """Test calendar month and week operations across cells."""

    def test_calendar_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: month calendar
            "import calendar\nweeks = calendar.monthcalendar(2024, 2)  # Feb 2024\nnum_weeks = len(weeks)\ndays_in_month = calendar.monthrange(2024, 2)[1]\nprint(f'num_weeks={num_weeks}')\nprint(f'days_in_feb_2024={days_in_month}')",
            # Cell 2: weekday for specific dates
            "day_name = calendar.day_name[calendar.weekday(2024, 1, 1)]  # Jan 1 2024\nprint(f'jan1_2024={day_name}')\nis_leap = calendar.isleap(2024)\nprint(f'is_leap_2024={is_leap}')",
            # Cell 3: count weekdays in month
            "weekdays_in_feb = sum(1 for week in weeks for day in week if day != 0 and calendar.weekday(2024, 2, day) < 5)\nprint(f'weekdays_feb_2024={weekdays_in_feb}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "days_in_feb_2024=29" in out1
        out2 = nb_runner.get_output(2)
        assert "jan1_2024=Monday" in out2
        assert "is_leap_2024=True" in out2
        out3 = nb_runner.get_output(3)
        assert "weekdays_feb_2024=21" in out3

    def test_calendar_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import calendar\nyear = 2023\nis_leap = calendar.isleap(year)\ndays_feb = calendar.monthrange(year, 2)[1]\nprint(f'leap={is_leap}')\nprint(f'feb_days={days_feb}')",
            "total_days = sum(calendar.monthrange(year, m)[1] for m in range(1, 13))\nprint(f'total_days={total_days}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leap=False" in nb_runner.get_output(1)
        assert "feb_days=28" in nb_runner.get_output(1)
        assert "total_days=365" in nb_runner.get_output(2)

        # Change to leap year
        nb_runner.set_cell_source(1, "import calendar\nyear = 2024\nis_leap = calendar.isleap(year)\ndays_feb = calendar.monthrange(year, 2)[1]\nprint(f'leap={is_leap}')\nprint(f'feb_days={days_feb}')")
        nb_runner.run_cells([1, 2])
        assert "leap=True" in nb_runner.get_output(1)
        assert "feb_days=29" in nb_runner.get_output(1)
        assert "total_days=366" in nb_runner.get_output(2)

    def test_calendar_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import calendar\nfirst_weekday, num_days = calendar.monthrange(2024, 7)\nprint(f'first_weekday={first_weekday}')\nprint(f'num_days={num_days}')",
            "info = f'July 2024: starts on day {first_weekday}, has {num_days} days'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "num_days=31" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "num_days=31" in nb_runner.get_output(1)
