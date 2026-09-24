"""datetime, timedelta, time zones and calendar across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


class TestDatetimePatterns:
    """Test datetime operations across cells."""

    @pytest.mark.integration
    def test_date_formatting(self, nb_runner):
        """Date formatting across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime",
                "dt = datetime(2024, 3, 15, 14, 30, 0)",
                textwrap.dedent("""\
                formatted = dt.strftime('%Y-%m-%d %H:%M')
                print(formatted)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2024-03-15 14:30" in nb_runner.get_output(3)

    @pytest.mark.integration
    def test_date_parsing_and_comparison(self, nb_runner):
        """Parse and compare dates across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime",
                textwrap.dedent("""\
                dates_str = ['2024-01-15', '2024-03-20', '2024-02-10']
                dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]
            """),
                textwrap.dedent("""\
                earliest = min(dates)
                latest = max(dates)
                print(f"earliest={earliest.strftime('%m/%d')} latest={latest.strftime('%m/%d')}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "earliest=01/15 latest=03/20" in nb_runner.get_output(3)

    # datetime and time complex patterns.
    @pytest.mark.integration
    def test_datetime_formatting(self, nb_runner):
        """strftime/strptime formatting."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from datetime import datetime
                dt = datetime(2024, 6, 15, 14, 30, 0)
                iso = dt.isoformat()
                formatted = dt.strftime('%B %d, %Y at %I:%M %p')
                parsed = datetime.strptime('2024-12-25 08:00', '%Y-%m-%d %H:%M')
                parsed_str = parsed.strftime('%A, %B %d')
            """),
                "print(f'iso={iso}')\nprint(f'fmt={formatted}')\nprint(f'parsed={parsed_str}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "2024-06-15T14:30:00" in out
        assert "June 15, 2024" in out
        assert "Wednesday, December 25" in out

    @pytest.mark.integration
    def test_datetime_propagation(self, nb_runner):
        """Date computation with upstream change propagation."""
        nb_runner.create_notebook(
            [
                "year = 2024",
                textwrap.dedent("""\
                from datetime import date
                jan1 = date(year, 1, 1)
                dec31 = date(year, 12, 31)
                days_in_year = (dec31 - jan1).days + 1
            """),
                "print(f'year={year} days={days_in_year}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "year=2024 days=366" in nb_runner.get_output(3)  # leap year

        nb_runner.set_cell_source(1, "year = 2023")
        nb_runner.run_cells([1, 2, 3])
        assert "year=2023 days=365" in nb_runner.get_output(3)  # not leap

    @pytest.mark.integration
    def test_time_zones_naive(self, nb_runner):
        """Timezone-naive datetime operations."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from datetime import datetime, timedelta
                meetings = [
                    datetime(2024, 3, 15, 9, 0),
                    datetime(2024, 3, 15, 11, 30),
                    datetime(2024, 3, 15, 14, 0),
                    datetime(2024, 3, 15, 16, 45),
                ]
                gaps = []
                for i in range(len(meetings) - 1):
                    gap = meetings[i + 1] - meetings[i]
                    gaps.append(gap.total_seconds() / 60)
                total_meeting_span = (meetings[-1] - meetings[0]).total_seconds() / 3600
            """),
                "print(f'gaps_min={gaps}')\nprint(f'span_hrs={total_meeting_span}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "150.0" in out  # 2.5 hours = 150 min between first two
        assert "span_hrs=" in out


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDatetimeEdits:
    """Editing datetime computations."""

    def test_edit_date_arithmetic(self, nb_runner):
        """Edit date arithmetic."""
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta",
                "start = date(2024, 1, 1)  # date start",
                "end = start + timedelta(days=30)\nprint(f'end = {end}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "end = 2024-01-31" in nb_runner.get_output(3)

        # Change delta
        nb_runner.set_cell_source(3, "end = start + timedelta(days=365)\nprint(f'end = {end}')")
        nb_runner.run_all()
        assert "end = 2024-12-31" in nb_runner.get_output(3)

    def test_edit_date_source(self, nb_runner):
        """Edit the source date."""
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta",
                "d = date(2024, 6, 15)  # source date",
                "weekday = d.strftime('%A')\nprint(f'weekday = {weekday}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "weekday = Saturday" in nb_runner.get_output(3)

        # Change date
        nb_runner.set_cell_source(2, "d = date(2024, 12, 25)  # source date v2")
        nb_runner.run_all()
        assert "weekday = Wednesday" in nb_runner.get_output(3)


@pytest.mark.timeout(90)
class TestDatetimeArithmeticOps:
    """datetime arithmetic and timedelta operations."""

    def test_timedelta_add(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta\nstart = date(2024, 1, 1)",
                "end = start + timedelta(days=30)\ndiff = (end - start).days\nprint(f'end={end.isoformat()} diff={diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "end=2024-01-31" in nb_runner.get_output(2)
        assert "diff=30" in nb_runner.get_output(2)

    def test_datetime_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import date\nd = date(2024, 3, 15)",
                "weekday = d.strftime('%A')\niso = d.isoformat()\nprint(f'weekday={weekday} iso={iso}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "weekday=Friday" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from datetime import date\nd = date(2024, 12, 25)")
        nb_runner.run_all()
        assert "weekday=Wednesday" in nb_runner.get_output(2)
        assert "iso=2024-12-25" in nb_runner.get_output(2)

    def test_date_range(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta\nstart = date(2024, 1, 1)\nend = date(2024, 1, 5)",
                "dates = []\ncurrent = start\nwhile current <= end:\n    dates.append(current.day)\n    current += timedelta(days=1)\nprint(f'dates={dates}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dates=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestDatetimeTimedeltaArith:
    """datetime timedelta and date arithmetic."""

    def test_date_arithmetic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta",
                "d1 = date(2024, 1, 1)\nd2 = d1 + timedelta(days=100)\ndiff = d2 - d1\nprint(f'd2={d2.isoformat()} diff_days={diff.days}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d2=2024-04-10" in out
        assert "diff_days=100" in out

    def test_timedelta_components(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import timedelta",
                "td = timedelta(days=2, hours=3, minutes=30, seconds=45)\ntotal_sec = int(td.total_seconds())\nhours = total_sec // 3600\nmins = (total_sec % 3600) // 60\nprint(f'days={td.days} total_sec={total_sec} hours={hours} mins={mins}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "days=2" in out
        assert "total_sec=185445" in out

    def test_date_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta",
                "d = date(2024, 6, 15) + timedelta(days=30)\nprint(f'result={d.isoformat()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=2024-07-15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = date(2024, 12, 25) + timedelta(days=7)\nprint(f'result={d.isoformat()}')")
        nb_runner.run_all()
        assert "result=2025-01-01" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.timeout(90)
class TestDatetimeArithmeticInteraction:
    """Test datetime arithmetic patterns with cache invalidation."""

    def test_timedelta_chain_edit(self, nb_runner):
        """Editing chained timedelta operations should propagate."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timedelta\nbase = datetime(2024, 1, 1)",
                "d1 = timedelta(days=10)\nd2 = timedelta(hours=5)",
                "result_dt = base + d1 + d2",
                "print(f'result={result_dt.strftime(\"%Y-%m-%d %H:%M\")}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta\nstart = date(2024, 3, 1)",
                "n_days = 3",
                "dates = [start + timedelta(days=i) for i in range(n_days)]",
                "result = len(dates)",
                "print(f'count={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=3" in out

        nb_runner.set_cell_source(2, "n_days = 7")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=7" in out


@pytest.mark.timeout(90)
class TestDatetimeTimezoneOps:
    """Test timezone-aware datetime operations across cells."""

    def test_timezone_conversions(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create timezone-aware datetimes
                "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\nest = timezone(timedelta(hours=-5))\njst = timezone(timedelta(hours=9))\nnow_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=utc)\nprint(f'utc={now_utc.isoformat()}')",
                # Cell 2: convert between timezones
                "now_est = now_utc.astimezone(est)\nnow_jst = now_utc.astimezone(jst)\nprint(f'est_hour={now_est.hour}')\nprint(f'jst_hour={now_jst.hour}')\nprint(f'same_instant={now_utc == now_est == now_jst}')",
                # Cell 3: timedelta arithmetic
                "future = now_utc + timedelta(days=30, hours=6)\ndiff = future - now_utc\nprint(f'future_month={future.month}')\nprint(f'diff_days={diff.days}')\nprint(f'diff_secs={diff.seconds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "utc=2024-06-15T12:00:00+00:00" in out1
        out2 = nb_runner.get_output(2)
        assert "est_hour=7" in out2
        assert "jst_hour=21" in out2
        assert "same_instant=True" in out2
        out3 = nb_runner.get_output(3)
        assert "future_month=7" in out3
        assert "diff_days=30" in out3
        assert "diff_secs=21600" in out3

    def test_timezone_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\ndt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=utc)\nprint(f'year={dt.year}')",
                "tz_offset = timezone(timedelta(hours=5, minutes=30))\nconverted = dt.astimezone(tz_offset)\nprint(f'converted_hour={converted.hour}')\nprint(f'converted_min={converted.minute}')",
                "day_changed = converted.day != dt.day\nprint(f'day_changed={day_changed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "converted_hour=5" in nb_runner.get_output(2)
        assert "converted_min=30" in nb_runner.get_output(2)
        assert "day_changed=False" in nb_runner.get_output(3)

        # Change to a time that causes day rollover
        nb_runner.set_cell_source(
            1,
            "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\ndt = datetime(2024, 1, 1, 20, 0, 0, tzinfo=utc)\nprint(f'year={dt.year}')",
        )
        nb_runner.run_cells([1, 2, 3])
        # 20:00 UTC + 5:30 = 01:30 next day
        assert "converted_hour=1" in nb_runner.get_output(2)
        assert "converted_min=30" in nb_runner.get_output(2)
        assert "day_changed=True" in nb_runner.get_output(3)

    def test_timezone_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timezone, timedelta\ndt = datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)\nprint(f'ts={dt.timestamp()}')",
                "epoch_diff = dt.timestamp() - datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()\nprint(f'epoch={int(epoch_diff)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "epoch=" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "epoch=" in nb_runner.get_output(2)


@pytest.mark.integration
class TestDatetimeEdgeCases:
    """Test edge cases with dates."""

    def test_timezone_naive_operations(self, nb_runner):
        """Timezone-naive datetime operations."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timedelta",
                textwrap.dedent("""\
                now = datetime(2024, 6, 15, 12, 0, 0)
                intervals = [timedelta(days=d) for d in range(7)]
                week = [now + dt for dt in intervals]
            """),
                textwrap.dedent("""\
                weekdays = [d.strftime('%A') for d in week]
                print(weekdays[0])  # Saturday
                print(len(weekdays))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Saturday" in output
        assert "7" in output


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDateFormatEdits:
    """Editing date formatting."""

    def test_edit_format_string(self, nb_runner):
        """Edit the date format string."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime",
                "dt = datetime(2024, 3, 14, 9, 26, 53)  # format source",
                "formatted = dt.strftime('%Y-%m-%d')\nprint(f'formatted = {formatted}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted = 2024-03-14" in nb_runner.get_output(3)

        # Change format
        nb_runner.set_cell_source(3, "formatted = dt.strftime('%d/%m/%Y %H:%M')\nprint(f'formatted = {formatted}')")
        nb_runner.run_all()
        assert "formatted = 14/03/2024 09:26" in nb_runner.get_output(3)

    def test_edit_timedelta_chain(self, nb_runner):
        """Edit a chain of timedelta operations."""
        nb_runner.create_notebook(
            [
                "from datetime import date, timedelta",
                "base = date(2024, 1, 1)  # timedelta chain base",
                "step1 = base + timedelta(weeks=4)  # step 1",
                "step2 = step1 + timedelta(days=10)  # step 2",
                "print(f'final = {step2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Jan 1 + 28 days = Jan 29, + 10 = Feb 8
        assert "final = 2024-02-08" in nb_runner.get_output(5)

        # Change step 1
        nb_runner.set_cell_source(3, "step1 = base + timedelta(weeks=8)  # step 1 v2")
        nb_runner.run_all()
        # Jan 1 + 56 days = Feb 26, + 10 = Mar 7
        assert "final = 2024-03-07" in nb_runner.get_output(5)


@pytest.mark.timeout(90)
class TestCalendarMonthWeek:
    """Test calendar month and week operations across cells."""

    def test_calendar_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: month calendar
                "import calendar\nweeks = calendar.monthcalendar(2024, 2)  # Feb 2024\nnum_weeks = len(weeks)\ndays_in_month = calendar.monthrange(2024, 2)[1]\nprint(f'num_weeks={num_weeks}')\nprint(f'days_in_feb_2024={days_in_month}')",
                # Cell 2: weekday for specific dates
                "day_name = calendar.day_name[calendar.weekday(2024, 1, 1)]  # Jan 1 2024\nprint(f'jan1_2024={day_name}')\nis_leap = calendar.isleap(2024)\nprint(f'is_leap_2024={is_leap}')",
                # Cell 3: count weekdays in month
                "weekdays_in_feb = sum(1 for week in weeks for day in week if day != 0 and calendar.weekday(2024, 2, day) < 5)\nprint(f'weekdays_feb_2024={weekdays_in_feb}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import calendar\nyear = 2023\nis_leap = calendar.isleap(year)\ndays_feb = calendar.monthrange(year, 2)[1]\nprint(f'leap={is_leap}')\nprint(f'feb_days={days_feb}')",
                "total_days = sum(calendar.monthrange(year, m)[1] for m in range(1, 13))\nprint(f'total_days={total_days}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leap=False" in nb_runner.get_output(1)
        assert "feb_days=28" in nb_runner.get_output(1)
        assert "total_days=365" in nb_runner.get_output(2)

        # Change to leap year
        nb_runner.set_cell_source(
            1,
            "import calendar\nyear = 2024\nis_leap = calendar.isleap(year)\ndays_feb = calendar.monthrange(year, 2)[1]\nprint(f'leap={is_leap}')\nprint(f'feb_days={days_feb}')",
        )
        nb_runner.run_cells([1, 2])
        assert "leap=True" in nb_runner.get_output(1)
        assert "feb_days=29" in nb_runner.get_output(1)
        assert "total_days=366" in nb_runner.get_output(2)

    def test_calendar_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import calendar\nfirst_weekday, num_days = calendar.monthrange(2024, 7)\nprint(f'first_weekday={first_weekday}')\nprint(f'num_days={num_days}')",
                "info = f'July 2024: starts on day {first_weekday}, has {num_days} days'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "num_days=31" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "num_days=31" in nb_runner.get_output(1)


@pytest.mark.integration
class TestTimeSeriesPatterns:
    """Test time series operations with pandas across cells."""

    def test_date_range_change_propagation(self, nb_runner):
        """Change date range → downstream aggregation updates."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                textwrap.dedent("""\
                np.random.seed(42)
                dates = pd.date_range('2024-01-01', periods=10, freq='D')
                ts = pd.Series(range(10), index=dates, name='val')
            """),
                textwrap.dedent("""\
                total = ts.sum()
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # sum(0..9) = 45
        assert "total=45" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=5, freq='D')
            ts = pd.Series(range(5), index=dates, name='val')
        """),
        )
        nb_runner.run_all()
        # sum(0..4) = 10
        assert "total=10" in nb_runner.get_output(3)
