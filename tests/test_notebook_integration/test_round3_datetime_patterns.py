"""
Batch 37: Datetime, time, and scheduling patterns across cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestDatetimePatterns:
    """Test datetime operations across cells."""

    def test_datetime_arithmetic(self, nb_runner):
        """Datetime arithmetic across cells."""
        nb_runner.create_notebook([
            "from datetime import datetime, timedelta",
            textwrap.dedent("""\
                start = datetime(2024, 1, 1)
                end = datetime(2024, 12, 31)
                duration = end - start
            """),
            textwrap.dedent("""\
                print(f"days={duration.days}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "days=365" in nb_runner.get_output(3)

    def test_timedelta_operations(self, nb_runner):
        """Timedelta operations across cells."""
        nb_runner.create_notebook([
            "from datetime import datetime, timedelta",
            textwrap.dedent("""\
                base = datetime(2024, 6, 15, 10, 30, 0)
                offset = timedelta(hours=5, minutes=30)
                result = base + offset
            """),
            textwrap.dedent("""\
                print(f"{result.hour}:{result.minute:02d}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "16:00" in nb_runner.get_output(3)

    def test_date_formatting(self, nb_runner):
        """Date formatting across cells."""
        nb_runner.create_notebook([
            "from datetime import datetime",
            "dt = datetime(2024, 3, 15, 14, 30, 0)",
            textwrap.dedent("""\
                formatted = dt.strftime('%Y-%m-%d %H:%M')
                print(formatted)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "2024-03-15 14:30" in nb_runner.get_output(3)

    def test_date_parsing_and_comparison(self, nb_runner):
        """Parse and compare dates across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "earliest=01/15 latest=03/20" in nb_runner.get_output(3)


class TestTimeSeriesPatterns:
    """Test time series operations with pandas across cells."""

    def test_pandas_date_range(self, nb_runner):
        """pandas date_range across cells."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            textwrap.dedent("""\
                np.random.seed(42)
                dates = pd.date_range('2024-01-01', periods=30, freq='D')
                ts = pd.Series(np.random.randn(30).cumsum(), index=dates, name='price')
            """),
            textwrap.dedent("""\
                weekly = ts.resample('W').mean()
                print(f"weeks={len(weekly)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "weeks=" in output

    def test_date_range_change_propagation(self, nb_runner):
        """Change date range → downstream aggregation updates."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # sum(0..9) = 45
        assert "total=45" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, textwrap.dedent("""\
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=5, freq='D')
            ts = pd.Series(range(5), index=dates, name='val')
        """))
        nb_runner.run_all()
        # sum(0..4) = 10
        assert "total=10" in nb_runner.get_output(3)


class TestDatetimeEdgeCases:
    """Test edge cases with dates."""

    def test_timezone_naive_operations(self, nb_runner):
        """Timezone-naive datetime operations."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Saturday" in output
        assert "7" in output

    def test_calendar_operations(self, nb_runner):
        """calendar module across cells."""
        nb_runner.create_notebook([
            "import calendar",
            "year = 2024",
            textwrap.dedent("""\
                leap = calendar.isleap(year)
                days_feb = calendar.monthrange(year, 2)[1]
                print(f"leap={leap} feb_days={days_feb}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leap=True feb_days=29" in nb_runner.get_output(3)
