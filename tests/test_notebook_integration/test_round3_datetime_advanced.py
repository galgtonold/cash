"""Batch 95 – datetime and time complex patterns."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDatetimePatterns:
    """Complex datetime manipulation patterns."""

    def test_datetime_arithmetic(self, nb_runner):
        """Date arithmetic with timedelta."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from datetime import datetime, timedelta, date
                start = date(2024, 1, 15)
                end = start + timedelta(days=90)
                diff = (end - start).days
                mid = start + timedelta(days=diff // 2)
                is_leap = (date(start.year, 12, 31) - date(start.year, 1, 1)).days == 365
            """),
            "print(f'end={end} diff={diff} mid={mid} leap={is_leap}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "end=2024-04-14" in out
        assert "diff=90" in out
        assert "leap=True" in out  # 2024 is a leap year

    def test_datetime_formatting(self, nb_runner):
        """strftime/strptime formatting."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from datetime import datetime
                dt = datetime(2024, 6, 15, 14, 30, 0)
                iso = dt.isoformat()
                formatted = dt.strftime('%B %d, %Y at %I:%M %p')
                parsed = datetime.strptime('2024-12-25 08:00', '%Y-%m-%d %H:%M')
                parsed_str = parsed.strftime('%A, %B %d')
            """),
            "print(f'iso={iso}')\nprint(f'fmt={formatted}')\nprint(f'parsed={parsed_str}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "2024-06-15T14:30:00" in out
        assert "June 15, 2024" in out
        assert "Wednesday, December 25" in out

    def test_date_range_generation(self, nb_runner):
        """Generate date ranges and business days."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from datetime import date, timedelta
                start = date(2024, 1, 1)
                end = date(2024, 1, 15)
                all_days = []
                business_days = []
                current = start
                while current <= end:
                    all_days.append(current)
                    if current.weekday() < 5:  # Mon-Fri
                        business_days.append(current)
                    current += timedelta(days=1)
                total = len(all_days)
                biz = len(business_days)
            """),
            "print(f'total={total} business={biz}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "total=15" in out
        assert "business=11" in out

    def test_datetime_propagation(self, nb_runner):
        """Date computation with upstream change propagation."""
        nb_runner.create_notebook([
            "year = 2024",
            textwrap.dedent("""\
                from datetime import date
                jan1 = date(year, 1, 1)
                dec31 = date(year, 12, 31)
                days_in_year = (dec31 - jan1).days + 1
            """),
            "print(f'year={year} days={days_in_year}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "year=2024 days=366" in nb_runner.get_output(3)  # leap year

        nb_runner.set_cell_source(1, "year = 2023")
        nb_runner.run_cells([1, 2, 3])
        assert "year=2023 days=365" in nb_runner.get_output(3)  # not leap

    def test_time_zones_naive(self, nb_runner):
        """Timezone-naive datetime operations."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "150.0" in out  # 2.5 hours = 150 min between first two
        assert "span_hrs=" in out
