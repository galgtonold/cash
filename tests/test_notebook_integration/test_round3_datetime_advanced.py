"""datetime and time complex patterns."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDatetimePatterns:
    """Complex datetime manipulation patterns."""

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
