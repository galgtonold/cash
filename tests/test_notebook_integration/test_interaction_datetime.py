"""Batch 191 – Datetime / time-based computation interaction tests.

Tests editing datetime computations, timedelta operations,
and formatting.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDatetimeEdits:
    """Editing datetime computations."""

    def test_edit_date_arithmetic(self, nb_runner):
        """Edit date arithmetic."""
        nb_runner.create_notebook([
            "from datetime import date, timedelta",
            "start = date(2024, 1, 1)  # date start",
            "end = start + timedelta(days=30)\nprint(f'end = {end}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "end = 2024-01-31" in nb_runner.get_output(3)

        # Change delta
        nb_runner.set_cell_source(
            3, "end = start + timedelta(days=365)\nprint(f'end = {end}')"
        )
        nb_runner.run_all()
        assert "end = 2024-12-31" in nb_runner.get_output(3)

    def test_edit_date_source(self, nb_runner):
        """Edit the source date."""
        nb_runner.create_notebook([
            "from datetime import date, timedelta",
            "d = date(2024, 6, 15)  # source date",
            "weekday = d.strftime('%A')\nprint(f'weekday = {weekday}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "weekday = Saturday" in nb_runner.get_output(3)

        # Change date
        nb_runner.set_cell_source(2, "d = date(2024, 12, 25)  # source date v2")
        nb_runner.run_all()
        assert "weekday = Wednesday" in nb_runner.get_output(3)


class TestDateFormatEdits:
    """Editing date formatting."""

    def test_edit_format_string(self, nb_runner):
        """Edit the date format string."""
        nb_runner.create_notebook([
            "from datetime import datetime",
            "dt = datetime(2024, 3, 14, 9, 26, 53)  # format source",
            "formatted = dt.strftime('%Y-%m-%d')\nprint(f'formatted = {formatted}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted = 2024-03-14" in nb_runner.get_output(3)

        # Change format
        nb_runner.set_cell_source(
            3, "formatted = dt.strftime('%d/%m/%Y %H:%M')\nprint(f'formatted = {formatted}')"
        )
        nb_runner.run_all()
        assert "formatted = 14/03/2024 09:26" in nb_runner.get_output(3)

    def test_edit_timedelta_chain(self, nb_runner):
        """Edit a chain of timedelta operations."""
        nb_runner.create_notebook([
            "from datetime import date, timedelta",
            "base = date(2024, 1, 1)  # timedelta chain base",
            "step1 = base + timedelta(weeks=4)  # step 1",
            "step2 = step1 + timedelta(days=10)  # step 2",
            "print(f'final = {step2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Jan 1 + 28 days = Jan 29, + 10 = Feb 8
        assert "final = 2024-02-08" in nb_runner.get_output(5)

        # Change step 1
        nb_runner.set_cell_source(3, "step1 = base + timedelta(weeks=8)  # step 1 v2")
        nb_runner.run_all()
        # Jan 1 + 56 days = Feb 26, + 10 = Mar 7
        assert "final = 2024-03-07" in nb_runner.get_output(5)
