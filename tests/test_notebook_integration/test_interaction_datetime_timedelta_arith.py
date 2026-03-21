"""Batch 509: datetime timedelta and date arithmetic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDatetimeTimedeltaArith:
    def test_date_arithmetic(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import date, timedelta",
            "d1 = date(2024, 1, 1)\nd2 = d1 + timedelta(days=100)\ndiff = d2 - d1\nprint(f'd2={d2.isoformat()} diff_days={diff.days}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d2=2024-04-10" in out
        assert "diff_days=100" in out

    def test_timedelta_components(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import timedelta",
            "td = timedelta(days=2, hours=3, minutes=30, seconds=45)\ntotal_sec = int(td.total_seconds())\nhours = total_sec // 3600\nmins = (total_sec % 3600) // 60\nprint(f'days={td.days} total_sec={total_sec} hours={hours} mins={mins}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "days=2" in out
        assert "total_sec=185445" in out

    def test_date_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import date, timedelta",
            "d = date(2024, 6, 15) + timedelta(days=30)\nprint(f'result={d.isoformat()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=2024-07-15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = date(2024, 12, 25) + timedelta(days=7)\nprint(f'result={d.isoformat()}')")
        nb_runner.run_all()
        assert "result=2025-01-01" in nb_runner.get_output(2)
