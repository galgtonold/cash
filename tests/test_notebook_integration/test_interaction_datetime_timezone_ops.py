"""
Interaction test: datetime timezone-aware operations with timedelta.
Tests timezone creation, conversion, timedelta arithmetic, and
cross-cell timezone-aware datetime manipulation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDatetimeTimezoneOps:
    """Test timezone-aware datetime operations across cells."""

    def test_timezone_conversions(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create timezone-aware datetimes
            "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\nest = timezone(timedelta(hours=-5))\njst = timezone(timedelta(hours=9))\nnow_utc = datetime(2024, 6, 15, 12, 0, 0, tzinfo=utc)\nprint(f'utc={now_utc.isoformat()}')",
            # Cell 2: convert between timezones
            "now_est = now_utc.astimezone(est)\nnow_jst = now_utc.astimezone(jst)\nprint(f'est_hour={now_est.hour}')\nprint(f'jst_hour={now_jst.hour}')\nprint(f'same_instant={now_utc == now_est == now_jst}')",
            # Cell 3: timedelta arithmetic
            "future = now_utc + timedelta(days=30, hours=6)\ndiff = future - now_utc\nprint(f'future_month={future.month}')\nprint(f'diff_days={diff.days}')\nprint(f'diff_secs={diff.seconds}')",
        ])
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
        nb_runner.create_notebook([
            "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\ndt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=utc)\nprint(f'year={dt.year}')",
            "tz_offset = timezone(timedelta(hours=5, minutes=30))\nconverted = dt.astimezone(tz_offset)\nprint(f'converted_hour={converted.hour}')\nprint(f'converted_min={converted.minute}')",
            "day_changed = converted.day != dt.day\nprint(f'day_changed={day_changed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "converted_hour=5" in nb_runner.get_output(2)
        assert "converted_min=30" in nb_runner.get_output(2)
        assert "day_changed=False" in nb_runner.get_output(3)

        # Change to a time that causes day rollover
        nb_runner.set_cell_source(1, "from datetime import datetime, timezone, timedelta\nutc = timezone.utc\ndt = datetime(2024, 1, 1, 20, 0, 0, tzinfo=utc)\nprint(f'year={dt.year}')")
        nb_runner.run_cells([1, 2, 3])
        # 20:00 UTC + 5:30 = 01:30 next day
        assert "converted_hour=1" in nb_runner.get_output(2)
        assert "converted_min=30" in nb_runner.get_output(2)
        assert "day_changed=True" in nb_runner.get_output(3)

    def test_timezone_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from datetime import datetime, timezone, timedelta\ndt = datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)\nprint(f'ts={dt.timestamp()}')",
            "epoch_diff = dt.timestamp() - datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()\nprint(f'epoch={int(epoch_diff)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "epoch=" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "epoch=" in nb_runner.get_output(2)
