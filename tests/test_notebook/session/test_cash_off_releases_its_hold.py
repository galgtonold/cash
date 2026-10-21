"""%cash_off puts back the I/O watch hold %cash_on took.

The autouse fixture in tests/conftest.py closes every %cash_on hold a test
left open, so no other test would fail if %cash_off itself stopped
releasing the watch. This one checks it inside the test, before the
fixture runs.
"""

from cash.tracking import io_watch


def test_cash_off_releases_the_hold_cash_on_took(cash_magics):
    assert not io_watch.holding()
    cash_magics.cash_on("")
    cash_magics.cash_on("")  # a second %cash_on takes no second hold
    assert io_watch.holding()
    cash_magics.cash_off("")
    assert not io_watch.holding()
