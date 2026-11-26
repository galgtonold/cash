"""A cached function that sets the week's first day or the time zone runs
every time, and says so.

``calendar.setfirstweekday(6)`` and ``time.tzset()`` change a setting no
variable holds, so a hit did not make it again: in a new process (here,
the setting put back as one finds it) the call was served and
``calendar.month`` started its weeks on Monday, ``time.localtime``
answered in the zone the process started in.
"""

from __future__ import annotations

import calendar
import os
import time
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashImpurityWarning


def _week_start(x):
    calendar.setfirstweekday(6)
    return x + 1


def _time_zone(x):
    os.environ["TZ"] = "JST-9"  # @cash:assume-safe
    time.tzset()
    return x + 1


def _only_reads(x):
    return x + calendar.firstweekday() + len(time.tzname)


def _zone_as_started():
    os.environ["TZ"] = "UTC0"
    time.tzset()


CASES = {
    "week-start": (_week_start, calendar.firstweekday, lambda: calendar.setfirstweekday(0), 6),
    "time-zone": (_time_zone, lambda: time.timezone, _zone_as_started, -32400),
}


@pytest.fixture
def settings_put_back(monkeypatch):
    first = calendar.firstweekday()
    monkeypatch.setenv("TZ", os.environ.get("TZ", "UTC0"))
    try:
        yield
    finally:
        calendar.setfirstweekday(first)
        monkeypatch.undo()
        if hasattr(time, "tzset"):
            time.tzset()


@pytest.mark.usefixtures("settings_put_back")
@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_made_on_every_call(tmp_path, kind):
    if kind == "time-zone" and not hasattr(time, "tzset"):
        pytest.skip("no time.tzset on this platform")
    func, read, reset, made = CASES[kind]
    cached = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(func)
    reset()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        first = cached(1)
        for _ in range(2):
            # As a new process finds it.
            reset()
            assert cached(1) == first
            assert read() == made, "served without making the setting"
    said = [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]
    assert any("no variable holds" in text for text in said), said
    assert any(("calendar.setfirstweekday" if kind == "week-start" else "time.tzset") in text for text in said), said


@pytest.mark.usefixtures("settings_put_back")
def test_a_function_that_only_reads_them_is_stored_without_a_word(tmp_path):
    cached = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(_only_reads)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert cached(1) == cached(1)
    assert cached.cache_info()["hits"] == 1, cached.cache_info()
    assert not [w for w in record if "no variable holds" in str(w.message)], [str(w.message) for w in record]
