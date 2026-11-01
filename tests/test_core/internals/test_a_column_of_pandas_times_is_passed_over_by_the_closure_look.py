"""The closure check's first look passes over pandas ``Timestamp`` and
``Timedelta`` values, which pickle as numbers and a time zone: a frame
concatenated from frames with and without times keeps one per row in an
object column, and each was read through its ``__reduce_ex__``."""

from __future__ import annotations

import datetime

import pytest

from cash import closure_sharing

pd = pytest.importorskip("pandas")


def _frame(tz=None):
    times = pd.DataFrame({"at": [datetime.datetime(2020, 1, 1, 0, 0, s) for s in range(50)]})
    if tz is not None:
        times["at"] = times["at"].dt.tz_localize(tz)
    frame = pd.concat([pd.DataFrame({"at": []}), times, pd.DataFrame({"at": [pd.Timedelta(5)]})], ignore_index=True)
    assert frame["at"].dtype == object
    return frame


def _reductions(monkeypatch) -> list:
    seen: list = []
    reduce = closure_sharing.reduce_counted

    def counted(item, protocol):
        seen.append(type(item))
        return reduce(item, protocol)

    monkeypatch.setattr(closure_sharing, "reduce_counted", counted)
    return seen


def test_no_time_is_read_without_a_function(monkeypatch):
    seen = _reductions(monkeypatch)
    assert closure_sharing.closures_sharing({"df": _frame()}, 5.0).names == []
    assert pd.Timestamp not in seen and pd.Timedelta not in seen


class _Zone(datetime.tzinfo):
    def __init__(self, fn):
        self.fn = fn

    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def dst(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        return "Z"


def test_a_function_in_a_times_zone_is_found():
    box = {"a": [1]}
    zone = _Zone(lambda: box["a"])
    frame = _frame(zone)
    assert closure_sharing.closures_sharing({"df": frame, "box": box}, 5.0).names == ["df"]
