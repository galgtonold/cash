"""A pickle without the memo writes the attributes a C base's reduce drops.

`kept_state` restores such an instance's ``__dict__`` through a state setter,
which pickle hands the object written a second time. Without the memo
(`pickle_unshared`, `content_dumps`) that second write is not a
back-reference but the object reduced again, without end: the C pickler
crashed the process -- a ``datetime`` in a frame's list whose ``tzinfo`` is a
class of the user's. Those bytes are only ever hashed, so the attributes are
written as data instead.

Both ways a value reaches the pickler: directly, and walked on the heap first
(`kept_state.FLAT_FIRST`, Python 3.10 on Windows), whose reducers went
without the memo-less rule and recursed there.
"""

from __future__ import annotations

import datetime

import pytest

from cash import _plain_data, kept_state

pytestmark = [pytest.mark.core]


@pytest.fixture(autouse=True, params=[False, True], ids=["pickled", "walked first"])
def _flat_first(request, monkeypatch):
    monkeypatch.setattr(kept_state, "FLAT_FIRST", request.param)


class Zone(datetime.tzinfo):
    def __init__(self, hours):
        self.hours = hours

    def utcoffset(self, dt):
        return datetime.timedelta(hours=self.hours)


class Tagged(datetime.date):
    """A C base whose reduce leaves the subclass's ``__dict__`` out."""


def _tagged(tag):
    value = Tagged(2021, 1, 1)
    value.tag = tag
    return value


@pytest.mark.parametrize("dumps", [_plain_data.pickle_unshared, _plain_data.content_dumps])
def test_an_attribute_of_the_subclass_is_in_the_bytes(dumps):
    assert dumps([_tagged(1)]) == dumps([_tagged(1)]) != dumps([_tagged(2)])


@pytest.mark.parametrize("dumps", [_plain_data.pickle_unshared, _plain_data.content_dumps])
def test_a_time_zone_of_the_users_is_in_the_bytes(dumps):
    def at(hours):
        return [datetime.datetime(2021, 1, 1, tzinfo=Zone(hours))]

    assert dumps(at(1)) == dumps(at(1)) != dumps(at(5))
