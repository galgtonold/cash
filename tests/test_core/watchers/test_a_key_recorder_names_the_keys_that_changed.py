"""``cash._cwatch.record(d)``: the keys of a dict that change while it
records, each with what it held before its first change, and the dict's
keys in order before that change. The decorator records an instance's
``__dict__`` across a call this way, to replay the attributes the call
bound (`cash.rewrite_in_place.AttrRecording`)."""

from __future__ import annotations

import gc

import pytest

from cash._native import cwatch

pytestmark = pytest.mark.skipif(cwatch is None, reason="the C extension is not loaded (Python below 3.12, or off)")


class Thing:
    def __init__(self):
        self.a = 1
        self.b = [2]


def test_each_changed_key_comes_with_what_it_held_before():
    t = Thing()
    old_b = t.b
    recorder = cwatch.record(vars(t))
    t.c = 3
    t.b = [5]
    t.b = [6]
    del t.a
    t.d = 4
    del t.d
    keys, before, flags = recorder.take()
    assert flags == 0
    assert keys == ["a", "b"]
    assert before["b"] is old_b and before["a"] == 1
    assert before["c"] is cwatch.ABSENT and before["d"] is cwatch.ABSENT
    # Taken: what changes after it is not recorded.
    t.e = 1
    assert "e" not in recorder.take()[1]


def test_nothing_changed_reads_as_no_keys():
    t = Thing()
    recorder = cwatch.record(vars(t))
    assert t.a == 1  # a read is no change
    assert recorder.take() == (None, {}, 0)


def test_a_cleared_dict_is_flagged():
    t = Thing()
    recorder = cwatch.record(vars(t))
    vars(t).clear()
    assert recorder.take()[2] & cwatch.REC_CLEARED


def test_a_key_that_is_not_a_str_is_flagged():
    t = Thing()
    recorder = cwatch.record(vars(t))
    vars(t)[1] = "one"
    assert recorder.take()[2] & cwatch.REC_ODD


def test_two_recorders_on_one_dict_each_see_their_own_span():
    t = Thing()
    outer = cwatch.record(vars(t))
    t.x = 1
    inner = cwatch.record(vars(t))
    t.x = 2
    t.y = 3
    assert inner.take()[1] == {"x": 1, "y": cwatch.ABSENT}
    t.z = 4
    keys, before, _ = outer.take()
    assert keys == ["a", "b"]
    assert before == {"x": cwatch.ABSENT, "y": cwatch.ABSENT, "z": cwatch.ABSENT}


def test_a_recorder_dropped_without_take_stops_recording():
    t = Thing()
    recorder = cwatch.record(vars(t))
    del recorder
    gc.collect()
    t.a = 9  # the watcher, if still on the dict, finds no recorder
    assert t.a == 9


def test_specialised_attribute_stores_reach_the_recorder():
    """A method whose ``self.x = ...`` the interpreter has specialised
    (inline values, 3.13+) still writes through the recorded dict."""

    class Warm:
        def set(self, v):
            self.x = v

    for _ in range(100):
        Warm().set(1)
    w = Warm()
    w.set(0)
    recorder = cwatch.record(vars(w))
    for v in range(1, 5):
        w.set(v)
    assert recorder.take()[1] == {"x": 0}
