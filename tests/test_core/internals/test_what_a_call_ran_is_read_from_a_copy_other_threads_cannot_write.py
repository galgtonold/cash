"""What a traced call ran is read from a copy no other thread writes to.

The recorder's callbacks run on whichever thread starts code and write
into every call recording then. One on another thread that picked a call
up just before the call ended wrote into its record while the call's own
thread was reading it: "dictionary changed size during iteration", raised
out of the cached call after its body had finished fine.

The test holds such a callback at the point where it has picked the call
up, lets the body end, and releases it just before the call's record is
read: a write that late lands in nothing the call is judged by.
"""

from __future__ import annotations

import threading
import types

import pytest

from cash.decorator import code_trace
from cash.tracking.trace_recorder import Unit

WAIT = 20


def test_a_callback_still_writing_when_the_call_ends_does_not_reach_its_record(disk_cash, monkeypatch):
    start, in_callback, go, wrote = (threading.Event() for _ in range(4))
    late: list[types.CodeType] = []

    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": range(5), "b": list("xyxyx")})

    def other_work():
        assert start.wait(WAIT)
        frame.describe()  # library code the call itself never runs

    other = threading.Thread(target=other_work)
    other.start()

    @disk_cash.cache
    def score(x):
        start.set()
        if not in_callback.wait(WAIT):
            raise RuntimeError("the other thread's library code was not seen")
        return x * 2

    recorder = code_trace.recorder_or_reason()
    assert recorder.tool is not None, recorder.reason
    real_seen = type(recorder)._seen_library_now

    def seen(self, code, frame, units):
        if threading.current_thread() is other and not late:
            late.append(code)
            in_callback.set()  # it holds the call's unit; the body may end
            assert go.wait(WAIT)
            try:
                return real_seen(self, code, frame, units)
            finally:
                wrote.set()
        return real_seen(self, code, frame, units)

    monkeypatch.setattr(type(recorder), "_seen_library_now", seen)
    real_codes = code_trace.library_codes
    read: list[dict] = []

    def library_codes(unit):
        go.set()  # the call ended: now the held callback writes
        assert wrote.wait(WAIT)
        read.append(real_codes(unit))
        return read[-1]

    monkeypatch.setattr(code_trace, "library_codes", library_codes)
    try:
        assert score(21) == 42
    finally:
        go.set()
        other.join(WAIT)

    assert late and read, "the callback was held and the record was read"
    assert late[0] not in read[0], "code another thread started as the call ended is in the call's record"


def test_a_detached_unit_shares_no_container_with_the_one_callbacks_hold():
    unit = code_trace.TraceUnit("decorator") if code_trace.TraceUnit is not None else None
    if unit is None:
        code_trace.recorder_or_reason()
        unit = code_trace.TraceUnit("decorator")
    code = test_a_detached_unit_shares_no_container_with_the_one_callbacks_hold.__code__
    other = _a_helper.__code__
    module_globals = globals()
    unit.codes[code] = (module_globals, __name__)
    unit.lib_codes[code] = module_globals
    unit.lib_keys[code] = {"display.width"}
    unit.line_ran = {code: {1, 2}}
    unit.attr_params[code] = {"self": {1}}
    unit.attr_targets[code] = {"self": {1: False}}
    unit.sites[code] = {(code, 4)}
    unit.lib_items.append(("setting", "pandas", "display.width"))
    unit.markers.add("m")

    done = unit.detached()

    assert type(done) is type(unit) and done.kind == "decorator" and done.thread == unit.thread
    assert done.lib_codes[code] is module_globals, "a code's globals are the dict itself, never a copy"
    assert done.codes == unit.codes and done.lib_keys == unit.lib_keys and done.line_ran == unit.line_ran

    # What a callback on another thread does to the unit it still holds:
    unit.codes[other] = (module_globals, __name__)
    unit.lib_codes[other] = module_globals
    unit.lib_keys[code].add(None)
    unit.line_ran[code].add(3)
    unit.attr_params[code]["self"].add(2)
    unit.attr_targets[code]["self"][2] = False
    unit.sites[code].add((other, 8))
    unit.lib_items.append(("late",))
    unit.markers.add("late")

    assert other not in done.codes and other not in done.lib_codes
    assert done.lib_keys[code] == {"display.width"}
    assert done.line_ran[code] == {1, 2}
    assert done.attr_params[code] == {"self": {1}}
    assert done.attr_targets[code] == {"self": {1: False}}
    assert done.sites[code] == {(code, 4)}
    assert done.lib_items == [("setting", "pandas", "display.width")]
    assert done.markers == {"m"}


def test_an_inner_unit_is_absorbed_from_a_copy(monkeypatch):
    """The notebook's units end into the one around them: read from a copy
    too, while a callback may still write into the one that ended."""
    seen: list[Unit] = []
    monkeypatch.setattr(Unit, "absorb", lambda self, inner: seen.append(inner))
    recorder = code_trace.recorder_or_reason()
    assert recorder.tool is not None, recorder.reason
    outer = recorder.begin("stmt")
    inner = recorder.begin("call")
    recorder.end(inner)
    recorder.end(outer)
    assert seen and seen[0] is not inner and seen[0].kind == "call"


def _a_helper():
    return None
