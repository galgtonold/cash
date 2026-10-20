"""The trace recorder is ready before any of its events is on.

`code_trace._Recorder.install` called the base class's install, which turned
global PY_START events on with the subclass's `_on_start` as their callback,
before the subclass had made the `rearm` set that callback writes to. The
first cached call installs the recorder, so in a thread pool the other
workers ran that callback in the gap and raised AttributeError into whatever
code they were starting. A pool worker that died so left its task's future
unset, and the caller waited on it forever: the 180 s hang of
`test_pool_misses_after_a_clear_say_the_entries_are_gone` (about one run in 20
with fast thread switches).

The check runs in a fresh process, whose monitoring tool ids are free.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on Python 3.12+"),
]

READY = textwrap.dedent("""
    import sys
    from cash.decorator import code_trace

    mon = sys.monitoring
    r = code_trace._recorder_class()()
    seen = []
    set_events = mon.set_events

    def checked(tool, events):
        if events:
            # What a callback would find, at the moment its events go on.
            start = mon.register_callback(tool, mon.events.PY_START, None)
            mon.register_callback(tool, mon.events.PY_START, start)
            seen.append((
                getattr(start, "__name__", None),
                all(hasattr(r, name) for name in ("rearm", "user_codes", "by_string", "rearm_works")),
                getattr(r, "disc", None) == tool,
            ))
        return set_events(tool, events)

    mon.set_events = checked
    try:
        assert r.install(), r.reason
    finally:
        mon.set_events = set_events
    print(seen)
    assert seen == [("_discover", True, True)], seen
    r.uninstall()
""")


def test_every_callback_finds_the_recorder_ready_when_its_events_go_on(tmp_path):
    run_python("-c", READY, cwd=tmp_path, timeout=60)
