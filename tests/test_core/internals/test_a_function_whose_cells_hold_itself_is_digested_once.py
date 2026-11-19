"""A function whose own cells hold it is digested once, not to the cap.

A closure that calls itself (lmfit's ``Parameters.update_constraints``
defines ``_update_param`` so) holds itself in a cell. A trace digested it
by its cells, the cell by the function, and so on until the node cap:
0.5 s, every time such a function started inside a traced call, to call it
opaque. It now ends where it meets itself, as an object holding itself
does, and the cells it started with stand for its cells as for any other.
"""

from __future__ import annotations

import sys

from cash.tracking.trace_items import call_starts, start_cells, value_digest


def _walker(step, held):
    def walk(n):
        if n == 0:
            return 0
        held.append(n)
        return step + walk(n - 1)

    return walk


def _record_start(walk):
    """Run *walk*, keeping the cells its first frame started with."""
    code = walk.__code__
    seen = {}

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is code and not seen:
            seen[code] = start_cells(frame, code)

    sys.setprofile(tracer)
    try:
        walk(3)
    finally:
        sys.setprofile(None)
    return seen


def test_it_has_a_digest_that_follows_what_it_captured():
    first = _walker(1, [])
    again = _walker(1, [])
    other = _walker(2, [])
    assert not value_digest(first).startswith("opaque:")
    assert value_digest(first) == value_digest(again) != value_digest(other)


def test_the_cells_it_started_with_stand_for_its_cells():
    held = [0]
    walk = _walker(1, held)
    before = value_digest(walk)
    starts = _record_start(walk)
    assert held == [0, 3, 2, 1]
    assert None not in starts[walk.__code__]
    with call_starts(starts):
        # As it was when it started, not as the run left what it holds.
        assert value_digest(walk) == before
    assert value_digest(walk) != before
