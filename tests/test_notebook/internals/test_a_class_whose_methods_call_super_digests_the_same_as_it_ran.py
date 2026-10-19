"""A class whose methods call ``super()`` digests the same in a trace as when
the trace is checked.

``super()`` gives a method a ``__class__`` cell holding its own class. Its
start digest (`start_cells`) was the class digested on its own, which never
matched the class's digest of that method, so the check said the class had
changed and every statement making an instance (``model = Net()``) ran again.
"""

from __future__ import annotations

import sys

from cash.tracking import trace_items as items


class _Base:
    def __init__(self):
        self.a = 1


_STARTS: dict = {}


def _class_and_its_start_cells():
    class Net(_Base):
        def __init__(self):
            frame = sys._getframe()
            _STARTS[frame.f_code] = items.start_cells(frame, frame.f_code)
            super().__init__()

    Net()
    return Net, dict(_STARTS)


def test_the_class_digests_the_same_with_its_start_cells():
    cls, starts = _class_and_its_start_cells()
    with items.call_starts(starts):
        as_it_ran = items.value_digest(cls)
    assert as_it_ran == items.value_digest(cls)


def test_a_cell_holding_data_still_counts_as_it_started():
    log = []

    def note():
        frame = sys._getframe()
        starts = {frame.f_code: items.start_cells(frame, frame.f_code)}
        log.append(1)
        return starts

    starts = note()
    with items.call_starts(starts):
        as_it_started = items.value_digest(note)
    assert as_it_started != items.value_digest(note)
