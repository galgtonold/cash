"""A value holding the process's output stream is sized without the stream.

``pbar = pyprind.ProgBar(n, stream=sys.stdout)`` holds the kernel's output
stream, which reaches the kernel, its session and every message queue. A
store keeps the bar's stream as the name ``sys.stdout`` (`live_state`), not
as a copy, but the RAM tier's size walk counted all of it as the bar's own:
a million objects walked for every store of a loop's variables, in a loop
reading 50,000 files.

What must hold: the stream is left out only while it is the process's own
``sys.stdout``; the same object bound nowhere is counted through.
"""

from __future__ import annotations

import io
import sys

from cash import sizing
from cash.sizing import memory_footprint


class OutStream(io.TextIOBase):
    """Stands in for ipykernel's stream (`live_state._own_stream` knows it
    by its class's name and module)."""

    __module__ = "ipykernel.iostream"

    def __init__(self):
        super().__init__()
        # What the kernel's stream reaches: many objects of their own.
        self.session = [object() for _ in range(50_000)]


class Bar:
    def __init__(self, stream):
        self.stream = stream
        self.count = 0


def _walked(value) -> int:
    calls = 0
    real = sizing._own_footprint

    def counting(obj, seen, stack):
        nonlocal calls
        calls += 1
        return real(obj, seen, stack)

    sizing._own_footprint = counting
    try:
        memory_footprint(value)
    finally:
        sizing._own_footprint = real
    return calls


def test_the_process_stream_a_value_holds_is_not_walked(monkeypatch):
    stream = OutStream()
    monkeypatch.setattr(sys, "stdout", stream)
    bar = Bar(stream)
    assert _walked(bar) < 100
    assert memory_footprint(bar) < 10_000


def test_the_same_object_bound_nowhere_is_counted_through():
    stream = OutStream()
    bar = Bar(stream)
    assert sys.stdout is not stream
    assert _walked(bar) > 50_000, "control: an object that is no process stream is walked"
    assert memory_footprint(bar) > 50_000 * 16
