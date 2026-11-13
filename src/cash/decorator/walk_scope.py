"""Which parts of one key build's walk can be remembered for the rest of it.

The key build walks the code surface of what a call reaches (`CodeSurface`),
and the state of each cached function that code reaches
(`KeyBuilder.callee_state`). Both end a cycle with a marker naming what is
already being walked: a back-reference, ``cycle:``. A digest that holds such
a marker depends on where the walk entered the cycle, so it cannot simply be
remembered: serving it to a walk entered elsewhere would make the key depend
on which object was keyed first.

A cached function's state is built in a walk of its own (`isolated`): the
code surfaces and the globals folds running around it are not its frames
(`walk_local`), so what it folds
depends only on which cached functions are being built around it. Each
remembered digest carries the cached functions its walk reached (`touched`),
and is served only while none of them is being built.

The code surfaces of one walk are remembered as Tarjan's strongly connected
components walk finds them:

* a frame whose walk met no back-reference to a frame below it is the same
  wherever the walk enters, cycles that close on itself included;
* a frame whose walk met one is a member of that cycle: not remembered, and
  it and the members it walked through are added to the members of the frame
  below. A remembered digest is served only while none of its members is
  being walked: entering the cycle from one of them would fold a
  back-reference to it instead.

Nothing is left out of a key by this: a served digest is the one walking
again would compute.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any, NamedTuple

__all__ = ["Record", "close_on", "enter", "isolated", "leave", "note_callees", "servable", "walk_local"]

_LOCAL = threading.local()


class Record(NamedTuple):
    """What a remembered digest may be served without."""

    #: The surfaces on a cycle through it, none of which may be walked.
    members: frozenset[Hashable]
    #: The cached functions its walk reached, none of which may be built.
    touched: frozenset[str]


class _Frame:
    __slots__ = ("item", "low", "members", "touched")

    def __init__(self, item: Hashable) -> None:
        self.item = item
        #: The lowest frame a back-reference in this frame's walk named.
        self.low = sys.maxsize
        #: The cycle members this frame's walk went through.
        self.members: set[Hashable] = set()
        #: The cached functions this frame's walk reached.
        self.touched: set[str] = set()


def _frames() -> list[_Frame]:
    frames = getattr(_LOCAL, "frames", None)
    if frames is None:
        frames = _LOCAL.frames = []
    return frames


def walk_local(owner: object, factory: Callable[[], Any]) -> Any:
    """*owner*'s record of what it is walking on this thread (a stack, a
    set), for this walk: a cached function's state built `isolated` starts
    with a fresh one."""
    walk = getattr(_LOCAL, "walk", None)
    if walk is None:
        walk = _LOCAL.walk = {}
    entry = walk.get(id(owner))
    if entry is None or entry[0]() is not owner:
        entry = walk[id(owner)] = (weakref.ref(owner), factory())
    return entry[1]


def _building() -> list[str]:
    building = getattr(_LOCAL, "building", None)
    if building is None:
        building = _LOCAL.building = []
    return building


def enter(item: Hashable) -> _Frame:
    """Push a frame for *item* (an object's surface, a cached function's
    state) being walked now."""
    frame = _Frame(item)
    _frames().append(frame)
    return frame


def leave(frame: _Frame) -> Record | None:
    """Pop *frame*: what its digest may be served without, or None when the
    digest depends on where the walk entered and must not be remembered."""
    frames = _frames()
    for at in range(len(frames) - 1, -1, -1):
        if frames[at] is frame:
            del frames[at:]  # with any frame an exception left above it
            break
    else:
        return None
    here = len(frames)
    if frames:
        frames[-1].touched |= frame.touched
    if frame.low < here:
        below = frames[-1]
        below.low = min(below.low, frame.low)
        below.members |= frame.members
        below.members.add(frame.item)
        return None
    return Record(frozenset(frame.members), frozenset(frame.touched))


def close_on(item: Hashable) -> None:
    """A walk met a back-reference to *item*, a frame further down."""
    frames = _frames()
    for depth, frame in enumerate(frames):
        if frame.item == item:
            top = frames[-1]
            top.low = min(top.low, depth)
            return


def note_callees(names: Iterable[str]) -> None:
    """The walk of the frame on top reached these cached functions."""
    frames = _frames()
    if frames:
        frames[-1].touched.update(names)


def servable(record: Record) -> bool:
    """True when a digest remembered with *record* (`leave`) is the one
    walking again here would compute."""
    if record.touched and not record.touched.isdisjoint(_building()):
        return False
    if not record.members:
        return True
    return not any(frame.item in record.members for frame in _frames())


@contextlib.contextmanager
def isolated(item: Hashable, name: str) -> Iterator[_Frame]:
    """Build cached function *name*'s state in a walk of its own: the
    frames around it are set aside, and its frame (*item*) collects the
    cached functions it reaches."""
    saved = getattr(_LOCAL, "frames", None)
    saved_walk = getattr(_LOCAL, "walk", None)
    _LOCAL.frames = []
    _LOCAL.walk = {}
    building = _building()
    building.append(name)
    frame = enter(item)
    try:
        yield frame
    finally:
        building.pop()
        _LOCAL.frames = saved
        _LOCAL.walk = saved_walk
