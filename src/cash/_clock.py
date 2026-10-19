"""The clock cash times itself with, out of reach of clock test doubles.

freezegun replaces ``time.perf_counter`` -- in the ``time`` module and in every
module attribute that holds it -- with a function that returns the frozen
instant. Cash measures bodies with it, so under ``freeze_time`` every body
"ran 0.00s", fell under the persistence floor, and nothing computed under a
fake clock was ever stored. Cash's cost model is about how long
work really took, never about the time the program is pretending it is.

The real function is kept inside a tuple: freezegun swaps module attributes
that ARE the real function, not what a container holds. If cash itself is
imported under a frozen clock, freezegun's own saved original is used.
"""

from __future__ import annotations

import sys
import threading
import time
import types


def _real_perf_counter():
    current = time.perf_counter
    if isinstance(current, types.BuiltinFunctionType):
        return current
    saved = getattr(sys.modules.get("freezegun.api"), "real_perf_counter", None)
    return saved if callable(saved) else current


_REAL = (_real_perf_counter(),)


def perf_counter() -> float:
    """``time.perf_counter()``, whatever a test has done to it."""
    return _REAL[0]()


#: This thread's CPU time (`CpuLimit`); a name of its own, so a test can
#: stand in a clock for it.
_thread_time = time.thread_time

#: How long in wall time a `CpuLimit` may run, in multiples of its CPU
#: time: a limit kept waiting by a busy machine still ends.
WALL_STRETCH = 10.0


class CpuLimit:
    """A limit of *seconds* of this thread's CPU time on a check.

    The share check, a statement's analysis and a change check's digest are
    limited because a check that costs more than the statement it serves
    never pays. Timed by the wall clock, a busy machine charged them for the
    time other processes ran, and a check of 0.05 s ran past its 0.1 s and
    gave up: the statement ran uncached that run. What the check costs this
    thread is its CPU time, so that is what is counted. ``thread_time``
    costs ~0.6 us a read against ~0.1 us for ``perf_counter``, and the
    walks check at every object, so the wall clock is read first: only
    when it passes `deadline` is the CPU time read (`passed`), and the
    deadline moves on by the CPU time still left. On an idle machine the
    two agree and a runaway check stops when it did; on a busy one it runs
    on, up to `WALL_STRETCH` times *seconds* of wall time. Asked from
    another thread, whose CPU time says nothing of this one's, the wall
    clock answers alone.
    """

    __slots__ = ("cpu_started", "deadline", "hard", "seconds", "started", "thread")

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.started = perf_counter()
        self.cpu_started = _thread_time()
        self.thread = threading.get_ident()
        self.deadline = self.started + seconds
        self.hard = self.started + seconds * WALL_STRETCH

    def spent(self) -> float:
        """The CPU seconds this thread spent since the limit started."""
        if threading.get_ident() != self.thread:
            return perf_counter() - self.started
        return _thread_time() - self.cpu_started

    def passed(self) -> bool:
        """Is the limit spent? Asked once the wall clock is past `deadline`;
        when it is not, `deadline` moves on to when it could be."""
        now = perf_counter()
        if now <= self.deadline:
            return False
        if threading.get_ident() != self.thread:
            return True
        left = self.seconds - (_thread_time() - self.cpu_started)
        if left <= 0 or now >= self.hard:
            return True
        self.deadline = min(now + left, self.hard)
        return False
