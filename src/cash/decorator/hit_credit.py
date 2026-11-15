"""How much time a hit is credited with saving (``total_time_saved``).

A hit is credited with what the body would cost if it ran now, and the
stored time of the miss that wrote the entry is not always that. The first
call of a function in a process pays work no later call pays again: a numba
function compiles, a lazy import loads, a cache fills. An entry written by
that call stores the warm-up with the body, and crediting every hit with
the whole of it reported 1912 s saved over a walk whose uncached run took
909 s.

The rule, per cached function and per process:

* An entry written by a call that was not its function's first in its
  process holds a warm time, and a hit is credited with it.
* An entry written by a first call (or by a version of cash that did not
  record which) may hold a warm-up. Its hit is credited with the stored time
  only when nothing in this process has paid that warm-up yet: no body of
  the function ran here and no earlier hit was credited with it. That hit
  did stand in for the warm-up, once.
* Every later hit of such an entry is credited with the smallest warm time
  seen for the function in this process (a warm call, or a hit of a warm
  entry), and never more than the stored time. Before any warm time has
  been seen there is nothing smaller to go by, and it keeps the stored time.

So the total never counts a warm-up more than once per function per
process once a warm call has been seen; it can understate a function whose
arguments differ widely in cost, which is the side to err on.
"""

from __future__ import annotations

import threading
from typing import Any

#: Functions tracked at most; past it a new function's hits keep their
#: stored time, as before.
MAX_TRACKED = 4096


class _Seen:
    __slots__ = ("ran", "warm_up_credited", "warm_min")

    def __init__(self) -> None:
        #: A body of the function ran in this process.
        self.ran = False
        #: A hit of a first-call entry was credited with its whole time.
        self.warm_up_credited = False
        #: The smallest warm time seen, or None.
        self.warm_min: float | None = None


def stored_seconds(metadata: Any) -> float:
    """The time the miss that wrote the entry recorded for a hit to stand in for."""
    saves = metadata.saves_seconds
    return (saves if saves is not None else metadata.execution_time) or 0.0


class HitCredit:
    """This process's view of each cached function's warm-up (see the module doc)."""

    def __init__(self) -> None:
        self._seen: dict[str, _Seen] = {}
        self._lock = threading.Lock()

    def _get(self, func_name: str) -> _Seen | None:
        seen = self._seen.get(func_name)
        if seen is None and len(self._seen) < MAX_TRACKED:
            seen = self._seen.setdefault(func_name, _Seen())
        return seen

    def _note_warm(self, seen: _Seen, seconds: float) -> None:
        if seen.warm_min is None or seconds < seen.warm_min:
            seen.warm_min = seconds

    def body_ran(self, func_name: str, seconds: float | None = None) -> bool:
        """A body of *func_name* ran and took *seconds* (None: it raised).
        Returns whether it was the function's first in this process, whose
        time may hold a warm-up."""
        with self._lock:
            seen = self._get(func_name)
            if seen is None:
                return True
            first = not seen.ran
            seen.ran = True
            if not first and seconds is not None:
                self._note_warm(seen, seconds)
            return first

    def credit(self, func_name: str, metadata: Any, *, steady: bool = False) -> tuple[float, float | None]:
        """What a hit of the entry *metadata* describes is credited with:
        ``(saved, body)``, the wall time it stands in for and the body time
        scaled alike (None when the entry did not record one).

        *steady*: for a hit served again later (`hit_memo`), which is never
        the one credited with a warm-up.
        """
        stored = stored_seconds(metadata)
        body = metadata.body_seconds
        with self._lock:
            seen = self._get(func_name)
            if seen is None:
                return stored, body
            if metadata.first_call is False:
                self._note_warm(seen, stored)
                return stored, body
            if not steady and not seen.ran and not seen.warm_up_credited:
                seen.warm_up_credited = True
                return stored, body
            if seen.warm_min is None or seen.warm_min >= stored:
                return stored, body
            saved = seen.warm_min
        if body is not None and stored > 0:
            body = body * (saved / stored)
        return saved, body
