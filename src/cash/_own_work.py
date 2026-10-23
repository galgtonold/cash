"""Code of the user's that cash itself runs: pickling a value to store or
digest it calls the value's ``__reduce_ex__`` and ``__getstate__``.

The user's program does not run that code; plain Python would not. A unit
recording what the statement ran (`cash.decorator.code_trace`) leaves it
out while a thread is inside `own_work`, and arms what it left out again
afterwards, so a later call of the same code by the user's own code is seen.
"""

from __future__ import annotations

import threading
import types
from collections.abc import Callable

__all__ = ["inside", "left_out", "own_work", "rearm_with"]

_STATE = threading.local()
#: Called with the codes left out once a thread leaves its outermost
#: `own_work` (`Recorder` arms their events again).
_REARM: list[Callable[[set[types.CodeType]], None]] = []


class _OwnWork:
    __slots__ = ()

    def __enter__(self) -> None:
        _STATE.depth = getattr(_STATE, "depth", 0) + 1

    def __exit__(self, *exc: object) -> None:
        depth = _STATE.depth - 1
        _STATE.depth = depth
        if depth:
            return
        codes = getattr(_STATE, "left_out", None)
        if codes:
            _STATE.left_out = None
            for rearm in _REARM:
                rearm(codes)


#: ``with own_work:`` around cash's own pickling of a value.
own_work = _OwnWork()


def inside() -> bool:
    """Is this thread inside cash's own work on a value now?"""
    return getattr(_STATE, "depth", 0) > 0


def left_out(code: types.CodeType) -> None:
    """*code* started inside `own_work` and was not recorded."""
    codes = getattr(_STATE, "left_out", None)
    if codes is None:
        codes = _STATE.left_out = set()
    codes.add(code)


def rearm_with(callback: Callable[[set[types.CodeType]], None]) -> None:
    if callback not in _REARM:
        _REARM.append(callback)
