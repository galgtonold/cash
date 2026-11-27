"""CACHE-HIT-COPIES-HELD: a function's hits keep copying a result held elsewhere.

A hit of a result the RAM tier keeps as bytes hands back the copy the last
hit handed out when nothing outside holds it (`handed_copies`). A caller
that keeps every result it is given -- a tokenizer keeping each word trie
by its id -- leaves the tier a copy that is held, so the hit makes a new
one from it (``clone_held``): 55-65 ms and the garbage collection after it
for a 183,000-node trie, on every hit. ``@cash.cache(frozen=True)`` hands
every hit the one object, about 6x faster on that workload, and nothing
tells the user it exists.

So the tier counts, per function, its hits of such results, the ones that
copied a held result, and the seconds those copies took (`tally`), and
once both are clear -- `MIN_COPIES` copies, more than half the hits, and
`MIN_SECONDS` spent on them -- it hands the function to the `Cash` whose
function it is (`report`), which says it once (`text`). A function whose
results ``frozen=True`` would not hand back (`blocks_frozen`), or which is
frozen already, is never counted. Nothing changes until the user opts in.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from typing import Any

__all__ = ["MIN_COPIES", "MIN_SECONDS", "Tally", "blocks_frozen", "on_found", "report", "tally", "text"]

#: Hits that copied a held result before the hint is worth saying: a caller
#: holding one result once (a tokenizer built at start-up) is not a pattern.
MIN_COPIES = 5

#: Seconds those copies must have taken together. Below it the copies cost
#: less than reading the warning does; a 183,000-node trie copies in 60 ms.
MIN_SECONDS = 0.25


class Tally:
    """One function's hits of results kept as bytes in the RAM tier."""

    __slots__ = ("copies", "hits", "seconds", "told")

    def __init__(self) -> None:
        self.hits = 0
        #: Hits that copied a held result (``clone_held``), and their time.
        self.copies = 0
        self.seconds = 0.0
        #: Handed to `report` already: counted no more.
        self.told = False


def blocks_frozen(metadata: Any) -> bool:
    """Whether an entry of *metadata* is one ``frozen=True`` would not hand
    back as one object (`memory_backend._keeps_frozen`), or is frozen
    already: no hint for it."""
    return bool(
        metadata.get("frozen") or metadata.get("held_refs") or metadata.get("result_ref") or metadata.get("args_left")
    )


def tally(tallies: dict[str, Tally], name: str, copied: bool, seconds: float) -> Tally | None:
    """Count a hit of *name*'s, *copied* from a held result in *seconds*;
    the tally when the hint is due now, once. Without a lock: a lost count
    at worst delays a hint."""
    found = tallies.get(name)
    if found is None:
        found = tallies[name] = Tally()
    if found.told:
        return None
    found.hits += 1
    if not copied:
        return None
    found.copies += 1
    found.seconds += seconds
    if found.copies >= MIN_COPIES and found.copies * 2 > found.hits and found.seconds >= MIN_SECONDS:
        found.told = True
        return found
    return None


def text(name: str, found: Tally) -> tuple[str, str]:
    """``(what happened, fix)``. Keep it and
    docs/warnings.md#cache-hit-copies-held saying the same."""
    what = (
        f"@cash.cache on {name}: {found.copies} of its {found.hits} hits so far had to copy the "
        f"cached result ({found.seconds:.2f}s in all), because other code still holds the results "
        f"earlier calls returned, so cash cannot hand back the object it kept. The results are "
        f"correct either way."
    )
    fix = (
        f"if callers never change what {name} returns, declare @cash.cache(frozen=True): every hit "
        f"then hands back the same object, without copying. If something does change it after all, "
        f"cash notices (KEY-FROZEN-MUTATED) and hands out a new copy from then on."
    )
    return what, fix


_lock = threading.Lock()
#: Who says the hint for a function: each `Cash`'s hook, a bound method held
#: weakly so it keeps no `Cash` alive, answering whether the function is one
#: of its own.
_HOOKS: list[weakref.WeakMethod] = []


def on_found(hook: Callable[[str, Tally], bool]) -> None:
    """Call *hook*, a bound method, with each function whose hint is due; it
    returns whether the function is its own (and it said the hint)."""
    with _lock:
        _HOOKS[:] = [ref for ref in _HOOKS if ref() is not None]
        _HOOKS.append(weakref.WeakMethod(hook))  # type: ignore[arg-type]


def report(name: str, found: Tally) -> None:
    """Hand *name*'s hint to the hook that owns the function. A function no
    `Cash` owns (an entry stored by hand) gets none: the fix is a decorator
    argument."""
    with _lock:
        hooks = [ref() for ref in _HOOKS]
    for hook in hooks:
        if hook is None:
            continue
        try:
            if hook(name, found):
                return
        except Exception:  # noqa: BLE001 - a hint must never fail the hit
            continue
