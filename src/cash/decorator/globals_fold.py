"""Folding the globals a function reads, and what they carry, into its key."""

from __future__ import annotations

import functools
from typing import Any

# Two fix lines are shared by more than one emit site, because more than one
# site tells the same story: a global whose value cannot be hashed is one
# problem reached through two channels (a function's own globals and a
# helper's), and a refused write is one problem whether the value went whole or
# as a chunked manifest. Sharing the text is what keeps the two halves of each
# pair from drifting into two different pieces of advice for one doc section.
UNHASHABLE_GLOBAL_FIX = (
    "register a hasher for its type with cash.register_hasher, or read the "
    "part the result actually depends on -- a URL, a connection string -- "
    "instead of the live object."
)


def reduced_state(value: Any) -> Any:
    """What ``__reduce_ex__`` says *value* was built with, or None.

    For a C callable with no ``__dict__`` -- ``operator.itemgetter("n")``
    reduces to ``(itemgetter, ("n",))`` -- that is the only place its data
    lives. None when the reduction is just a global name (``np.add``, ``len``:
    nothing carried) or the object refuses to be reduced.
    """
    try:
        reduced = value.__reduce_ex__(4)
    except Exception:  # noqa: BLE001 - not reducible: nothing to fold
        return None
    if isinstance(reduced, str) or not isinstance(reduced, tuple) or len(reduced) < 2:
        return None
    return reduced[:3]


def held_partials(value: Any) -> list[tuple[tuple, dict]]:
    """The arguments of the ``functools.partial`` objects a wrapper instance
    holds as attributes (``np.vectorize.pyfunc``), for a wrapper that is not
    itself a partial."""
    if isinstance(value, functools.partial):
        return []
    state = getattr(value, "__dict__", None)
    if not isinstance(state, dict):
        return []
    return [(p.args, dict(p.keywords)) for p in state.values() if isinstance(p, functools.partial)]


LOG_METHOD_NAMES = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
    }
)
