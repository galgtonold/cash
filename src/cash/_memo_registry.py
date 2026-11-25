"""The list of cash's process-wide memos, and the one reset that clears them.

A memo kept at module level outlives the session that filled it. Each one
says here how long what it holds stays true (its *scope*), so that
``cash.reset_session()`` clears every one whose facts belong to the session
it ends, without knowing their names, and a reader has one place to check
a lifetime.

The scopes, shortest first:

``statement``
    True while one statement is processed. Its module clears it then.
``cell``
    True while one cell runs. Its module clears it when the cell ends.
``session``
    True for one notebook session: keyed by lineages, cache keys or what a
    session was told once. A new session starts without it.
``process``
    True of the code, types and live objects of the process, whichever
    session asks. Never cleared by a reset.
``block``
    Not a memo: the state of a ``with`` block in progress (a stack, a
    current value). Its block empties it; a reset must not.

`reset` clears a scope and every shorter one. Dropping an entry of a memo
registered with a ``clear`` must always be allowed: it may cost a
recomputation or a cache miss, never a wrong result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

__all__ = ["CLEARED_SCOPES", "SCOPES", "register_memo", "registered", "reset"]

logger = logging.getLogger(__name__)

#: The scopes a reset clears, shortest first.
CLEARED_SCOPES = ("statement", "cell", "session")
#: Every scope a memo may have.
SCOPES = (*CLEARED_SCOPES, "process", "block")

#: ``name -> (scope, clear)``; *clear* is None for a scope no reset clears.
_MEMOS: dict[str, tuple[str, Callable[[], None] | None]] = {}


def register_memo(name: str, clear: Callable[[], None] | None, scope: str) -> None:
    """List the memo *name* (``module.attribute``) with its *scope*. *clear*
    empties it; required for the scopes a reset clears, unused for the
    others. Registering a name again replaces its entry."""
    if scope not in SCOPES:
        raise ValueError(f"unknown memo scope {scope!r} for {name}: one of {', '.join(SCOPES)}")
    if scope in CLEARED_SCOPES and clear is None:
        raise ValueError(f"the {scope} memo {name} needs a clear")
    _MEMOS[name] = (scope, clear if scope in CLEARED_SCOPES else None)


def registered() -> dict[str, str]:
    """``name -> scope`` of every memo registered."""
    return {name: scope for name, (scope, _clear) in _MEMOS.items()}


def reset(scope: str = "session") -> list[str]:
    """Clear every memo of *scope* and of the shorter scopes; returns their
    names. A clear that raises is logged and the rest still run: a reset
    half done would leave some memos of the old session behind."""
    if scope not in CLEARED_SCOPES:
        raise ValueError(f"a reset clears one of {', '.join(CLEARED_SCOPES)}, not {scope!r}")
    upto = CLEARED_SCOPES[: CLEARED_SCOPES.index(scope) + 1]
    cleared = []
    for name, (its_scope, clear) in list(_MEMOS.items()):
        if its_scope not in upto or clear is None:
            continue
        try:
            clear()
        except Exception:  # noqa: BLE001 - one memo's clear must not keep the others
            logger.warning("could not clear the memo %s", name, exc_info=True)
            continue
        cleared.append(name)
    return cleared
