"""The state of objects pickled by reference that a statement's inputs hold.

A logger pickles by name (``getLogger(name)``), so the name finds it from
anywhere: ``logging.getLogger('app').setLevel(10)``, at the top level or in
a function, changes the logger that ``cfg`` holds without naming ``cfg``,
and nothing moves ``cfg``'s lineage. A reader of ``cfg`` stored before was
served the old level. The key is built before such a change is known, so
the entry keeps, beside it, the digest each such object had when the
statement ran (`states`), and a lookup serves it only while each one the
recipe finds now digests the same (`changed`), as for an environment
variable (`env_reads`).

The objects an input holds are known from its content digest
(`value_hash.held_by_reference`), taken when the variable was bound.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .. import kept_state
from ..value_hash import HASH_ERRORS, held_by_reference, known_by_reference, recipes_met

__all__ = ["changed", "for_call", "known_states", "moved", "moved_in_statement", "running", "states"]

#: The key of an object no recipe could be written for: it is not found
#: again, and the entry is never served.
UNFINDABLE = ""


def known_states() -> dict[str, str]:
    """``{recipe: digest}`` of every object pickled by reference a value of
    the user's was seen holding in this process (`known_by_reference`), as
    each is now."""
    with kept_state.one_keying():
        # What each attribute is, told once for all of them: every logger
        # holds the root logger as its parent.
        return {_key(recipe): _digest_now(recipe) for recipe in known_by_reference()}


def moved(before: Mapping[str, str]) -> list[str]:
    """The objects of *before* (`known_states`) whose state is not what it
    was then, each described."""
    found = []
    with kept_state.one_keying():
        for key, digest in before.items():
            if key == UNFINDABLE:
                continue
            recipe = base64.b64decode(key)
            if _digest_now(recipe) != digest:
                found.append(_describe_recipe(recipe))
    return found


def states(
    names: Iterable[str], session_hashes: Mapping[str, str], known: Mapping[str, str] | None = None
) -> dict[str, str] | None:
    """``{recipe: digest}`` for each object pickled by reference that a
    value of *names* holds, by the content digest *session_hashes* has for
    it; None when they hold none. A digest *known* has (`known_states`,
    taken just before) is not taken again."""
    found: dict[str, str] = {}
    for name in names:
        for recipe in held_by_reference(session_hashes.get(name)):
            key = _key(recipe)
            if key not in found:
                digest = known.get(key) if known is not None else None
                if digest is None:
                    digest = UNFINDABLE if recipe is None else _digest_now(recipe)
                found[key] = digest
    return found or None


#: The statement running now: the states of the objects known when it
#: started (`known_states`), and those it reads (`states`).
_RUNNING: contextvars.ContextVar[tuple[Mapping[str, str], Mapping[str, str]] | None] = contextvars.ContextVar(
    "cash_reference_states_running", default=None
)


@contextlib.contextmanager
def running(before: Mapping[str, str], read: Mapping[str, str] | None) -> Iterator[None]:
    """Within it, a statement runs that started with the objects *before*
    (`known_states`) and reads those of *read* (`states`): what a call it
    makes is checked against (`for_call`, `moved_in_statement`)."""
    token = _RUNNING.set((before, read or {}))
    try:
        yield
    finally:
        _RUNNING.reset(token)


def for_call(met: list) -> dict[str, str] | None:
    """``{recipe: digest}`` now for a call about to be looked up: the objects
    the statement around it reads (`running`) and those its key's digests
    met (*met*, `kept_state.met_by_reference`). None when there are none:
    nothing more to check."""
    current = _RUNNING.get()
    keys = dict.fromkeys(current[1] if current is not None else ())
    for recipe in recipes_met(met):
        keys[_key(recipe)] = None
    if not keys:
        return None
    with kept_state.one_keying():
        return {key: UNFINDABLE if key == UNFINDABLE else _digest_now(base64.b64decode(key)) for key in keys}


def moved_in_statement() -> list[str]:
    """The objects known when the statement running now started whose
    state is not what it was then (`moved`): one a call in it changed, which
    a hit of that call would not change again."""
    current = _RUNNING.get()
    return moved(current[0]) if current is not None and current[0] else []


def _key(recipe: bytes | None) -> str:
    return UNFINDABLE if recipe is None else base64.b64encode(recipe).decode("ascii")


def _describe_recipe(recipe: bytes) -> str:
    try:
        return _describe(kept_state.find_by_recipe(recipe))
    except Exception:  # noqa: BLE001 - whatever loading the recipe raises: named by its bytes
        return "an object pickled by reference"


def changed(recorded: Mapping[str, str] | None) -> str | None:
    """Why an entry that recorded *recorded* (`states`) is not served now:
    an object one of its inputs held, found again by its recipe, digests
    otherwise than when it ran, or cannot be found or read. None when all
    match."""
    for key, digest in (recorded or {}).items():
        if key == UNFINDABLE:
            return "an object it read is pickled by reference in a way cash cannot find again"
        try:
            obj = kept_state.find_by_recipe(base64.b64decode(key))
        except Exception:  # noqa: BLE001 - whatever loading the recipe raises: not found
            return "an object it read, pickled by reference, cannot be found again"
        now = _digest(obj)
        if now is None or now != digest:
            return f"{_describe(obj)}, which a value it read holds by reference, has changed"
    return None


def _digest_now(recipe: bytes) -> str:
    """The digest of the object *recipe* finds now, or `UNFINDABLE` when it
    cannot be found or read: never served."""
    try:
        obj = kept_state.find_by_recipe(recipe)
    except Exception:  # noqa: BLE001 - whatever loading the recipe raises: not found
        return UNFINDABLE
    return _digest(obj) or UNFINDABLE


def _digest(obj: Any) -> str | None:
    """The digest of *obj* with its state (`kept_state.state_digest`), or
    None when it cannot be read."""
    try:
        return kept_state.state_digest(obj)
    except (*HASH_ERRORS, MemoryError):
        return None


def _describe(obj: Any) -> str:
    text = repr(obj)
    text = text if len(text) <= 80 else f"{text[:77]}..."
    buffering = kept_state.buffering_handlers(obj)
    if buffering:
        # Each record it logs changes it: say so, or the statement seems to
        # run again for nothing.
        names = sorted(set(buffering))
        held = f"{names[0]} holds" if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]} hold"
        text += f" (its {held} each record it logs)"
    return text
