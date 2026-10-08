"""Sentinels held inside a cached call's result, put back as the notebook's
own objects.

``MISSING = object()`` in a cell, and ``res = lookup(keys)`` where the
call to ``lookup`` is cached on its own: a hit's result is a copy, and the
copy of ``MISSING`` in it was another object, so ``v is MISSING`` was False
on every hit. A Run All also defines a new ``MISSING``, which a call that
runs again would hold. The call entry records where the sentinels the
function names sit in its result (`cash.identity_refs`), and a hit puts the
object each name holds now in their place.

A statement's own entry needs none of this: a variable that holds one of
its outputs' objects is stored and restored with them (``holders``).

Only a bare ``object()``: an instance of a class of the notebook is a value
the notebook tracks by its lineage, restored as the copy it was stored as.
"""

from __future__ import annotations

from typing import Any

from cash import identity_refs

__all__ = ["find_for_call", "put_back_for_call"]


def _bare(value: Any) -> bool:
    return type(value) is object


def find_for_call(fn: Any, result: Any) -> list | None:
    """The entries for a call entry's *result*: where the bare ``object()``
    sentinels *fn*'s body names sit in it."""
    named = identity_refs.named_objects(fn, _bare)
    return identity_refs.find(result, named) if named else None


def put_back_for_call(fn: Any, value: Any, held: list | None) -> Any:
    """A call entry hit's *value* with *fn*'s sentinels put back."""
    if not held:
        return value
    return identity_refs.put_back(value, held, identity_refs.resolver(fn))
