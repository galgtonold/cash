"""``gc.get_referents``, also for an object's ``__dict__`` on Pythons that hide its values.

There an object keeps its attributes inline, and its ``__dict__``, once
read, points at those values without reporting them: ``gc.get_referents(
p.__dict__)`` gave ``[]`` where ``p.child`` was in it. The walks below read
an object's state through its ``__dict__`` (`notebook.statement.in_place`),
so a weak reference the notebook held to ``p.child`` went unseen, a loop
restored ``p`` into a copy of its child, and ``held() is p.child`` gave
False where plain Python gives True. Python 3.14.4 reports them again;
3.13.12 hides them too, so the Python is asked, not its version number.

`get_referents` is ``gc.get_referents`` itself on every other Python; on
those, a dict that reports nothing while it holds items is read
through ``dict.values``, which is what a dict of string keys reports.
"""

from __future__ import annotations

import gc
from typing import Any


def _inline_values_hidden() -> bool:
    """Whether this Python's object ``__dict__`` hides its inline values
    from the garbage collector's referents: asked of a fresh object, not of
    the version number, since 3.13.12 does it as well as 3.14.0-3.14.3."""

    class _Probe:
        def __init__(self) -> None:
            self.held = held

    held = object()
    probe = _Probe()
    return not any(item is held for item in gc.get_referents(probe.__dict__))


#: Whether this Python's object ``__dict__`` hides its inline values from
#: the garbage collector's referents.
INLINE_VALUES_HIDDEN = _inline_values_hidden()


def _get_referents(*objs: Any) -> list[Any]:
    """``gc.get_referents(*objs)``, with a dict that reports nothing while
    it holds items read through ``dict.values``."""
    found = gc.get_referents(*objs)
    # An object's ``__dict__`` is a plain dict; a subclass's ``__len__``
    # is its own code, never run here.
    if not any(type(obj) is dict and obj for obj in objs):
        return found
    found = []
    for obj in objs:
        held = gc.get_referents(obj)
        if not held and type(obj) is dict and obj:
            held = list(dict.values(obj))
        found.extend(held)
    return found


get_referents = _get_referents if INLINE_VALUES_HIDDEN else gc.get_referents
