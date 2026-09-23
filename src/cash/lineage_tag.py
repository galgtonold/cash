"""The ``_cash_lineage_hash`` tag: only ever an INSTANCE's own attribute.

The notebook's lineage store tags a value it records, and the decorator and
the loop handler trust that tag as the value's identity. A tag set on a CLASS
is inherited by every instance, so reading it with ``getattr`` made every
instance of the class the same value: ``from pathlib import Path`` in a cached
cell tagged ``Path``, and every path argument to a ``@cash.cache`` function
then keyed alike -- a call on one file was served another file's result
(round 28, found as a "flaky" test in the unit suite). Classes, modules and
functions are therefore never tagged, and a tag is only ever read from the
object's own ``__dict__``.
"""

from __future__ import annotations

import inspect
import types
from typing import Any

__all__ = ["own_tag", "taggable"]


def taggable(value: Any) -> bool:
    """Whether *value* may carry a lineage tag: not a class, module or function."""
    return not (isinstance(value, (type, types.ModuleType)) or inspect.isroutine(value))


def own_tag(value: Any, name: str = "_cash_lineage_hash") -> Any:
    """*value*'s own *name* attribute, never one inherited from its class."""
    if isinstance(value, type):
        return None
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    try:
        return own.get(name)
    except (AttributeError, TypeError):
        return None
