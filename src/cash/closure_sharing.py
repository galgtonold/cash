"""Functions in a value whose closure holds another part of it.

``return arr, lambda: arr``: a function from a module cannot be pickled, so
a stored value keeps it by reference while the rest is copied, and the
copy's ``arr`` is no longer the one the function returns. A write through a
view of one stops reaching the other, which no plain run does. Such a value
is not stored: in a notebook the statement runs again, and ``@cash.cache``
says STORE-PARTS-SHARED.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["closures_sharing"]


def _parts(item: Any) -> Iterable[Any]:
    """What `closures_sharing` looks inside *item* for: the items
    of a builtin container and the attributes of a user object -- the parts
    pickle writes with the value, in about the time it takes to."""
    kind = type(item)
    if kind in (tuple, list, set, frozenset):
        return item
    if kind is dict:
        return item.values()
    if issubclass(type(item), (type, types.ModuleType, types.FunctionType)):
        return ()
    attributes = attribute_or(item, "__dict__", None)
    return attributes.values() if type(attributes) is dict else ()


def closures_sharing(captured_vars: Mapping[str, Any]) -> list[str]:
    """The names among *captured_vars* (a statement's outputs, or one
    ``{"result": value}``) that are, or hold in a container
    or an object's attributes, a function whose closure reaches an object
    found elsewhere in the outputs (``return arr, lambda: box['a']`` with
    ``box = {'a': arr}``). Kept by reference, such a function would not
    share it with the restored copies.

    Walked whole: it looks at what the store pickles anyway, and a cap that
    let a big output use it up passed the function behind it unseen."""
    places: set[int] = set()
    functions: list[tuple[str, types.FunctionType]] = []
    seen: set[int] = set()
    for name, value in captured_vars.items():
        stack = [value]
        while stack:
            item = stack.pop()
            if type(item) in IMMUTABLE_LEAF_TYPES or id(item) in seen:
                continue
            seen.add(id(item))
            places.add(id(item))
            if type(item) is types.FunctionType:
                functions.append((name, item))
            stack.extend(_parts(item))
    found: set[str] = set()
    for name, fn in functions:
        stack = []
        for cell in fn.__closure__ or ():
            try:
                stack.append(cell.cell_contents)
            except ValueError:  # an empty cell
                continue
        reached: set[int] = set()
        while stack:
            held = stack.pop()
            if type(held) in IMMUTABLE_LEAF_TYPES or type(held) is types.FunctionType or id(held) in reached:
                continue
            if id(held) in places:
                found.add(name)
                break
            reached.add(id(held))
            stack.extend(_parts(held))
    return sorted(found)
