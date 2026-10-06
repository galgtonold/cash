"""The variables a statement's entry stores with its outputs, by where they
hold them.

A variable that holds an object of a statement's outputs too -- ``data`` in
``train = data['train']`` then ``train['f1'] = f(train)`` -- is restored with
them, so that it holds the restored objects again (`share_group`). Stored
whole, every entry carried everything else the variable holds: the 80 MB
frame next to ``train`` in ``data`` was written again for each feature
statement.

A hit needs less. The holder is current (its lineage is the one the entry
recorded), so all it lacks is the restored objects in the places it held the
old ones. `HolderPatch` records those places, and `apply_patch` puts the
restored objects there, in the live holder, which keeps its own identity as
it does when the statement runs. A holder holding an object where no patch
can put it back (in a tuple, a set, under a dict key that is not a plain
value) is stored whole, as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .shared_objects import _VALUE_TYPES, _attributes_of, _children, _is_value, _library_value_types

__all__ = ["HolderPatch", "apply_patch", "holder_patches"]

#: One step from a container to what it holds: ``("key", k)`` for a dict
#: value, ``("item", i)`` for a list or tuple item, ``("attr", name)`` for an
#: attribute of an object `_attributes_of` opens.
Step = tuple[str, Any]
Path = tuple[Step, ...]


@dataclass(frozen=True)
class HolderPatch:
    """Where a holder variable holds objects of the outputs: each place is
    ``(holder_path, output, output_path)``. An empty holder path is the
    variable itself."""

    places: tuple[tuple[Path, str, Path], ...]


def _steps(value: Any, value_types: tuple[type, ...]) -> tuple[list[tuple[Step, Any]], list[Any]]:
    """``(named, unnamed)``: what *value* holds that a step can name, and
    what it holds that none can (a set's items, a dict's keys and the values
    under a key that is not a plain value, an object's insides no walk
    opens)."""
    if isinstance(value, dict):
        named, unnamed = [], []
        for k, v in value.items():
            if _is_value(k, value_types):
                named.append((("key", k), v))
            else:
                unnamed += [k, v]
        return named, unnamed
    if isinstance(value, (list, tuple)):
        return [(("item", i), v) for i, v in enumerate(value)], []
    attrs = _attributes_of(value)
    if attrs is not None:
        return [(("attr", k), v) for k, v in attrs.items()], []
    return [], list(_children(value) or ())


def _reachable(roots: list[Any], value_types: tuple[type, ...]) -> set[int]:
    """The ids of everything a walk reaches from *roots*."""
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop()
        if _is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        stack.extend(_children(obj) or ())
    return seen


def _paths(outputs: Mapping[str, Any], value_types: tuple[type, ...]) -> dict[int, tuple[str, Path]]:
    """A path from an output to each object of the outputs a step can name."""
    found: dict[int, tuple[str, Path]] = {}
    for name, root in outputs.items():
        if _is_value(root, value_types) or id(root) in found:
            continue
        found[id(root)] = (name, ())
        stack: list[tuple[Any, Path]] = [(root, ())]
        while stack:
            obj, path = stack.pop()
            for step, child in _steps(obj, value_types)[0]:
                if _is_value(child, value_types) or id(child) in found:
                    continue
                found[id(child)] = (name, (*path, step))
                stack.append((child, (*path, step)))
    return found


def holder_patches(holders: Mapping[str, Any], outputs: Mapping[str, Any]) -> dict[str, HolderPatch] | None:
    """A `HolderPatch` for each of *holders*, or None when one of them holds
    an object of the outputs where a patch cannot put it back."""
    value_types = _VALUE_TYPES + _library_value_types()
    group = _paths(outputs, value_types)
    targets = _reachable(list(outputs.values()), value_types)
    patches: dict[str, HolderPatch] = {}
    for name, value in holders.items():
        places = _places(value, targets, value_types)
        if not places or any(key not in group for _path, key in places):
            return None
        patches[name] = HolderPatch(tuple((path, *group[key]) for path, key in places))
    return patches


def _places(value: Any, targets: set[int], value_types: tuple[type, ...]) -> list[tuple[Path, int]] | None:
    """``(path, id)`` for each place under *value* that holds one of
    *targets*; None when one is a place no patch can set."""
    places: list[tuple[Path, int]] = []
    seen: set[int] = set()
    stack: list[tuple[Any, Path, bool]] = [(value, (), False)]
    while stack:
        obj, at, in_tuple = stack.pop()
        if id(obj) in targets:
            if in_tuple:
                # A tuple's items cannot be set.
                return None
            places.append((at, id(obj)))
            continue
        if _is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        named, unnamed = _steps(obj, value_types)
        if unnamed and _reachable(unnamed, value_types) & targets:
            return None
        stack.extend((child, (*at, step), isinstance(obj, tuple)) for step, child in named)
    return places


def _get(obj: Any, step: Step) -> Any:
    kind, label = step
    return getattr(obj, label) if kind == "attr" else obj[label]


def apply_patch(live: Any, patch: HolderPatch, restored: Mapping[str, Any]) -> Any:
    """Put the restored objects into *live*, the holder as the namespace
    has it, where *patch* says it held them; the holder's value afterwards
    (*live*, or a restored object for the variable itself).

    Every place is found before any is set, so a holder that no longer has
    one (KeyError) is left as it was."""
    sets = []
    for path, output, out_path in patch.places:
        try:
            target = restored[output]
            for step in out_path:
                target = _get(target, step)
            if not path:
                live = target
                continue
            container = live
            for step in path[:-1]:
                container = _get(container, step)
        except (IndexError, AttributeError, TypeError) as e:
            raise KeyError(f"the holder has no place {path!r} any more") from e
        sets.append((container, path[-1], target))
    for container, (kind, label), target in sets:
        if kind == "attr":
            setattr(container, label, target)
        else:
            container[label] = target
    return live
