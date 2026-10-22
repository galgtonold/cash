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

import itertools
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from .. import _plain_data
from .shared_objects import (
    TREE_LEAVES,
    VALUE_TYPES,
    attributes_of,
    check_walk_budget,
    children_of,
    is_value,
    library_value_types,
)

__all__ = ["FrameCellsPatch", "HolderPatch", "apply_patch", "frame_cells_patch", "holder_patches", "shares_categories"]

#: One step from a container to what it holds: ``("key", k)`` for a dict
#: value, ``("item", i)`` for a list or tuple item, ``("attr", name)`` for an
#: attribute of an object `attributes_of` opens.
Step = tuple[str, Any]
Path = tuple[Step, ...]


@dataclass(frozen=True)
class HolderPatch:
    """Where a holder variable holds objects of the outputs: each place is
    ``(holder_path, output, output_path)``. An empty holder path is the
    variable itself."""

    places: tuple[tuple[Path, str, Path], ...]


@dataclass(frozen=True)
class FrameCellsPatch(HolderPatch):
    """A pandas frame or series holding objects of the outputs only as the
    cells of its object columns that the outputs' frames hold too:
    ``df`` for ``df2 = df[mask]``. The holder stays as it is; the restored
    outputs take its cells back where they held them
    (`frame_cells.relink_cells`, *links*, with the holder as its one
    argument)."""

    links: tuple[dict, ...] = ()


def _steps(value: Any, value_types: tuple[type, ...]) -> tuple[list[tuple[Step, Any]], list[Any]]:
    """``(named, unnamed)``: what *value* holds that a step can name, and
    what it holds that none can (a set's items, a dict's keys and the values
    under a key that is not a plain value, an object's insides no walk
    opens)."""
    if isinstance(value, dict):
        named, unnamed = [], []
        for k, v in value.items():
            if is_value(k, value_types):
                named.append((("key", k), v))
            else:
                unnamed += [k, v]
        return named, unnamed
    if isinstance(value, (list, tuple)):
        return [(("item", i), v) for i, v in enumerate(value)], []
    attrs = attributes_of(value)
    if attrs is not None:
        return [(("attr", k), v) for k, v in attrs.items()], []
    return [], list(children_of(value) or ())


def _reachable(roots: list[Any], value_types: tuple[type, ...], left: Collection[int] = ()) -> set[int]:
    """The ids of everything a walk reaches from *roots*, not entering the
    objects of *left*."""
    seen = _reachable_within(roots, value_types, None, left)
    assert seen is not None
    return seen


def _reachable_within(
    roots: list[Any], value_types: tuple[type, ...], limit: int | None, left: Collection[int] = ()
) -> set[int] | None:
    """`_reachable`, or None when that is more than *limit* objects."""
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        if not len(seen) & 1023:
            check_walk_budget()
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen or id(obj) in left:
            continue
        seen.add(id(obj))
        if limit is not None and len(seen) > limit:
            return None
        stack.extend(children_of(obj) or ())
    return seen


def _paths(outputs: Mapping[str, Any], value_types: tuple[type, ...]) -> dict[int, tuple[str, Path]]:
    """A path from an output to each object of the outputs a step can name:
    a frame's categorical's objects among them (`_category_steps`)."""
    found: dict[int, tuple[str, Path]] = {}
    for name, root in outputs.items():
        if is_value(root, value_types) or id(root) in found:
            continue
        found[id(root)] = (name, ())
        stack: list[tuple[Any, Path]] = [(root, ())]
        while stack:
            obj, path = stack.pop()
            for step, child in _steps(obj, value_types)[0] + _category_steps(obj):
                if is_value(child, value_types) or id(child) in found:
                    continue
                found[id(child)] = (name, (*path, step))
                stack.append((child, (*path, step)))
    return found


#: Most objects `holder_patches` walks in the holders to look them up in
#: outputs of JSON-like data a level at a time; past it, the outputs are
#: walked as before.
_HELD_MAX = 20_000


def holder_patches(
    holders: Mapping[str, Any], outputs: Mapping[str, Any], left: Collection[int] = ()
) -> dict[str, HolderPatch] | None:
    """A `HolderPatch` for each of *holders*, or None when one of them holds
    an object of the outputs where a patch cannot put it back.

    *left*: the ids of frames a hit leaves as they are, the outputs taking
    their cells back from them (`FrameCellsPatch`). A holder is not walked
    into them: ``frames = {'a': df}`` holds ``df2 = flt(df)``'s cells only
    as ``df``'s, and stays as it is too, an empty patch. Asked to place
    them, it was stored whole with ``df``, and the restored ``df2`` held
    copies of the cells the live ``df`` holds."""
    value_types = VALUE_TYPES + library_value_types()
    group = _tree_group(holders, outputs, value_types)
    if group is None:
        group = _paths(outputs, value_types)
        # And the categorical's objects, which no walk of a frame reads.
        targets = _reachable(list(outputs.values()), value_types) | set(group)
    else:
        targets = set(group)
    patches: dict[str, HolderPatch] = {}
    for name, value in holders.items():
        reached: list[bool] = []
        places = _places(value, targets, value_types, left, reached)
        if places is None or (not places and not reached) or any(key not in group for _path, key in places):
            return None
        patches[name] = HolderPatch(tuple((path, *group[key]) for path, key in places))
    return patches


def _tree_group(
    holders: Mapping[str, Any], outputs: Mapping[str, Any], value_types: tuple[type, ...]
) -> dict[int, tuple[str, Path]] | None:
    """`_paths` of the objects of the outputs that the holders reach, found
    a level at a time when each output is a value or JSON-like data
    (`_plain_data.tree_paths`); None otherwise, when the holders reach more
    than `_HELD_MAX` objects, or when one of those sits in more than one
    place, where which path `_paths` names depends on the order it walks in.

    `_places` asks only of objects the holders reach whether they are the
    outputs', so these are all it needs: re-sorting a list of 450,000 parsed
    pairs a loop variable still held one list of walked every pair, twice,
    for that one place.
    """
    roots = [root for root in outputs.values() if not is_value(root, value_types)]
    if not all(type(root) in (list, dict) for root in roots):
        return None
    held = _reachable_within(list(holders.values()), value_types, _HELD_MAX)
    if held is None:
        return None
    found: dict[int, tuple[str, Path]] = {}
    for name, root in outputs.items():
        if is_value(root, value_types):
            continue
        places = _plain_data.tree_paths(root, held, TREE_LEAVES)
        if places is None:
            return None
        for key, paths in places.items():
            if len(paths) > 1:
                return None
            found.setdefault(key, (name, paths[0]))
    return found


def _places(
    value: Any,
    targets: set[int],
    value_types: tuple[type, ...],
    left: Collection[int] = (),
    reached: list[bool] | None = None,
) -> list[tuple[Path, int]] | None:
    """``(path, id)`` for each place under *value* that holds one of
    *targets*; None when one is a place no patch can set. Not under a frame
    of *left*, which the hit leaves as it is; *reached* gets True when one
    is met."""
    places: list[tuple[Path, int]] = []
    seen: set[int] = set()
    stack: list[tuple[Any, Path, bool]] = [(value, (), False)]
    while stack:
        if not len(seen) & 1023:
            check_walk_budget()
        obj, at, in_tuple = stack.pop()
        if id(obj) in left:
            if reached is not None:
                reached.append(True)
            continue
        if id(obj) in targets:
            if in_tuple:
                # A tuple's items cannot be set.
                return None
            places.append((at, id(obj)))
            continue
        if is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        named, unnamed = _steps(obj, value_types)
        if unnamed and _reachable(unnamed, value_types, left) & targets:
            return None
        stack.extend((child, (*at, step), isinstance(obj, tuple)) for step, child in named)
    return places


def _category_steps(value: Any) -> list[tuple[Step, Any]]:
    """``(("cat", (column, index)), object)`` for each object a categorical
    of the frame or series *value* holds among its categories (column -1
    for a series). A restore copies them apart from a list holding them
    too (``ks``, ``df['ck'] = pd.Categorical([ks[i % 3] ...])``): the
    holder takes the restored ones back at its own places."""
    from .frame_cells import _categorical_columns, _pandas
    from .shared_objects import _object_categories

    pd = _pandas()
    if pd is None or not isinstance(value, (pd.DataFrame, pd.Series)):
        return []
    if isinstance(value, pd.Series):
        columns = [(-1, categories) for categories in _object_categories(value)]
    else:
        columns = _categorical_columns(value, pd)
    steps: list[tuple[Step, Any]] = []
    for position, categories in columns:
        steps += [(("cat", (position, i)), item) for i, item in enumerate(categories.tolist())]
    return steps


def _get(obj: Any, step: Step) -> Any:
    kind, label = step
    if kind == "cat":
        from .frame_cells import _dtype_at

        position, index = label
        return _dtype_at(obj, position).categories[index]
    return getattr(obj, label) if kind == "attr" else obj[label]


def shares_categories(holder: Any, outputs: Mapping[str, Any]) -> bool:
    """Whether a categorical of *holder*'s frames holds one of the objects a
    categorical of the *outputs*' frames holds. Stored whole with them, the
    two come back apart: pandas copies a categorical's categories without
    the copy's memo of what it already copied. A frame that is an output's
    own (``w = mk(df)``, ``df`` and ``w`` one frame) comes back as one."""
    from .frame_cells import frames_in
    from .shared_objects import _object_categories

    def ids(frames: list[Any]) -> set[int]:
        return {id(item) for frame in frames for array in _object_categories(frame) for item in array.tolist()}

    theirs = [frame for _path, frame in frames_in(tuple(outputs.values()))]
    same = {id(frame) for frame in theirs}
    mine = ids([frame for _path, frame in frames_in(holder) if id(frame) not in same])
    return bool(mine) and not mine.isdisjoint(ids(theirs))


def frame_cells_patch(holder: Any, outputs: Mapping[str, Any]) -> FrameCellsPatch | None:
    """A `FrameCellsPatch` for *holder*, when it is a frame or series that
    holds objects of *outputs* only as cells the outputs' frames hold too,
    and only it and they hold them; None otherwise."""
    from .frame_cells import cell_links, frames_in, is_frame, linked_cell_ids, put_back_dtypes
    from .shared_objects import frames_together, holds_part_of

    if not is_frame(holder):
        return None
    links = cell_links(outputs, (holder,), {})
    if links is None:
        return None
    frames = itertools.chain((frame for _path, frame in frames_in(outputs)), (holder,))
    with frames_together(frames, linked_cell_ids(links)):
        # A categorical's objects too: no cell link puts them back, only a
        # dtype one (``df.assign(...)`` keeps ``df``'s).
        if holds_part_of(tuple(outputs.values()), [holder], True, put_back_dtypes(outputs, links[0])):
            return None
    return FrameCellsPatch((), tuple(links[0]))


def apply_patch(live: Any, patch: HolderPatch, restored: Mapping[str, Any]) -> Any:
    """Put the restored objects into *live*, the holder as the namespace
    has it, where *patch* says it held them; the holder's value afterwards
    (*live*, or a restored object for the variable itself). A
    `FrameCellsPatch` leaves *live* as it is and puts its cells back in the
    restored outputs instead.

    Every place is found before any is set, so a holder that no longer has
    one (KeyError) is left as it was."""
    if isinstance(patch, FrameCellsPatch):
        from .frame_cells import RELINK_FAILED, relink_cells

        if relink_cells(restored, (live,), {}, list(patch.links)) is RELINK_FAILED:
            raise KeyError("the frame no longer holds the cells the outputs held of it")
        return live
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
