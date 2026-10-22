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
from collections.abc import Collection, Iterable, Mapping
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

__all__ = [
    "FrameCellsPatch",
    "HolderPatch",
    "apply_patch",
    "frame_cells_patch",
    "frame_cells_patches",
    "holder_patches",
    "put_back_objects",
    "shares_categories",
]

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

    #: Never changed once made: the RAM tier keeps it as it is when storing,
    #: instead of copying it by a pickle round trip whose reading, next to a
    #: statement of a millisecond, had the copy of the whole entry taken
    #: again to time it.
    _cash_stored_as_is = True


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
        # And the cells of the outputs' frames the holders reach, which no
        # step names either: ``rect`` in ``df``'s column ``m``.
        held = _reachable_within(list(holders.values()), value_types, _HELD_MAX, left)
        if held:
            for key, place in _cell_paths(outputs, held).items():
                group.setdefault(key, place)
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


def _cell_paths(outputs: Mapping[str, Any], held: set[int]) -> dict[int, tuple[str, Path]]:
    """``(output, path)`` to each cell of the outputs' frames (themselves,
    or at their top level) whose object's id is in *held*, by address
    (`frame_cells.addresses`): ``("cell", (column, row))`` from the frame,
    column -1 for a series."""
    from .frame_cells import addresses, frames_with, loaded_pandas, object_columns

    pd = loaded_pandas()
    np = __import__("sys").modules.get("numpy")
    if pd is None or np is None or not held:
        return {}
    wanted = np.fromiter(held, dtype=np.int64, count=len(held))
    found: dict[int, tuple[str, Path]] = {}
    for name, value in outputs.items():
        kind = "key" if type(value) is dict else "item"
        for path, frame in frames_with(value, pd):
            steps = tuple((kind, step) for step in path)
            for position, array in object_columns(frame, pd):
                own = addresses(array)
                for row in np.flatnonzero(np.isin(own, wanted)).tolist():
                    found.setdefault(int(own[row]), (name, (*steps, ("cell", (position, row)))))
    return found


def _category_steps(value: Any) -> list[tuple[Step, Any]]:
    """``(("cat", (column, index)), object)`` for each object a categorical
    of the frame or series *value* holds among its categories (column -1
    for a series). A restore copies them apart from a list holding them
    too (``ks``, ``df['ck'] = pd.Categorical([ks[i % 3] ...])``): the
    holder takes the restored ones back at its own places."""
    from .frame_cells import categorical_columns, loaded_pandas
    from .shared_objects import object_categories

    pd = loaded_pandas()
    if pd is None or not isinstance(value, (pd.DataFrame, pd.Series)):
        return []
    if isinstance(value, pd.Series):
        columns = [(-1, categories) for categories in object_categories(value)]
    else:
        columns = categorical_columns(value, pd)
    steps: list[tuple[Step, Any]] = []
    for position, categories in columns:
        steps += [(("cat", (position, i)), item) for i, item in enumerate(categories.tolist())]
    return steps


def _get(obj: Any, step: Step) -> Any:
    kind, label = step
    if kind == "cell":
        from .frame_cells import column_at

        position, row = label
        return column_at(obj, position)[row]
    if kind == "cat":
        from .frame_cells import dtype_at

        position, index = label
        return dtype_at(obj, position).categories[index]
    return getattr(obj, label) if kind == "attr" else obj[label]


def shares_categories(holder: Any, outputs: Mapping[str, Any]) -> bool:
    """Whether a categorical of *holder*'s frames holds one of the objects a
    categorical of the *outputs*' frames holds. Stored whole with them, the
    two come back apart: pandas copies a categorical's categories without
    the copy's memo of what it already copied. A frame that is an output's
    own (``w = mk(df)``, ``df`` and ``w`` one frame) comes back as one."""
    from .frame_cells import frames_in
    from .shared_objects import object_categories

    def ids(frames: list[Any]) -> set[int]:
        return {id(item) for frame in frames for array in object_categories(frame) for item in array.tolist()}

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


def frame_cells_patches(holders: Mapping[str, Any], outputs: Mapping[str, Any]) -> dict[str, FrameCellsPatch]:
    """`frame_cells_patch` for each of *holders*, those it gives one, in one
    pass: the outputs' identities are read once, and the frames are read
    together once.

    Each cell of the outputs' frames needs taking back from one holder
    only: the first one, in *holders*' order, that holds it. Once those
    linked so far take back every cell of the outputs' frames, a further
    holder frame holds the outputs' cells only as those -- the live cells it
    holds too -- and is given no links of its own: ``df`` and fifty views
    or filters of it, for ``v = df.iloc[3:]``, link to ``df`` alone instead
    of reading half a million addresses per view. Whether a holder holds
    the outputs' objects otherwise is asked of each as before
    (`holds_part_of`)."""
    from .frame_cells import addresses, cell_links, frames_in, is_frame, put_back_dtypes
    from .frame_cells import object_columns as frame_object_columns
    from .shared_objects import (
        VALUE_TYPES,
        frames_together,
        identities,
        library_value_types,
        object_columns_of,
    )

    np = __import__("sys").modules.get("numpy")
    framed = {name: value for name, value in holders.items() if is_frame(value)}
    if not framed or np is None:
        return {}
    out_frames = frames_in(outputs)
    # Only frames among the outputs, at their top level: anything else they
    # hold is compared a holder at a time (`frame_cells_patch`).
    if any(not is_frame(value) for value in outputs.values()):
        return {name: patch for name in framed if (patch := frame_cells_patch(framed[name], outputs)) is not None}
    pd = __import__("sys").modules["pandas"]
    # The object columns of the outputs' frames, and those a holder takes
    # back whole (a link of every row, in order): ``df`` does for ``v =
    # df.iloc[3:]``.
    needed = {(tuple(path), position) for path, frame in out_frames for position, _ in frame_object_columns(frame, pd)}
    whole: set[tuple] = set()
    # The addresses of each output frame's cells no holder takes back yet,
    # read only when some column is not taken back whole.
    left: dict[int, Any] | None = None
    found: dict[str, list[dict]] = {}
    linked: dict[int, list[Any]] = {}
    for name, holder in framed.items():
        if needed <= whole or (left is not None and all(not len(rest) for rest in left.values())):
            found[name] = []
            continue
        if left is None and found:
            left = {
                id(frame): _not_in(
                    np.unique(np.concatenate([addresses(array) for array in arrays])), linked.get(id(frame), ()), np
                )
                for _path, frame in out_frames
                if (arrays := object_columns_of(frame))
            }
            if all(not len(rest) for rest in left.values()):
                found[name] = []
                continue
        links = cell_links(outputs, (holder,), {})
        if links is None:
            continue
        found[name] = links[0]
        whole.update(
            (tuple(link["result"]), link["column"])
            for link in links[0]
            if not link.get("dtype_of") and link["rows"] is None
        )
        for key, ids in links[1].items():
            linked.setdefault(key, []).append(ids)
            if left is not None and key in left:
                left[key] = _not_in(left[key], (ids,), np)
    if not found:
        return {}
    every = [link for links in found.values() for link in links]
    put_back = put_back_dtypes(outputs, every)
    value_types = VALUE_TYPES + library_value_types()
    joined = {key: parts[0] if len(parts) == 1 else np.unique(np.concatenate(parts)) for key, parts in linked.items()}
    frames = itertools.chain((frame for _path, frame in out_frames), (framed[name] for name in found))
    patches: dict[str, FrameCellsPatch] = {}
    with frames_together(frames, joined):
        own = set(identities(tuple(outputs.values()), value_types, True, put_back))
        for name in found:
            check_walk_budget()
            # Nor the holder's categoricals with a dtype the outputs take
            # back: its objects are the ones the restored outputs hold.
            if own and any(key in own for key in identities(framed[name], value_types, True, put_back)):
                continue
            patches[name] = FrameCellsPatch((), tuple(found[name]))
    return patches


def _not_in(ids: Any, parts: Iterable[Any], np: Any) -> Any:
    """The sorted unique *ids* that none of the sorted unique *parts* holds."""
    for part in parts:
        ids = ids[~np.isin(ids, part, assume_unique=True)]
    return ids


def put_back_objects(outputs: Mapping[str, Any], patches: Iterable[FrameCellsPatch]) -> set[int]:
    """The ids of the objects of the outputs' categoricals whose dtype
    *patches* take back from a holder frame on a hit (``dtype_of``): they
    come back as that frame's own, which ``ks`` holding them holds too, so
    nothing need put them back into it."""
    from .frame_cells import dtype_at, frames_in

    links = [link for patch in patches for link in patch.links if link.get("dtype_of")]
    if not links:
        return set()
    framed = {tuple(path): frame for path, frame in frames_in(outputs)}
    found: set[int] = set()
    for link in links:
        frame = framed.get(tuple(link["result"]))
        if frame is not None:
            found.update(map(id, dtype_at(frame, link["column"]).categories.to_numpy(copy=False).tolist()))
    return found


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
