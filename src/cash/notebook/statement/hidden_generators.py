"""Generators a statement draws from that its key holds nowhere.

A generator bound to a variable the statement reads is followed: its draw
moves the variable's lineage (`carrier_advances`). One held deeper -- an
attribute of an object (``self.rng`` in a model built elsewhere, the
``random_state`` an estimator keeps), an item of a container, a global or
closure of a function the statement calls -- was not: the statement was
stored, and a hit served the first draw again while the generator stayed
where it was, so the next draw differed from a run from the top.

:func:`positions` walks what a statement can reach from its inputs, a few
levels deep and within a budget, and notes each generator met with its
state; :func:`moved` names those a run moved. Such a run is not stored.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash.diagnostics import warn_diagnostic
from cash.tracking.randomness import CashRandomnessWarning
from cash.tracking.randomness.state import object_carrier_state, rng_carrier_kind, same_carrier_state
from cash.tracking.trace_recorder import is_user_code

__all__ = ["moved", "positions", "warn"]

#: How deep below an input the walk goes, how many items of one container
#: or attributes of one object it looks at, and how many objects in all.
_DEPTH = 3
_WIDTH = 64
_BUDGET = 2000

_LEAVES = (str, bytes, bytearray, int, float, complex, bool, type(None), type)
#: A module of the user's is walked (``tlib.MODEL.rng``) up to this many globals.
_MODULE_WIDTH = 256


def _children(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, types.ModuleType):
        # A library's generators are its own business (`np.random`'s global
        # stream has its own watch); a module of the user's is walked.
        attrs = obj.__dict__
        if len(attrs) <= _MODULE_WIDTH and is_user_code(obj):
            yield from ((f".{k}", v) for k, v in list(attrs.items()) if not k.startswith("__"))
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        if len(obj) <= _WIDTH:
            for i, item in enumerate(obj):
                yield f"[{i}]", item
        return
    if isinstance(obj, dict):
        if len(obj) <= _WIDTH:
            for k, v in obj.items():
                yield f"[{k!r}]", v
        return
    if isinstance(obj, types.FunctionType):
        code = obj.__code__
        for name, cell in zip(code.co_freevars, obj.__closure__ or (), strict=False):
            try:
                yield name, cell.cell_contents
            except ValueError:
                continue
        g = obj.__globals__
        for name in code.co_names[:_WIDTH]:
            if name in g:
                yield name, g[name]
        return
    if isinstance(obj, types.MethodType):
        yield "__self__", obj.__self__
        yield "__func__", obj.__func__
        return
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except (AttributeError, TypeError):
        return
    if isinstance(attrs, dict) and len(attrs) <= _WIDTH:
        yield from ((f".{k}", v) for k, v in attrs.items())


def positions(roots: Mapping[str, Any], followed: Iterable[Any] = ()) -> dict[int, tuple[str, Any, Any]]:
    """``{id(generator): (path, generator, state)}`` for each generator
    reachable from *roots* (name -> value) below the top level, but not one
    of *followed* (those the variables' lineage already follows)."""
    skip = {id(obj) for obj in followed}
    found: dict[int, tuple[str, Any, Any]] = {}
    seen: set[int] = set()
    frontier = [(name, value, 0) for name, value in roots.items()]
    budget = _BUDGET
    while frontier and budget > 0:
        path, obj, depth = frontier.pop()
        if isinstance(obj, _LEAVES) or id(obj) in seen:
            continue
        seen.add(id(obj))
        budget -= 1
        if rng_carrier_kind(obj) is not None:
            if depth > 0 and id(obj) not in skip:
                state = object_carrier_state(obj)
                if state is not None:
                    found.setdefault(id(obj), (path, obj, state))
            continue
        if depth >= _DEPTH:
            continue
        try:
            for label, child in _children(obj):
                frontier.append(
                    (f"{path}{label}" if label.startswith(("[", ".")) else f"{path}.{label}", child, depth + 1)
                )
        except Exception:  # noqa: BLE001 - a probe of arbitrary objects must not break the statement
            continue
    return found


def moved(found: Mapping[int, tuple[str, Any, Any]]) -> list[str]:
    """The paths of the generators in *found* whose state moved since."""
    return [path for path, obj, before in found.values() if not same_carrier_state(before, object_carrier_state(obj))]


def warn(code: str, drawn: list[str]) -> None:
    """RANDOM-HIDDEN-GENERATOR for the statement *code*, which drew from the
    generators at *drawn*."""
    first = code.strip().splitlines()[0] if code.strip() else code
    if len(first) > 60:
        first = first[:57] + "..."
    shown = ", ".join(f"`{p}`" for p in drawn[:3])
    warn_diagnostic(
        CashRandomnessWarning,
        "RANDOM-HIDDEN-GENERATOR",
        f"`{first}` drew from {shown}, a generator held where its key does not follow it, "
        "so it runs every time: a stored result would leave the generator where it was.",
        "hold the generator in a variable and pass it in (`model.fit(X, rng=rng)`), or seed it "
        "in the cell that draws from it.",
    )
