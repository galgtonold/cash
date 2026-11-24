"""What a missed call bound on the user's classes.

``Cfg.items = Cfg.items + [x]``, ``Cfg.count += 1``, ``setattr(Cfg, "k",
v)``, a classmethod's ``cls.items = ...``: plain Python leaves the class
holding the new binding after every call, in every process. A hit runs no
body, so it would leave the class as it found it, and a helper reading
``Cfg.items`` afterwards, or code holding the old list, would see another
program. Such a call is not stored (`CallRunner._refuse_class_writes`).

`ClassState` reads, before the body, what each user class the call can
reach binds: the classes in the module globals of the functions a call
runs (`FunctionRegistry.code_functions`), and those of its arguments.
`changed` names the attributes bound to another object, bound anew or
deleted. An immutable atom rebound to an equal one (``Cfg.n = int(Cfg.n)``)
is no change: nothing can tell the two apart. Dunder names are left out:
``copy`` and ``pickle`` cache ``__slotnames__`` on a class the first time
they meet it, which is the protocol's bookkeeping, not the user's state.

A waived stretch of the call (``with cash.assume_safe():``, a line
carrying ``# @cash:assume-safe``) is left out the way it is for the process
registries: `read` where it begins, `waive_since` where it ends
(`EffectObserver.open_waiver`).

Module globals are not compared here: a ``global N; N += 1`` counter is
the documented case of state a hit skips (IMPURE-SIDE-EFFECTS).
"""

from __future__ import annotations

import operator
import types
from typing import TYPE_CHECKING, Any

from .._type_memo import class_dict, class_mro, class_qualname
from .._user_attrs import plain_getattr
from ..value_types import IMMUTABLE_LEAF_TYPES
from .user_code import is_user_class

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ClassState", "candidates", "changed", "read", "waive_since"]

#: Past this many classes the call is not watched: reading them all on
#: every miss would cost more than the miss is worth checking for.
_MAX_CLASSES = 2048


def _members(cls: type) -> dict[str, Any] | None:
    try:
        return dict(class_dict(cls))
    except Exception:  # noqa: BLE001 - an unreadable namespace: not watched
        return None


def _dunder(name: Any) -> bool:
    return isinstance(name, str) and name.startswith("__") and name.endswith("__")


#: ``class -> is it the user's``, as `is_user_class` answered: asked of
#: every class a module's globals hold, on every miss.
_USER: dict[type, bool] = {}


def _user_classes_in(value: Any, into: dict[int, type]) -> None:
    if not isinstance(value, type):
        return
    for klass in class_mro(value):
        if id(klass) in into:
            continue
        known = _USER.get(klass)
        if known is None:
            try:
                known = bool(is_user_class(klass))
            except Exception:  # noqa: BLE001 - cannot tell: not watched
                known = False
            if len(_USER) > 4096:
                _USER.clear()
            _USER[klass] = known
        if known:
            into[id(klass)] = klass


def candidates(functions: Iterable[Any], args: tuple, kwargs: dict) -> list[type]:
    """The user classes a call of *functions* with *args* can reach by name
    or through an argument: those its functions' module globals and
    closures hold, the classes the methods among them are bound to, and the
    classes of the arguments (and the arguments that are classes), with
    their bases."""
    found: dict[int, type] = {}
    seen_globals: set[int] = set()
    for fn in functions:
        if isinstance(fn, type):
            # A class the call runs the code of: asked nothing through its
            # metaclass, as cash's other reads of a class are not.
            _user_classes_in(fn, found)
            continue
        target = plain_getattr(fn, "__func__", fn)
        g = plain_getattr(target, "__globals__", None)
        if isinstance(g, dict) and id(g) not in seen_globals:
            seen_globals.add(id(g))
            for value in list(g.values()):
                _user_classes_in(value, found)
        for cell in plain_getattr(target, "__closure__", None) or ():
            try:
                _user_classes_in(cell.cell_contents, found)
            except ValueError:  # an empty cell
                continue
        owner = plain_getattr(fn, "__self__", None)
        if owner is not None and not isinstance(owner, types.ModuleType):
            _user_classes_in(owner if isinstance(owner, type) else type(owner), found)
    for value in (*args, *kwargs.values()):
        _user_classes_in(value if isinstance(value, type) else type(value), found)
    if len(found) > _MAX_CLASSES:
        return []
    return list(found.values())


class ClassState:
    """What each watched class bound when the call began, and the
    attributes a stretch the call does not count moved on from there."""

    __slots__ = ("before", "found")

    def __init__(self, classes: list[type]) -> None:
        self.before: dict[type, dict[str, Any]] = {}
        for cls in classes:
            members = _members(cls)
            if members is not None:
                self.before[cls] = members
        #: ``Class.attr`` changed before a waived stretch moved it on.
        self.found: set[str] = set()


def read(state: ClassState) -> dict[type, dict[str, Any] | None]:
    """The watched classes as they are now, where a waived stretch begins."""
    return {cls: _members(cls) for cls in state.before}


def _moved(old: Any, new: Any) -> bool:
    if new is old:
        return False
    return not (type(new) is type(old) and isinstance(new, IMMUTABLE_LEAF_TYPES) and _equal(new, old))


def _equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b) and (a == a)  # NaN rebound to NaN: equal enough, neither compares
    except Exception:  # noqa: BLE001 - cannot compare: a change
        return False


_ABSENT = object()


def _diff(cls: type, then: dict[str, Any], now: dict[str, Any]) -> list[str]:
    # A rebinding keeps a key's place, so the same keys in the same order
    # holding the same objects is no change: the common case, compared in C.
    if len(now) == len(then) and all(map(operator.is_, now.values(), then.values())) and list(now) == list(then):
        return []
    keys = [k for k, v in now.items() if then.get(k, _ABSENT) is not v]
    keys = [k for k in keys if k not in then or _moved(then[k], now[k])]
    keys += [k for k in then if k not in now]
    keys = [k for k in keys if not _dunder(k)]
    if not keys:
        return []
    name = _label(cls)
    return [f"{name}.{k}" for k in keys]


def _label(cls: type) -> str:
    try:
        return class_qualname(cls).replace(".<locals>", "")
    except Exception:  # noqa: BLE001
        return repr(cls)


def waive_since(state: ClassState, start: dict[type, dict[str, Any] | None]) -> None:
    """Leave out of *state* what the classes changed since *start*
    (`read`): a stretch of the call the user waived. What the call changed
    before it still counts."""
    for cls, then in start.items():
        now = _members(cls)
        if then is None or now is None or not _diff(cls, then, now):
            continue
        before = state.before.get(cls)
        if before is not None:
            state.found.update(_diff(cls, before, then))
        state.before[cls] = now


def changed(state: ClassState | None) -> list[str]:
    """``Class.attr`` for each attribute of a watched class the call bound
    to another object, bound anew or deleted, sorted."""
    if state is None:
        return []
    found = set(state.found)
    for cls, then in state.before.items():
        try:
            live = class_dict(cls)
            if (
                len(live) == len(then)
                and all(map(operator.is_, live.values(), then.values()))
                and all(map(operator.is_, live, then))
            ):
                continue  # the common case, read in place
        except Exception:  # noqa: BLE001 - read again below, as a copy
            pass
        now = _members(cls)
        if now is not None:
            found.update(_diff(cls, then, now))
    return sorted(found)
