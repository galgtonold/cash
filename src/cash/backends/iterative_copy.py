"""A deep copy that does not recurse: `deep_copy`.

``copy.deepcopy`` calls itself a few times per level of the value, and
pickle's C pickler once or twice: a linked list of a few thousand nodes
(a parsed tree, a chain of ``Node(v, next)``) ran out of recursion before it
was copied, and on Windows, whose threads get a smaller C stack, a deep
enough walk took the process down instead. This copy keeps its own stack on
the heap: each container or object being copied is a generator that yields
the parts it needs copied and is sent their copies back, so the depth of
the value never reaches the C stack.

What it copies, and how, follows a pickle round trip (the copy the RAM tier
makes): an object is rebuilt from its reduce (``kept_state``'s, which keeps
the instance attributes a C base's reduce drops), memoised before its state
is copied so cycles come back as cycles, and classes, functions and modules
are shared as pickle shares them, by name. A class's own ``__deepcopy__`` is
honoured, as ``deepcopy`` honours it.
"""

from __future__ import annotations

import pickle
import types
import weakref
from collections.abc import Callable, Generator
from typing import Any

from .. import kept_state

__all__ = ["deep_copy"]

#: Shared, never copied: immutable leaves, and what pickle and ``deepcopy``
#: both hand back as the same object (code is looked up by name).
_SHARED: frozenset[type] = frozenset(
    {
        type(None),
        type(Ellipsis),
        type(NotImplemented),
        int,
        float,
        bool,
        complex,
        str,
        bytes,
        range,
        types.CodeType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.ModuleType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        weakref.ref,
        property,
    }
)

_Copier = Generator[Any, Any, Any]


def deep_copy(value: Any, memo: dict[int, Any], special: Callable[[Any], bool] | None = None) -> Any:
    """A deep copy of *value*, of any depth, on a stack of constant size.

    *memo* is ``id(original) -> copy``, as ``deepcopy``'s: what it holds is
    used as it is, and every copy made is added. *special*, when given, is
    asked first about every object met: True means it has put the object's
    copy in *memo* itself (a frame or an array, copied the RAM tier's way).

    Raises what the reduce of an object that cannot be copied raises
    (``TypeError`` for a lock, a generator, an open socket), and
    ``pickle.PicklingError`` for an object rebuilt from arguments that hold
    the object itself, which deepcopy and pickle cannot copy either.
    """
    alive: list[Any] = []  # every original copied, so no id in *memo* is reused
    reducers = kept_state.reducers(pickle.HIGHEST_PROTOCOL)
    making: set[int] = set()  # objects whose reduce arguments are being copied

    def start(obj: Any) -> tuple[bool, Any]:
        """``(True, copy)`` when *obj* is copied at once, else ``(False, a
        generator that copies it)``."""
        obj_type = type(obj)
        if obj_type in _SHARED or issubclass(obj_type, type):
            return True, obj
        key = id(obj)
        if key in memo:
            return True, memo[key]
        if special is not None and special(obj):
            return True, memo[key]
        alive.append(obj)
        if obj_type is list:
            return False, _list(obj)
        if obj_type is dict:
            return False, _dict(obj)
        if obj_type is tuple:
            return False, _tuple(obj)
        own = getattr(obj, "__deepcopy__", None)
        if own is not None:
            copied = own(memo)
            memo[key] = copied
            return True, copied
        if key in making:
            raise pickle.PicklingError(f"a {obj_type.__qualname__} is rebuilt from arguments that hold it")
        return False, _reduced(obj, reducers[obj_type](obj))

    def _list(obj: list) -> _Copier:
        copied: list = []
        memo[id(obj)] = copied
        for item in obj:
            copied.append((yield item))
        return copied

    def _dict(obj: dict) -> _Copier:
        copied: dict = {}
        memo[id(obj)] = copied
        for k, v in obj.items():
            new_key = yield k
            copied[new_key] = yield v
        return copied

    def _tuple(obj: tuple) -> _Copier:
        items = []
        for item in obj:
            items.append((yield item))
        if id(obj) in memo:  # a cycle through the tuple was copied meanwhile
            return memo[id(obj)]
        copied = obj if all(a is b for a, b in zip(items, obj)) else tuple(items)
        memo[id(obj)] = copied
        return copied

    def _reduced(obj: Any, reduced: Any) -> _Copier:
        if isinstance(reduced, str):
            return obj  # a global, by name: pickle stores the name
        rebuild, args, *rest = reduced
        rest += [None] * (4 - len(rest))
        state, listitems, dictitems, setter = rest[:4]
        making.add(id(obj))
        args = yield args
        making.discard(id(obj))
        copied = rebuild(*args)
        memo[id(obj)] = copied
        if state is not None:
            state = yield state
            if setter is not None:
                setter(copied, state)
            else:
                _set_state(copied, state)
        if listitems is not None:
            for item in listitems:
                copied.append((yield item))
        if dictitems is not None:
            for k, v in dictitems:
                new_key = yield k
                copied[new_key] = yield v
        return copied

    done, result = start(value)
    if done:
        return result
    stack: list[_Copier] = [result]
    sent: Any = None
    while stack:
        try:
            part = stack[-1].send(sent)
        except StopIteration as finished:
            stack.pop()
            sent = finished.value
            continue
        done, result = start(part)
        if done:
            sent = result
        else:
            stack.append(result)
            sent = None
    return sent


def _set_state(obj: Any, state: Any) -> None:
    """Put *state* into *obj*, as unpickling it does (``BUILD``)."""
    setstate = getattr(obj, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    slots = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slots = state
    if state:
        obj.__dict__.update(state)
    if slots:
        for name, value in slots.items():
            setattr(obj, name, value)
