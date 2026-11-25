"""How a key tells apart instances that hold nothing.

``KEEP = object()`` and ``DROP = object()`` are flag values: code tells them
apart with ``is``, and neither holds anything a key could read. Pickled, the
two are the same bytes, so ``clean(rows, missing=DROP)`` was served what
``clean(rows)`` computed. The same holds for two instances of a marker class
whose body gives them nothing.

For such an instance, which one it is IS its value. It is keyed the way a
function or a class is: by the name that finds it, the module global or the
class attribute holding it (``mod:DROP``, ``mod:Options.UNSET``). That name
is the same in every process. One that no name holds cannot be told from the
next one, so its call is not cached (`NamelessInstanceError`).

Which instances: an exact ``object``, and an instance with no state of a
class that gives its instances no behaviour either (`_marker_class`). A
class with methods of its own is a strategy or a service, not a flag: two of
its instances with no state do the same thing, and stay keyed by their class.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any

from ._type_memo import TypeMemo, type_key
from ._user_attrs import plain_getattr

#: What a class body may hold and still be a marker's: what every class gets,
#: and how an instance prints or tests. Anything else (a method, a constant,
#: ``__eq__``, ``__call__``, how it pickles) makes it a class with behaviour.
_MARKER_CLASS_NAMES = frozenset(
    {
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__slots__",
        "__repr__",
        "__str__",
        "__bool__",
        "__firstlineno__",
        "__static_attributes__",
        "__annotations__",
        "__annotate__",
        "__annotate_func__",
        "__annotations_cache__",
    }
)

_MARKER_CLASSES: TypeMemo[bool] = TypeMemo()


def _marker_class(t: type) -> bool:
    """Do instances of *t* get nothing from their class but a name and a
    way to print? Every class under it but ``object`` must be such a class
    written in Python."""
    known = _MARKER_CLASSES.get(type_key(t))
    if known is not None:
        return known
    answer = type(t) is type and all(
        klass is object
        or (bool(klass.__flags__ & _HEAP_TYPE) and type(klass) is type and _MARKER_CLASS_NAMES.issuperset(vars(klass)))
        for klass in t.__mro__
    )
    return _MARKER_CLASSES.put(t, answer)


#: ``Py_TPFLAGS_HEAPTYPE``: the class was made by a ``class`` statement or
#: ``type()``, not in C.
_HEAP_TYPE = 1 << 9


def may_be_bare(t: type) -> bool:
    """Can an instance of *t* be one that holds nothing (`is_bare`)?
    Decided per class."""
    return t is object or _marker_class(t)


def is_bare(value: Any) -> bool:
    """Is *value* an instance that holds nothing, whose identity is all
    that tells it from another?"""
    t = type(value)
    if t is object:
        return True
    if not _marker_class(t):
        return False
    state = plain_getattr(value, "__dict__", None)
    if state:
        return False
    for klass in t.__mro__:
        slots = vars(klass).get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in ("__dict__", "__weakref__") and hasattr(value, name):
                return False
    return True


class NamelessInstanceError(TypeError):
    """An instance that holds nothing, which no module global or class
    attribute holds: nothing tells it from the next one."""

    def __init__(self, value: Any) -> None:
        super().__init__(f"{type(value).__qualname__} instance that holds nothing and no name finds")
        self.value = value


#: The fix for `NamelessInstanceError`.
NAMELESS_FIX = (
    "an object() or an instance of an empty class is told from another only by which one it is, and cash "
    "keys it by the name that holds it. Assign it to a module-level name or a class attribute "
    "(MISSING = object()) and pass that, or pass a value that says what you mean (a string, an enum member)."
)

#: ``{id(instance): (instance, name)}`` for the instances a name was found
#: for. The instance is held: a flag value lives as long as its module, and
#: its address must not be taken by another object.
_NAMES: dict[int, tuple[Any, str]] = {}
_NAMES_LOCK = threading.Lock()


#: The instances no name was found for, held so that their addresses stay
#: theirs, while the modules loaded are the ones searched (`_NAMELESS_AT`).
_NAMELESS: dict[int, Any] = {}
_NAMELESS_MAX = 1024
#: ``len(sys.modules)`` when `_NAMELESS` was last emptied, and the searches
#: that found nothing since.
_NAMELESS_AT = [0, 0]
#: Searches that may find nothing before one inside an object's state is
#: skipped (`name_of` with *patient* false) until a module is loaded.
_FRUITLESS_MAX = 8


def name_of(value: Any, patient: bool = True) -> str | None:
    """The name that finds *value*: ``module:NAME`` for a module global,
    ``module:Class.NAME`` for an attribute of a class a module holds. None
    when no loaded module holds it.

    Of several names, the same one is picked in every process: the first
    in sorted order, a module global before a class attribute. A name found
    before is checked, not searched for again, and an instance found to
    have none is not searched for again until a module is loaded.

    A search reads every loaded module's namespace (20 ms with pandas and
    scikit-learn loaded). *patient* false is for a value met inside an
    object's state, where an unnamed one is no error and there may be one
    per object: after a few searches that found nothing, the next ones are
    skipped until a module is loaded.
    """
    held = _NAMES.get(id(value))
    if held is not None and held[0] is value and _resolves(held[1], value):
        return held[1]
    loaded = len(sys.modules)
    with _NAMES_LOCK:
        if _NAMELESS_AT[0] != loaded:
            _NAMELESS.clear()
            _NAMELESS_AT[:] = [loaded, 0]
        elif _NAMELESS.get(id(value)) is value and held is None:
            return None
        elif not patient and _NAMELESS_AT[1] >= _FRUITLESS_MAX:
            return None
    name = _search(value)
    with _NAMES_LOCK:
        if name is not None:
            _NAMES[id(value)] = (value, name)
            _NAMELESS.pop(id(value), None)
            _NAMELESS_AT[1] = 0
        else:
            _NAMES.pop(id(value), None)
            _NAMELESS_AT[1] += 1
            if len(_NAMELESS) >= _NAMELESS_MAX:
                _NAMELESS.clear()
            _NAMELESS[id(value)] = value
    return name


def named_instance(name: str) -> str:
    """What a key's pickle writes for an instance keyed by *name*
    (`name_of`). Hashed, never loaded."""
    return name


def _resolves(name: str, value: Any) -> bool:
    """Does *name* still find *value*?"""
    module_name, _, rest = name.partition(":")
    target: Any = sys.modules.get(module_name)
    for part in rest.split("."):
        namespace = plain_getattr(target, "__dict__", None)
        if namespace is None or part not in namespace:
            return False
        target = namespace[part]
    return target is value


def _names_in(module_name: str, namespace: dict, value: Any) -> list[tuple[int, str]]:
    """Every name in a module's *namespace* that holds *value*, a global
    ranked before an attribute of a class the module defines."""
    found: list[tuple[int, str]] = []
    for name, held in _items(namespace):
        if held is value:
            found.append((0, f"{module_name}:{name}"))
        elif isinstance(held, type) and vars(held).get("__module__") == module_name:
            qualname = vars(held).get("__qualname__", name)
            for attr, attr_value in _items(vars(held)):
                if attr_value is value and isinstance(qualname, str) and "<" not in qualname:
                    found.append((1, f"{module_name}:{qualname}.{attr}"))
    return found


def _items(namespace: Any) -> list[tuple[str, Any]]:
    """*namespace*'s items, though another thread may be adding to it."""
    for _ in range(4):
        try:
            return list(namespace.items())
        except RuntimeError:  # changed size during iteration
            continue
    return []


def _search(value: Any) -> str | None:
    found: list[tuple[int, str]] = []
    for module_name, module in list(sys.modules.items()):
        if type(module) is not types.ModuleType and not isinstance(module, types.ModuleType):
            continue
        namespace = plain_getattr(module, "__dict__", None)
        if isinstance(namespace, dict):
            found.extend(_names_in(module_name, namespace, value))
    return min(found)[1] if found else None
