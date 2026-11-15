"""How a key pickles a library's function or class that has no name to be
found by.

Pickle writes a function or a class by its module and qualified name.
A library can make either where no name reaches it: specutils builds the
model of a spectrum's axis with ``type("SpectralTabular1D", (Tabular1D,),
{...})`` inside a function, astropy's model metaclass gives that class
operator methods made in a function, and ``u.spectral()`` holds lambdas.
Pickle refuses each, and every ``Spectrum`` argument ran uncached
(KEY-UNHASHABLE-ARG).

A key's bytes are hashed, never loaded, so such an object can be written
as what defines it instead (`library_code_override`). The library's code
is keyed by the library's version, as code pickled by name is; what was
not found by name is keyed by what makes it what it is:

- a function by where it was written, what its code does
  (`source_norm.bytecode_identity`, which tells apart two lambdas on one
  line), what its closure captured and its defaults;
- a class by its metaclass, name, bases and the attributes its body or
  ``type()`` gave it.

What those hold is pickled as any other part of the key, so a capture
that cannot be pickled still refuses the key. The user's own code is not
written this way: the code-argument fold keys it, with what it reads.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

from ._type_memo import TypeMemo, type_key
from ._user_attrs import module_file

#: Class attributes that are bookkeeping, not definitions: the abc cache
#: (``_abc_impl``) fills as ``isinstance`` is asked, ``__slotnames__`` as
#: ``copyreg`` pickles an instance, and the other two are the descriptors
#: every class with a ``__dict__`` gets.
_DERIVED_CLASS_ATTRS = frozenset({"__dict__", "__weakref__", "_abc_impl", "__slotnames__"})


def unnamed_library_code(*parts: Any) -> None:
    """What a key pickle writes for a library's function or class pickle
    cannot find by its name (`library_code_override`): a reference to this
    function and the parts that define it. Never called: key bytes are not
    loaded."""


def refused_unnamed(exc: BaseException) -> bool:
    """Did pickling stop at a function or class no name finds? CPython's
    wording for it (3.10 to 3.14): a class or a function at module level
    that is not there (``attribute lookup X on m failed``), or one made in
    a function (``Can't get local object``, ``<locals>``)."""
    text = str(exc)
    return "attribute lookup" in text or "local object" in text or "<locals>" in text


def library_code_override(then: Callable[[Any], Any] | None = None) -> Callable[[Any], Any]:
    """A key pickler's ``reducer_override`` writing a library's function or
    class that pickle cannot find by name as what defines it (see the
    module docstring). *then* is the override to ask about everything else.

    One met again in the same pickle is written the second time as its
    name alone: pickle keeps no memo for an object until its reduce is
    written, so a class whose method's closure holds the class would be
    followed round the loop until the stack ran out. The first time it
    was met, in the same pickle, it was written whole.
    """
    building: set[int] = set()

    def override(obj: Any) -> Any:
        t = type(obj)
        if t is types.FunctionType or isinstance(obj, type):
            parts = _unnamed_parts(obj, building)
            if parts is not None:
                return parts
        return NotImplemented if then is None else then(obj)

    return override


def _unnamed_parts(obj: Any, building: set[int]) -> Any:
    """`library_code_override`'s reduce of *obj*, or None to pickle it as
    pickle does."""
    if type(obj) is types.FunctionType:
        if _found_by_name(obj):
            return None
    elif _class_found_by_name(obj):
        return None
    if _user_code(obj):
        return None
    name = (getattr(obj, "__module__", None), getattr(obj, "__qualname__", None))
    if id(obj) in building:
        return (unnamed_library_code, ("cycle", *name))
    building.add(id(obj))  # one pickle's: the objects stay alive while it runs
    if isinstance(obj, type):
        attrs: list[tuple[str, Any]] = []
        for attr, held in vars(obj).items():
            if attr in _DERIVED_CLASS_ATTRS:
                continue
            if isinstance(held, (staticmethod, classmethod)):
                held = (type(held).__name__, held.__func__)
            elif isinstance(held, property):
                held = ("property", held.fget, held.fset, held.fdel)
            attrs.append((attr, held))
        return (unnamed_library_code, ("class", *name, type(obj), obj.__bases__, tuple(attrs)))
    from .source_norm import bytecode_identity  # it imports the analysis package

    cells = tuple(_cell_contents(cell) for cell in obj.__closure__ or ())
    return (
        unnamed_library_code,
        ("function", *name, bytecode_identity(obj), cells, obj.__defaults__, obj.__kwdefaults__),
    )


_EMPTY_CELL = ("__cash_empty_cell__",)


def _cell_contents(cell: types.CellType) -> Any:
    try:
        return cell.cell_contents
    except ValueError:  # a variable not yet assigned when the closure was made
        return _EMPTY_CELL


def _found_by_name(obj: Any) -> bool:
    """Does pickle find *obj* again by its module and qualified name?"""
    target: Any = sys.modules.get(getattr(obj, "__module__", None) or "")
    for part in (getattr(obj, "__qualname__", "") or "").split("."):
        if target is None or part.startswith("<"):
            return False
        target = getattr(target, part, None)
    return target is obj


#: `_found_by_name` per class: asked of the class of every instance a key
#: pickle writes after the first refusal.
_CLASS_FOUND: TypeMemo[bool] = TypeMemo()


def _class_found_by_name(cls: type) -> bool:
    known = _CLASS_FOUND.get(type_key(cls))
    if known is None:
        known = _CLASS_FOUND.put(cls, _found_by_name(cls))
    return known


def _user_code(obj: Any) -> bool:
    """Is *obj* the user's code, judged by the module it names? A module
    without a file (a notebook's ``__main__``, ``exec``'d code) is the
    user's, and so is a module name nothing is loaded under."""
    from .install_paths import is_user_code_module  # it reads the installation

    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    if module is None or module_file(module) is None:
        return True
    return is_user_code_module(module)
