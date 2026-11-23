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
that cannot be pickled still refuses the key.

The user's own code is written this way only where the code-argument fold
keys it with what it reads: a function in a cached call's arguments, which
the decorator says (`write_user_functions_when`). Elsewhere (a global's
value, a helper's default) nothing else keys what it reads, and pickle
refuses it as before. A function made in a method of the user's and held
by an object an argument holds (lmfit's ``CompositeModel`` keeps one made
in its ``__init__``) refused every call's key when the library was the
user's own checkout.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

from ._type_memo import TypeMemo, class_bases, class_name, type_key
from ._user_attrs import module_file, named_attr, plain_getattr

#: Class attributes that are bookkeeping, not definitions: the abc cache
#: (``_abc_impl``) fills as ``isinstance`` is asked, ``__slotnames__`` as
#: ``copyreg`` pickles an instance, and the other two are the descriptors
#: every class with a ``__dict__`` gets.
_DERIVED_CLASS_ATTRS = frozenset({"__dict__", "__weakref__", "_abc_impl", "__slotnames__"})


#: The decorator's say on whether a function of the user's that no name
#: finds may be written as what defines it (`write_user_functions_when`).
_USER_FUNCTIONS: list[Callable[[Any], bool] | None] = [None]


def write_user_functions_when(check: Callable[[Any], bool]) -> None:
    """Write a function of the user's that no name finds as what defines
    it when *check* says so of it: while the arguments of a cached call are
    keyed, whose code-argument fold keys its code and what that reads."""
    _USER_FUNCTIONS[0] = check


def unnamed_library_code(*parts: Any) -> None:
    """What a key pickle writes for a library's function or class pickle
    cannot find by its name (`library_code_override`): a reference to this
    function and the parts that define it. Never called: key bytes are not
    loaded."""


def refused_unnamed(exc: BaseException) -> bool:
    """Did pickling stop at a function or class no name finds? CPython's
    wording for it: a class or a function at module level that is not
    there (``attribute lookup X on m failed`` up to 3.13, ``it's not found
    as m.X`` from 3.14, ``import of module 'm' failed`` for a module not
    loaded), or one made in a function (``Can't get local object``,
    ``<locals>``)."""
    text = str(exc)
    return any(marker in text for marker in _UNNAMED_WORDING)


_UNNAMED_WORDING = ("attribute lookup", "not found as", "import of module", "local object", "<locals>")


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
        check = _USER_FUNCTIONS[0]
        if type(obj) is not types.FunctionType or check is None or not check(obj):
            return None
    name = (plain_getattr(obj, "__module__", None), plain_getattr(obj, "__qualname__", None))
    if id(obj) in building:
        return (unnamed_library_code, ("cycle", *name))
    building.add(id(obj))  # one pickle's: the objects stay alive while it runs
    if isinstance(obj, type):
        attrs: list[tuple[str, Any]] = []
        for attr, held in vars(obj).items():
            if attr in _DERIVED_CLASS_ATTRS:
                continue
            if isinstance(held, (staticmethod, classmethod)):
                held = (class_name(type(held)), held.__func__)
            elif isinstance(held, property):
                held = ("property", held.fget, held.fset, held.fdel)
            attrs.append((attr, held))
        return (unnamed_library_code, ("class", *name, type(obj), class_bases(obj), tuple(attrs)))
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
    target: Any = sys.modules.get(plain_getattr(obj, "__module__", None) or "")
    for part in (plain_getattr(obj, "__qualname__", "") or "").split("."):
        if target is None or part.startswith("<"):
            return False
        target = named_attr(target, part, None)
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

    module = sys.modules.get(plain_getattr(obj, "__module__", None) or "")
    if module is None or module_file(module) is None:
        return True
    return is_user_code_module(module)
