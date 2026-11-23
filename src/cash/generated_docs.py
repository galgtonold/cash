"""How a key holds a docstring an object was given at run time.

Like a class's or a module's docstring, an object's own ``__doc__`` is an
input only when code reads docstrings (`reads_own_doc`): its class's code
loading ``__doc__`` (``self.__doc__``, ``getattr(self, "__doc__")``,
``inspect.getdoc``). Otherwise it is documentation, and the key leaves it
out. Where it is read, it is keyed as follows.

A library that builds its objects from tables (xclim's indicators) writes
each one's ``__doc__`` from what the object and its class hold: the choices
of a parameter, a description given as a lambda. Text rendered that way is
process-specific in two places only: a set renders in the order its string
hashes give it, fresh in each process (``{'mask', 'clip'}`` here,
``{'clip', 'mask'}`` there), and a function renders with its address
(``<function <lambda> at 0x7f...>``). Keyed as written, such a docstring
moved with every new process, and every call reaching the object missed.

Only those renderings are made the same, and only where they are the
rendering of a value the object or its class holds: a ``{...}`` fragment
whose items are a set held there is written in sorted order, and the
address in an ``at 0x...`` fragment that is the address of an object held
there is replaced by that object's type. The value itself is keyed where it
is held. Everything else in the text is keyed as it is: an item added or
dropped, a word changed, a list that is not a held set reordered, an address
of nothing held all still change the key.
"""

from __future__ import annotations

import ast
import dis
import re
import types
import weakref
from typing import Any

from . import _weak
from ._type_memo import class_dict, class_name, mro_of
from ._user_attrs import plain_getattr

#: An address in a repr (``<function f at 0x7f...>``).
_ADDRESS = re.compile(r"\bat 0x([0-9a-fA-F]+)")
#: A set display with no braces inside it.
_SET_DISPLAY = re.compile(r"\{[^{}]*\}")
#: How many objects `_held` looks through.
_HELD_BUDGET = 20000
#: How deep `_held` follows containers and object state.
_HELD_DEPTH = 8


#: Names whose load reads a docstring (as `global_reads.DOCSTRING_READS`).
_DOC_NAMES = frozenset({"__doc__", "getdoc", "cleandoc"})

#: code -> does it (or code nested in it) load a docstring name.
_CODE_READS: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def reads_own_doc(t: type) -> bool:
    """Does code of *t* or of a class it inherits from read a docstring at
    run time? A load of ``__doc__``, ``getdoc`` or ``cleandoc`` as a name or
    attribute, or ``"__doc__"`` as a constant (``getattr(self,
    "__doc__")``, ``vars(self)["__doc__"]``); a store (``self.__doc__ =
    ...``, which writes the docstring) is not a read. Asked of the class as
    it is now: a method replaced later is seen."""
    return any(_class_reads_doc(klass) for klass in t.__mro__)


#: class -> (its members then, their ids, whether its own code reads a
#: docstring). Weak: a class dropped is forgotten. The members are kept so
#: their ids are not reused while the entry stands.
_CLASS_READS: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def _class_reads_doc(klass: type) -> bool:
    """Does code *klass* itself holds read a docstring? Remembered while
    its members are the same objects: a method replaced is read again."""
    members = tuple(class_dict(klass).values())
    ids = list(map(id, members))
    try:
        entry = _CLASS_READS.get(klass)
    except TypeError:
        entry = None
    if entry is not None and entry[1] == ids:
        return entry[2]
    found = any(_code_reads_doc(fn.__code__) for member in members for fn in _functions_of(member))
    try:
        _CLASS_READS[klass] = (members, ids, found)
    except TypeError:
        pass
    return found


def _functions_of(member: Any) -> list:
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if type(f) is types.FunctionType]
    return [member] if type(member) is types.FunctionType else []


def _code_reads_doc(code: types.CodeType) -> bool:
    known = _CODE_READS.get(code)
    if known is None:
        known = _CODE_READS[code] = _loads_doc(code)
    return known


def _loads_doc(code: types.CodeType) -> bool:
    if not (_DOC_NAMES & set(code.co_names) or "__doc__" in code.co_consts):
        found = False
    else:
        found = any(
            (ins.opname.startswith("LOAD_") or ins.opname == "IMPORT_FROM") and ins.argval in _DOC_NAMES
            for ins in dis.get_instructions(code)
        )
    return found or any(_code_reads_doc(c) for c in code.co_consts if type(c) is types.CodeType)


def keyed_state_doc(owner: Any, doc: Any) -> tuple[bool, Any]:
    """``(keep, form)`` for the ``__doc__`` *owner* holds in its own state:
    left out of the key unless its class's code reads docstrings
    (`reads_own_doc`), then keyed by `doc_form`."""
    if type(doc) is not str:
        return True, doc
    if not reads_own_doc(type(owner)):
        return False, None
    return True, doc_form(owner, doc)


def doc_form(owner: Any, doc: str) -> str:
    """*doc*, the ``__doc__`` *owner* holds in its own state, with each
    rendering of a value *owner* or its class holds made the same in every
    process (see the module docstring). *doc* itself when it has none."""
    has_address = "at 0x" in doc
    has_set = "{" in doc
    if not (has_address or has_set):
        return doc
    ids, sets = _held(owner)
    out = doc
    if has_set and sets:
        out = _SET_DISPLAY.sub(lambda m: _sorted_set(m.group(0), sets), out)
    if has_address and ids:
        out = _ADDRESS.sub(lambda m: _held_address(m, ids), out)
    return out


def _sorted_set(text: str, sets: set[frozenset]) -> str:
    try:
        shown = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text
    if type(shown) is not set or len(shown) < 2 or frozenset(shown) not in sets:
        return text
    try:
        items = sorted(shown, key=lambda x: (class_name(type(x)), x))
    except TypeError:
        return text
    return "{" + ", ".join(map(repr, items)) + "}"


def _held_address(match: re.Match, ids: dict[int, str]) -> str:
    kind = ids.get(int(match.group(1), 16))
    return match.group(0) if kind is None else f"at <held {kind}>"


def _held(owner: Any) -> tuple[dict[int, str], set[frozenset]]:
    """The objects *owner*'s state and its classes' data hold, as ``{id:
    type name}``, and the sets of literals among them. Bounded: what lies
    past the budget is not found, and its rendering stays as written."""
    ids: dict[int, str] = {}
    sets: set[frozenset] = set()
    roots: list[Any] = []
    state = plain_getattr(owner, "__dict__", None)
    if isinstance(state, dict):
        roots.append(state)
    for klass in mro_of(type(owner)):
        if klass is not object:
            roots.append(dict(class_dict(klass)))
    stack = [(r, 0) for r in roots]
    budget = _HELD_BUDGET
    seen: set[int] = set()
    while stack and budget > 0:
        value, depth = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        budget -= 1
        t = type(value)
        if t in (str, bytes, int, float, bool, complex, type(None)):
            continue
        ids[id(value)] = t.__qualname__
        if depth >= _HELD_DEPTH or issubclass(t, (type, types.ModuleType)):
            continue
        if t is set or t is frozenset:
            if all(type(x) in (str, int, float, bool, type(None)) for x in value):
                sets.add(frozenset(value))
            stack.extend((x, depth + 1) for x in value)
        elif isinstance(value, dict):
            stack.extend((x, depth + 1) for x in value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend((x, depth + 1) for x in value)
        elif isinstance(value, (staticmethod, classmethod)):
            stack.append((value.__func__, depth + 1))
        elif isinstance(value, property):
            stack.extend((f, depth + 1) for f in (value.fget, value.fset, value.fdel) if f is not None)
        elif not callable(value) or t is types.MethodType:
            inner = plain_getattr(value, "__dict__", None)
            if type(inner) is dict:
                stack.append((inner, depth + 1))
        if t is types.FunctionType:
            for cell in value.__closure__ or ():
                try:
                    stack.append((cell.cell_contents, depth + 1))
                except ValueError:
                    pass
    return ids, sets
