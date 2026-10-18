"""What a recorded unit's trace holds, and how each part is checked again.

A trace is a list of *items*, each a small tuple that names something by
module name and qualified name (never by file or line, which an edit
moves and ``exec``'d code does not have) together with the digest it had
when the unit ran:

``("code", module, qualname, digest)``
    a function or method that ran: its bytecode, constants (nested code
    included), names and defaults.
``("held", module, name, qualname, digest)``
    code that cannot be found by its qualified name (a lambda, a closure,
    a function a decorator wrapped), found through the global *name* that
    holds it: its code and the values of its closure cells.
``("class", module, qualname, digest)``
    the data of a class one of whose methods ran, of its subclasses, and
    of the class a method ran on (``self``'s class).
``("global", module, name, digest)``
    a global a function that ran reads by name (its bytecode name table),
    ``"absent"`` when unbound. A notebook global is compared by its
    lineage when it has one, as the key compares it.
``("attr", module, "mod.a.b", digest)``
    an attribute chain through a module of the user's own code.
``("gap", module, qualname, why)``
    code that ran but that nothing here can find again; it is counted,
    never compared.

A value that is too big or too strange to digest cheaply digests to
``"opaque:<type>"``: two of those compare equal and are counted as
unverified.
"""

from __future__ import annotations

import builtins
import contextlib
import dis
import functools
import hashlib
import pickle
import struct
import sys
import threading
import types
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

__all__ = [
    "VALUE_CAP",
    "class_digest",
    "code_digest",
    "name_table",
    "resolve",
    "value_digest",
]

#: The largest value, in bytes, digested by content; above it a value is opaque.
VALUE_CAP = 1 << 20
#: Nodes one value digest walks before it gives up (opaque).
_NODES = 20_000
#: Subclasses of a class whose method ran checked with it.
_SUBCLASSES = 16

_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_GLOBAL_STORES = frozenset({"STORE_GLOBAL", "DELETE_GLOBAL", "STORE_NAME", "DELETE_NAME"})
_DEREF_LOADS = frozenset({"LOAD_DEREF", "LOAD_FROM_DICT_OR_DEREF", "LOAD_CLASSDEREF"})
_ATTR_LOADS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})
_SKIP_OPS = frozenset({"CACHE", "EXTENDED_ARG", "NOP"})


class NameTable:
    """What one code object (not its nested code) reads by name."""

    __slots__ = ("chains", "derefs", "reads", "stores")

    def __init__(self, reads: frozenset, stores: frozenset, derefs: frozenset, chains: frozenset) -> None:
        self.reads = reads
        self.stores = stores
        self.derefs = derefs
        self.chains = chains


_TABLES: dict[types.CodeType, NameTable] = {}
_DIGESTS: dict[types.CodeType, str] = {}


def _bounded(memo: dict) -> None:
    if len(memo) > 8192:
        memo.clear()


def name_table(code: types.CodeType) -> NameTable:
    """The globals *code* reads and writes by name, the closure cells it
    reads, and the attribute chains it reads off a global (``np.linalg``)."""
    got = _TABLES.get(code)
    if got is not None:
        return got
    reads: set[str] = set()
    stores: set[str] = set()
    derefs: set[str] = set()
    chains: set[str] = set()
    chain: list[str] | None = None
    for ins in dis.get_instructions(code):
        op = ins.opname
        if op in _SKIP_OPS:
            continue
        if op in _GLOBAL_LOADS:
            reads.add(ins.argval)
            chain = [ins.argval]
            continue
        if op in _GLOBAL_STORES:
            stores.add(ins.argval)
        elif op in _DEREF_LOADS:
            derefs.add(ins.argval)
        elif op in _ATTR_LOADS and chain is not None:
            chain.append(ins.argval)
            chains.add(".".join(chain))
            continue
        chain = None
    got = NameTable(frozenset(reads), frozenset(stores), frozenset(derefs & set(code.co_freevars)), frozenset(chains))
    _bounded(_TABLES)
    _TABLES[code] = got
    return got


def _const_bytes(value: Any, out: list[bytes]) -> None:
    if isinstance(value, frozenset):
        out.append(b"F" + repr(sorted(repr(v) for v in value)).encode())
    elif isinstance(value, tuple):
        out.append(b"(")
        for v in value:
            _const_bytes(v, out)
        out.append(b")")
    else:
        out.append(repr(value).encode())


_CO_HAS_DOCSTRING = 0x4000000


_HAS_CONST = frozenset(dis.hasconst)


def _instructions(code: types.CodeType) -> bytes:
    """*code*'s instructions with each constant by value, not by its index
    in ``co_consts``: 3.14 keeps a docstring there only when there is one,
    which moves every other index (the key reads source and does not
    see it)."""
    out: list[bytes] = []
    for ins in dis.get_instructions(code):
        out.append(ins.opname.encode())
        if ins.opcode in _HAS_CONST:
            if isinstance(ins.argval, types.CodeType):
                out.append(b"<code>")
            else:
                _const_bytes(ins.argval, out)
        else:
            out.append(str(ins.arg).encode())
    return b" ".join(out)


def code_digest(code: types.CodeType) -> str:
    """A digest of *code* as a re-run would see it: its instructions with
    the constants they load (nested code included, frozensets in a fixed
    order), names and shape."""
    got = _DIGESTS.get(code)
    if got is not None:
        return got
    h = hashlib.blake2b(digest_size=12)
    stack = [code]
    while stack:
        c = stack.pop()
        h.update(_instructions(c))
        h.update(
            repr(
                (
                    c.co_names,
                    c.co_varnames,
                    c.co_freevars,
                    c.co_cellvars,
                    c.co_argcount,
                    c.co_kwonlyargcount,
                    c.co_flags & ~_CO_HAS_DOCSTRING,
                )
            ).encode()
        )
        # The constants it loads are in its instructions; a docstring, which
        # it does not load, is left out as the key leaves it out.
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    got = h.hexdigest()
    _bounded(_DIGESTS)
    _DIGESTS[code] = got
    return got


class _Opaque(Exception):
    """The value cannot be digested cheaply."""


class _CappedWriter:
    """A file that refuses more than ``VALUE_CAP`` bytes, so pickling a big
    value stops early instead of building all of it."""

    __slots__ = ("h", "n")

    def __init__(self, h: Any) -> None:
        self.h = h
        self.n = 0

    def write(self, data: bytes) -> int:
        self.n += len(data)
        if self.n > VALUE_CAP:
            raise _Opaque
        self.h.update(data)
        return len(data)


_SCALARS = (int, float, complex, bool, type(None), type(Ellipsis))


#: The module name the notebook's code runs under (set by the shadow).
NOTEBOOK = ["__main__"]


def _is_user_type(t: type) -> bool:
    from ..install_paths import is_user_code_module

    if getattr(t, "__module__", None) == NOTEBOOK[0]:
        return True
    mod = sys.modules.get(getattr(t, "__module__", "") or "")
    return mod is not None and is_user_code_module(mod)


class _Digester:
    def __init__(self, h: Any) -> None:
        self.h = h
        self.nodes = 0
        self.active: set[int] = set()
        self.types: set[int] = set()

    def feed(self, v: Any) -> None:
        self.nodes += 1
        if self.nodes > _NODES:
            raise _Opaque
        h = self.h
        t = type(v)
        if t in _SCALARS:
            h.update(b"s" + t.__name__.encode() + repr(v).encode())
            return
        if t is str:
            if len(v) > VALUE_CAP:
                raise _Opaque
            h.update(b"u" + struct.pack("<q", len(v)) + v.encode("utf-8", "surrogatepass"))
            return
        if t is bytes or t is bytearray:
            if len(v) > VALUE_CAP:
                raise _Opaque
            h.update(b"b" + struct.pack("<q", len(v)) + bytes(v))
            return
        if isinstance(v, types.ModuleType):
            h.update(b"m" + str(v.__name__).encode())
            return
        if isinstance(v, types.FunctionType):
            self._function(v)
            return
        if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            owner = getattr(v, "__self__", None)
            h.update(b"B" + f"{getattr(v, '__module__', None)}.{getattr(v, '__qualname__', '')}".encode())
            if owner is not None and not isinstance(owner, types.ModuleType):
                h.update(b"@" + type(owner).__qualname__.encode())
            return
        if isinstance(v, type):
            self._type(v)
            return
        if id(v) in self.active:
            h.update(b"cycle")
            return
        self.active.add(id(v))
        try:
            self._compound(v, t)
        finally:
            self.active.discard(id(v))

    def _compound(self, v: Any, t: type) -> None:
        h = self.h
        if t is tuple or t is list:
            h.update(t.__name__.encode() + struct.pack("<q", len(v)))
            for x in v:
                self.feed(x)
            return
        if t is dict:
            h.update(b"d" + struct.pack("<q", len(v)))
            for k, x in v.items():
                self.feed(k)
                self.feed(x)
            return
        if t is set or t is frozenset:
            if len(v) > _NODES:
                raise _Opaque
            parts = sorted(value_digest(x) for x in v)
            h.update(t.__name__.encode() + "|".join(parts).encode())
            return
        if isinstance(v, types.MethodType):
            h.update(b"M" + type(v.__self__).__qualname__.encode())
            self.feed(v.__func__)
            return
        if isinstance(v, functools.partial):
            h.update(b"P")
            self.feed(v.func)
            self.feed(v.args)
            self.feed(dict(v.keywords))
            return
        if isinstance(v, (staticmethod, classmethod)):
            h.update(t.__name__.encode())
            self.feed(v.__func__)
            return
        if isinstance(v, property):
            h.update(b"prop")
            for f in (v.fget, v.fset, v.fdel):
                self.feed(f)
            return
        from ..jit_functions import jit_python_function

        jitted = jit_python_function(v)
        if jitted is not None:
            h.update(b"jit")
            self.feed(jitted)
            return
        nbytes = getattr(v, "nbytes", None)
        if isinstance(nbytes, int) and nbytes > VALUE_CAP:
            raise _Opaque
        shape = getattr(v, "shape", None)
        if isinstance(shape, tuple) and all(isinstance(n, int) for n in shape):
            size = 8
            for n in shape:
                size *= n
            if size > VALUE_CAP:
                raise _Opaque
        if _is_user_type(t) and t.__reduce_ex__ is object.__reduce_ex__:
            state = getattr(v, "__dict__", None)
            if isinstance(state, dict):
                h.update(b"o")
                self._type(t)
                self.feed(state)
                return
        h.update(b"p" + f"{t.__module__}.{t.__qualname__}".encode())
        try:
            pickle.Pickler(_CappedWriter(h), protocol=4).dump(v)
        except _Opaque:
            raise
        except Exception as exc:
            raise _Opaque from exc

    def _function(self, fn: types.FunctionType) -> None:
        h = self.h
        h.update(b"f" + f"{fn.__module__}.{fn.__qualname__}".encode())
        h.update(code_digest(fn.__code__).encode())
        if fn.__defaults__:
            self.feed(fn.__defaults__)
        if fn.__kwdefaults__:
            self.feed(fn.__kwdefaults__)
        starts = getattr(_STARTS, "cells", None)
        cells = starts.get(fn.__code__) if starts else None
        if cells is None or len(cells) != len(fn.__closure__ or ()):
            cells = _cell_digests(fn.__closure__ or (), self)
        # Each cell by its own digest, so the cells a function had as it
        # started (`start_cells`) stand in for what it left in them.
        for got in cells:
            if got is None:
                raise _Opaque
            h.update(b"c" + got.encode())

    def _type(self, t: type) -> None:
        h = self.h
        h.update(b"t" + f"{t.__module__}.{t.__qualname__}".encode())
        if id(t) in self.types or not _is_user_type(t):
            return
        self.types.add(id(t))
        h.update(class_digest(t, self).encode())


_STARTS = threading.local()


def _cell_digests(closure: Iterable[Any], within: _Digester | None) -> tuple[str | None, ...]:
    out: list[str | None] = []
    for cell in closure:
        try:
            contents = cell.cell_contents
        except ValueError:
            out.append("empty")
            continue
        out.append(_sub_digest(contents, within))
    return tuple(out)


def _sub_digest(v: Any, within: _Digester | None) -> str | None:
    """*v*'s digest inside another's, sharing its count of what was read
    (a closure that holds itself ends there); None when opaque."""
    h = hashlib.blake2b(digest_size=12)
    d = _Digester(h)
    if within is not None:
        d.nodes, d.active, d.types = within.nodes, within.active, within.types
    try:
        d.feed(v)
    except _Opaque:
        return None
    except Exception:  # noqa: BLE001 - a value that breaks on being read
        return None
    finally:
        if within is not None:
            within.nodes = d.nodes
    return h.hexdigest()


def start_cells(frame: Any, code: types.CodeType) -> tuple[str | None, ...] | None:
    """What *code*'s closure cells held as its *frame* started: a call
    that writes to them (a counter, a log list) is then checked by what it
    was called with, the way its key was built, not by what it left."""
    try:
        local = frame.f_locals
        values = [local.get(name, _EMPTY) for name in code.co_freevars]
    except Exception:  # noqa: BLE001 - cells that cannot be read are digested afterwards
        return None
    return tuple("empty" if v is _EMPTY else _sub_digest(v, None) for v in values)


_EMPTY = object()


@contextlib.contextmanager
def call_starts(cells: Mapping[types.CodeType, tuple] | None) -> Iterator[None]:
    """Digest functions by the cells they started with (`start_cells`)
    while building a unit's items."""
    before = getattr(_STARTS, "cells", None)
    _STARTS.cells = cells
    try:
        yield
    finally:
        _STARTS.cells = before


def value_digest(value: Any) -> str:
    """A digest of *value* that is the same in another process for an equal
    value, or ``"opaque:<type>"`` when it is too big or cannot be read."""
    h = hashlib.blake2b(digest_size=12)
    try:
        _Digester(h).feed(value)
    except _Opaque:
        return "opaque:" + type(value).__qualname__
    except Exception:  # noqa: BLE001 - a value that breaks on being read
        return "opaque:" + type(value).__qualname__
    return h.hexdigest()


#: Not what a class holds: ``__slotnames__`` is copyreg's cache, set the
#: first time an instance is pickled; the line a class starts on and the
#: attributes its methods assign come with its code.
_CLASS_SKIP = frozenset(
    {
        "__dict__",
        "__weakref__",
        "__doc__",
        "__module__",
        "__qualname__",
        "__annotations__",
        "__slotnames__",
        "__firstlineno__",
        "__static_attributes__",
    }
)


def class_digest(cls: type, within: _Digester | None = None) -> str:
    """A digest of what *cls* holds: its attributes (methods by code, data
    by value) and the names of its bases."""
    h = hashlib.blake2b(digest_size=12)
    d = _Digester(h)
    if within is not None:
        d.types = within.types
        d.types.add(id(cls))
    h.update(",".join(f"{c.__module__}.{c.__qualname__}" for c in cls.__mro__).encode())
    try:
        attrs = sorted((k, v) for k, v in vars(cls).items() if k not in _CLASS_SKIP)
    except TypeError:
        attrs = [(k, v) for k, v in vars(cls).items() if k not in _CLASS_SKIP]
    for name, value in attrs:
        h.update(b"\x00" + str(name).encode())
        try:
            d.feed(value)
        except _Opaque:
            h.update(b"opaque:" + type(value).__qualname__.encode())
        except Exception:  # noqa: BLE001 - a value that cannot be digested is opaque
            h.update(b"opaque")
    return h.hexdigest()


_MISSING = object()


def candidates(obj: Any) -> Iterable[types.FunctionType]:
    """The functions *obj* may stand for: itself, what a staticmethod,
    classmethod, bound method, property, partial or JIT dispatcher holds,
    and what a ``functools.wraps`` wrapper wraps. Read off known types
    only: no attribute lookup runs on an object of the user's."""
    from ..jit_functions import jit_python_function

    stack = [obj]
    seen = 0
    while stack and seen < 16:
        seen += 1
        o = stack.pop()
        if isinstance(o, types.FunctionType):
            yield o
            wrapped = o.__dict__.get("__wrapped__") if o.__dict__ else None
            if wrapped is not None:
                stack.append(wrapped)
        elif isinstance(o, (types.MethodType, staticmethod, classmethod)):
            stack.append(o.__func__)
        elif isinstance(o, property):
            stack.extend(f for f in (o.fget, o.fset, o.fdel) if f is not None)
        elif isinstance(o, functools.partial):
            stack.append(o.func)
        elif o is not None and not isinstance(o, (type, types.ModuleType)):
            jitted = jit_python_function(o)
            if jitted is not None:
                stack.append(jitted)


def _code_of(obj: Any, qualname: str) -> types.CodeType | None:
    for c in candidates(obj):
        if c.__code__.co_qualname == qualname:
            return c.__code__
    return None


def _nested_code(code: types.CodeType, qualname: str) -> types.CodeType | None:
    stack = [code]
    while stack:
        c = stack.pop()
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                if k.co_qualname == qualname:
                    return k
                stack.append(k)
    return None


def namespace_of(module: str, notebook: Mapping[str, Any], notebook_name: str) -> Mapping[str, Any] | None:
    """The globals of *module*: the notebook's for the notebook's own name."""
    if module == notebook_name:
        return notebook
    mod = sys.modules.get(module)
    ns = getattr(mod, "__dict__", None)
    return ns if isinstance(ns, dict) else None


def resolve(
    module: str, qualname: str, notebook: Mapping[str, Any], notebook_name: str
) -> tuple[types.CodeType | None, Any, Any]:
    """``(code, function, owner)`` for *qualname* in *module* as it is now:
    the code it names, the function holding it (None for nested code) and
    the object it was found on (a class for a method)."""
    ns = namespace_of(module, notebook, notebook_name)
    if ns is None:
        return None, None, None
    parts = qualname.split(".")
    obj: Any = ns.get(parts[0], _MISSING)
    if obj is _MISSING:
        return None, None, None
    owner: Any = None
    for i, part in enumerate(parts[1:], start=1):
        if part == "<locals>":
            prefix = ".".join(parts[:i])
            outer = _code_of(obj, prefix)
            if outer is None:
                return None, None, None
            return _nested_code(outer, qualname), None, obj
        owner = obj
        try:
            obj = inspect_getattr(obj, part)
        except AttributeError:
            return None, None, None
    return _code_of(obj, qualname), obj, owner


def inspect_getattr(obj: Any, name: str) -> Any:
    """``obj.name`` without running a descriptor: a class's own staticmethod,
    classmethod or property object, not what binding it gives."""
    if isinstance(obj, type):
        for klass in obj.__mro__:
            found = klass.__dict__.get(name, _MISSING)
            if found is not _MISSING:
                return found
        raise AttributeError(name)
    if isinstance(obj, types.ModuleType):
        found = obj.__dict__.get(name, _MISSING)
        if found is _MISSING:
            raise AttributeError(name)
        return found
    return getattr(obj, name)


def is_builtin_name(name: str) -> bool:
    return hasattr(builtins, name)
