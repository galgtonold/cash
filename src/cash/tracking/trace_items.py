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
``("surface", module, qualname, digest)``
    a class a method that ran was called on, by its surface
    (`class_surface_digest`): what it holds and its bases', each method by
    name only, for a trace whose code items check the methods that ran.
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
import contextvars
import dataclasses
import dis
import functools
import hashlib
import inspect
import pickle
import struct
import sys
import threading
import types
import typing
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ..kept_state import global_rng_stand_in
from ..pint_values import pint_kind, registry_digest

__all__ = [
    "SURFACE",
    "VALUE_CAP",
    "class_digest",
    "class_surface_digest",
    "code_digest",
    "name_table",
    "resolve",
    "value_digest",
]

#: ``type X = ...``'s class (3.12+), else None.
TYPE_ALIAS = getattr(typing, "TypeAliasType", None)

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
    # ``getattr(mod.a, "b")`` reads ``mod.a.b``, as the attribute read does:
    # the name ``getattr`` loaded just before a global's chain.
    by_getattr = False
    for ins in dis.get_instructions(code):
        op = ins.opname
        if op in _SKIP_OPS:
            continue
        if op in _GLOBAL_LOADS:
            reads.add(ins.argval)
            by_getattr = chain == ["getattr"]
            chain = [ins.argval]
            continue
        if op == "LOAD_CONST" and by_getattr and chain is not None and isinstance(ins.argval, str):
            chains.add(".".join((*chain, ins.argval)))
            by_getattr = False
            chain = None
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


class _SetsInOrder(pickle.Pickler):
    """Pickles a set inside another value in an order fixed by its items,
    not by string hashes: those change with each process, and a set held in
    an object (sklearn's ``StrOptions``, in the closure of every
    ``@validate_params`` function) pickled in another order made
    ``cross_val_score`` look changed after every restart."""

    def persistent_id(self, obj: Any) -> Any:
        # Not ``reducer_override``: the pickler may skip it for exact sets.
        t = type(obj)
        if t is set or t is frozenset:
            if len(obj) > _NODES:
                raise _Opaque
            return t.__name__ + "|" + "|".join(sorted(map(value_digest, obj)))
        if pint_kind(t) == "registry":
            return "pint|" + _registry_digest(obj)
        return None

    def reducer_override(self, obj: Any) -> Any:
        # A module's global RNG (``np.random.mtrand._rand``, which every
        # ``scipy.stats`` distribution holds) by the module's name, as a key
        # takes it: its state is fresh entropy in each process unseeded, and
        # the global RNG channel, not the value, accounts for it.
        return global_rng_stand_in(obj) or NotImplemented


def _registry_digest(registry: Any) -> str:
    """`registry_digest`, or `_Opaque` for a registry it cannot key."""
    try:
        return registry_digest(registry)
    except TypeError as exc:
        raise _Opaque from exc


_SCALARS = (int, float, complex, bool, type(None), type(Ellipsis))


#: The module name the notebook's code runs under (set by the shadow).
NOTEBOOK = ["__main__"]


def is_user_type(t: type) -> bool:
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
        #: Codes more than one function of a class being digested runs
        #: (`_shared_codes`): which of them started with the cells a call
        #: saw (`start_cells`) is not known, so each goes by its own.
        self.shared: frozenset[types.CodeType] = frozenset()
        #: The classes being digested now, around this point (`class_digest`).
        self.open: set[int] = set()

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
        if issubclass(type(v), types.ModuleType):
            h.update(b"m" + str(v.__name__).encode())
            return
        if type(v) is types.FunctionType:
            self._function(v)
            return
        if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            owner = getattr(v, "__self__", None)
            h.update(b"B" + f"{getattr(v, '__module__', None)}.{getattr(v, '__qualname__', '')}".encode())
            if owner is not None and not issubclass(type(owner), types.ModuleType):
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
        if pint_kind(t) == "registry":
            # A pint unit registry (xclim's ``units``, read by every
            # indicator) by its definitions, as a key holds one
            # (`cash.pint_values`): its pickle names only the class, and its
            # state holds caches it derived and a weak index. One whose
            # definitions cannot be keyed is opaque. Asked first: an
            # attribute read of a registry (``nbytes``) parses it as a unit
            # and adds that unit to its definitions.
            h.update(b"pint" + _registry_digest(v).encode())
            return
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
        if t is types.MappingProxyType:
            # A read-only view of a dict (a ``mappingproxy({})`` default):
            # by what it shows, as the dict is; pickle refuses it.
            h.update(b"mp" + struct.pack("<q", len(v)))
            for k, x in v.items():
                self.feed(k)
                self.feed(x)
            return
        if t is inspect.Signature:
            # Pickled, its parameters' defaults go through pickle, which
            # refuses a read-only mapping among them: field by field.
            h.update(b"sig")
            self.feed(v.return_annotation)
            for p in v.parameters.values():
                self.feed(p)
            return
        if t is inspect.Parameter:
            h.update(b"par" + f"{int(v.kind)}:{v.name}".encode())
            self.feed(v.default)
            self.feed(v.annotation)
            return
        if t is set or t is frozenset:
            if len(v) > _NODES:
                raise _Opaque
            parts = sorted(value_digest(x) for x in v)
            h.update(t.__name__.encode() + "|".join(parts).encode())
            return
        if type(v) is types.MethodType:
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
        if t is functools.cached_property:
            # Pickled, its function is looked up by a name that finds the
            # property: by its code instead.
            h.update(b"cprop")
            self.feed(v.func)
            return
        if t is TYPE_ALIAS:
            h.update(b"alias")
            self._alias(v)
            return
        if t is dataclasses.Field:
            h.update(b"field")
            self.feed(v.default)
            self.feed(v.default_factory)
            return
        from ..jit_functions import jit_python_function

        jitted = jit_python_function(v)
        if jitted is not None:
            h.update(b"jit")
            self.feed(jitted)
            return
        from .._user_attrs import metadata_attribute

        nbytes = metadata_attribute(v, "nbytes", None)
        if isinstance(nbytes, int) and nbytes > VALUE_CAP:
            raise _Opaque
        shape = metadata_attribute(v, "shape", None)
        if isinstance(shape, tuple) and all(isinstance(n, int) for n in shape):
            size = 8
            for n in shape:
                size *= n
            if size > VALUE_CAP:
                raise _Opaque
        if is_user_type(t) and t.__reduce_ex__ is object.__reduce_ex__:
            state = getattr(v, "__dict__", None)
            if isinstance(state, dict):
                h.update(b"o")
                self._type(t)
                if "__doc__" in state:
                    # A docstring given at run time: an input only where
                    # code reads it (`generated_docs`), as in a key.
                    from ..generated_docs import keyed_state_doc

                    keep, form = keyed_state_doc(v, state["__doc__"])
                    state = {k: x for k, x in state.items() if k != "__doc__"}
                    if keep:
                        state["__doc__"] = form
                self.feed(state)
                return
        h.update(b"p" + f"{t.__module__}.{t.__qualname__}".encode())
        try:
            _SetsInOrder(_CappedWriter(h), protocol=4).dump(v)
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
        cells = starts.get(fn.__code__) if starts and fn.__code__ not in self.shared else None
        if (
            cells is None
            or len(cells) != len(fn.__closure__ or ())
            or getattr(cells, "held", None) != _held_identities(_cell_values(fn.__closure__ or ()))
        ):
            # No start seen, or the one seen was another function of the
            # same code (stacked wrappers one decorator made, getters one
            # factory made): what its cells held was not this one's.
            cells = _cell_digests(fn.__closure__ or (), self)
        elif _AS_IT_IS in cells:
            now = _cell_digests(fn.__closure__ or (), self)
            cells = tuple(n if c == _AS_IT_IS else c for c, n in zip(cells, now))
        # Each cell by its own digest, so the cells a function had as it
        # started (`start_cells`) stand in for what it left in them.
        for got in cells:
            if got is None:
                raise _Opaque
            h.update(b"c" + got.encode())
        for name, forward in private_forwards(fn):
            # What an ``exec``-built stub forwards to: the stub's own code is
            # the same for every function its decorator wraps.
            h.update(b"g" + name.encode())
            self.feed(forward)

    def _alias(self, alias: Any) -> None:
        """A ``type X = ...`` alias by what defines it, read or not: the
        code that builds its value lazily, the globals that code reads and
        its type parameters. Its ``__value__`` is built once, on its first
        read, and pickles by name: keyed by that, an edit to the alias was
        seen only when the value was first built inside the call."""
        h = self.h
        h.update(f"{alias.__module__}.{alias.__name__}".encode())
        params = tuple(alias.__type_params__ or ())
        for p in params:
            self._type_param(p)
        build = _alias_builder(alias)
        if build is None:
            # Made by calling ``TypeAliasType(name, value)``: its value is
            # given, not built later.
            h.update(b"v")
            self.feed(alias.__value__)
            return
        h.update(code_digest(build.__code__).encode())
        code = build.__code__
        for name, cell in zip(code.co_freevars, build.__closure__ or ()):
            try:
                held = cell.cell_contents
            except ValueError:
                h.update(b"empty")
                continue
            if name == "__classdict__" or any(held is p for p in params):
                # The class it is made in (its names are read by the code
                # only through the class), or a type parameter, fed above.
                continue
            h.update(b"c" + name.encode())
            self.feed(held)
        ns = build.__globals__
        for name in sorted(_code_names(code)):
            h.update(b"g" + name.encode())
            value = ns.get(name, _MISSING)
            if value is _MISSING:
                value = builtins.__dict__.get(name, _MISSING)
            if value is _MISSING:
                h.update(b"absent")
            else:
                self.feed(value)

    def _type_param(self, p: Any) -> None:
        """A type parameter of an alias, by its kind, name and what bounds
        it."""
        h = self.h
        h.update(b"tp" + f"{type(p).__qualname__}:{getattr(p, '__name__', '')}".encode())
        for part in ("__bound__", "__constraints__", "__default__", "__covariant__", "__contravariant__"):
            got = getattr(p, part, _MISSING)
            if got is _MISSING or got is _NO_DEFAULT:
                continue
            h.update(part.encode())
            self.feed(got)

    def _type(self, t: type) -> None:
        h = self.h
        h.update(b"t" + f"{t.__module__}.{t.__qualname__}".encode())
        if id(t) in self.types or not is_user_type(t):
            return
        self.types.add(id(t))
        h.update(_class_digest_now(t, self, surface=SURFACE.get()).encode())


#: A type parameter's ``__default__`` when it has none (3.13+).
_NO_DEFAULT = getattr(typing, "NoDefault", object())


def _alias_builder(alias: Any) -> types.FunctionType | None:
    """The function a ``type X = ...`` statement made to build *alias*'s
    value on its first read (kept after), else None."""
    got = getattr(alias, "evaluate_value", None)  # 3.14
    if type(got) is types.FunctionType:
        return got
    import gc

    for ref in gc.get_referents(alias):
        if type(ref) is types.FunctionType:
            return ref
    return None


def _code_names(code: types.CodeType) -> set[str]:
    """The global names *code* and the code nested in it may read."""
    out: set[str] = set()
    stack = [code]
    while stack:
        c = stack.pop()
        out.update(c.co_names)
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return out


_STARTS = threading.local()


def _cell_digests(closure: Iterable[Any], within: _Digester | None) -> tuple[str | None, ...]:
    out: list[str | None] = []
    for cell in closure:
        try:
            contents = cell.cell_contents
        except ValueError:
            out.append("empty")
            continue
        # Each cell whole, the classes it holds included even where the
        # digest around it went through them already: so is a cell digested
        # as its function starts (`start_cells`), and the two must agree.
        out.append(_sub_digest(contents, within, own_types=True))
    return tuple(out)


def _sub_digest(v: Any, within: _Digester | None, *, own_types: bool = False) -> str | None:
    """*v*'s digest inside another's, sharing its count of what was read
    (a closure that holds itself ends there); None when opaque. With
    *own_types*, the classes the other went through are gone through again."""
    h = hashlib.blake2b(digest_size=12)
    d = _Digester(h)
    if within is not None:
        d.nodes, d.active, d.open = within.nodes, within.active, within.open
        # Own types: the classes digested already are gone through again,
        # but not one being digested around this point, where it ends.
        d.types, d.shared = (set(within.open), frozenset()) if own_types else (within.types, within.shared)
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
    return _StartCells(
        ("empty" if v is _EMPTY else _AS_IT_IS if isinstance(v, type) else _sub_digest(v, None) for v in values),
        _held_identities(values),
    )


class _StartCells(tuple):
    """A start's cell digests, and which objects the cells held
    (`_held_identities`): the function that started, among those sharing
    its code."""

    held: tuple[int | None, ...]

    def __new__(cls, digests: Iterable[str | None], held: tuple[int | None, ...]) -> _StartCells:
        self = super().__new__(cls, digests)
        self.held = held
        return self


#: Values a closure may rebind as it runs (``nonlocal n; n += 1``), left
#: out of `_held_identities`.
_REBOUND_TYPES = frozenset({int, float, complex, bool, str, bytes, type(None), tuple, frozenset})


def _held_identities(values: Iterable[Any]) -> tuple[int | None, ...]:
    """The ids of the objects cells hold, numbers, strings and tuples left
    out: a function's own cells keep them through its run (a counter it
    bumps aside), while another function of the same code holds others."""
    return tuple(None if v is _EMPTY or type(v) in _REBOUND_TYPES else id(v) for v in values)


def _cell_values(closure: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for cell in closure:
        try:
            out.append(cell.cell_contents)
        except ValueError:
            out.append(_EMPTY)
    return out


_EMPTY = object()
#: A start cell holding a class, digested where it is read instead: a
#: method's ``__class__`` cell (``super()``) holds its own class, which,
#: digested on its own as the call started, never matched the class's
#: digest of that method, and every statement making an instance ran again.
_AS_IT_IS = "as-it-is"


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
#: first time an instance is pickled (pydantic's setter memo likewise, on
#: the first assignment); the line a class starts on and the
#: attributes its methods assign come with its code. On 3.14 the annotations
#: live in ``__annotate_func__``, and the first read of them
#: (``typing.get_type_hints``, dataclasses, pydantic) stores
#: ``__annotations_cache__``: skipped as ``__annotations__`` is, the class
#: read and the class not read yet were two digests, and a process that read
#: them saw a class "changed" from the one an earlier process stored. A code
#: read of the annotations is keyed by their value, and what they make (a
#: dataclass's fields and methods) is in the class's other attributes.
_CLASS_SKIP = frozenset(
    {
        "__dict__",
        "__weakref__",
        "__doc__",
        "__module__",
        "__qualname__",
        "__annotations__",
        "__annotate_func__",
        "__annotations_cache__",
        "__slotnames__",
        # pydantic's memo of each field's setter, filled by the first
        # assignment to that field: made from the fields, not held apart.
        "__pydantic_setattr_handlers__",
        "__firstlineno__",
        "__static_attributes__",
    }
)


def class_digest(cls: type, within: _Digester | None = None) -> str:
    """A digest of what *cls* holds: its attributes (methods by code, data
    by value) and the names of its bases."""
    return _class_digest_now(cls, within)


#: Set while a trace's items are digested for a call whose key holds the
#: classes of module data by their surface (`code_trace.build`): a class
#: met inside a value is digested by its surface too (`_class_digest`),
#: each method by name, since the methods that ran have code items of
#: their own.
SURFACE: contextvars.ContextVar[bool] = contextvars.ContextVar("cash_class_surface", default=False)


def remembered_class_digest(cls: type) -> str:
    """`class_digest` of *cls* on its own, remembered while the class's
    world stands still (`class_memo`): a call's trace names about a dozen
    classes, each digested again on every miss that records it and every
    check of it."""
    return _remembered(cls, "digest", lambda: _class_digest_now(cls, None))


def class_surface_digest(cls: type) -> str | None:
    """*cls* by its surface: its name, the names of its bases, and for it
    and each base of the user's what it holds, a plain method (a function
    with no closure of its own, or a staticmethod, classmethod, property or
    ``cached_property`` of one) by its name and kind only, not its code.
    A class met in what it holds is digested the same way. None when what
    it holds cannot be digested.

    For a call whose trace has a code item for every method that ran: an
    edit to a method no call ran leaves this digest alone, and one to a
    method that ran moves that method's item. Remembered as
    `remembered_class_digest` is."""
    if not isinstance(cls, type) or not is_user_type(cls) or "<locals>" in cls.__qualname__:
        # One no module of the user's holds (made by ``exec``, or inside a
        # function), which no trace item finds again by name.
        return None
    parts = []
    for klass in cls.__mro__:
        if is_user_type(klass):
            got = _remembered(klass, "surface", lambda klass=klass: _own_surface(klass))
            if got.startswith("opaque:"):
                return None
            parts.append(got)
        else:
            parts.append(f"{klass.__module__}.{klass.__qualname__}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=12).hexdigest()


def _own_surface(cls: type) -> str:
    """What *cls* itself holds, by `class_surface_digest`'s rules."""
    token = SURFACE.set(True)
    try:
        h = hashlib.blake2b(digest_size=12)
        d = _Digester(h)
        d.types.add(id(cls))
        d.open.add(id(cls))
        return _class_digest(cls, h, d, surface=True)
    except Exception:  # noqa: BLE001 - a class that breaks on being read
        return "opaque:" + cls.__qualname__
    finally:
        SURFACE.reset(token)


def _remembered(cls: type, kind: str, compute: Any) -> str:
    """*compute*'s digest of *cls*, remembered while its world stands
    still (`class_memo`), apart for each *kind* and for digests taken by
    the surface of the classes met inside (`SURFACE`)."""
    if not getattr(_STARTS, "cells", None):
        # Not while a call's start cells stand in for its functions' cells
        # (`start_cells`): the digest then is of that call's, not the class's.
        from ..decorator import class_memo

        if class_memo.ENABLED:
            proof = class_memo.proof_for(cls, "digest")
            if proof is not None:
                key = (id(cls),) if kind == "digest" and not SURFACE.get() else (id(cls), kind, SURFACE.get())
                got = class_memo.DIGESTS.get(key, proof)
                if got is class_memo.MISS or class_memo.VERIFY:
                    fresh = compute()
                    if got is not class_memo.MISS and got != fresh:
                        raise class_memo.ClassMemoMismatch(f"the {kind} of {cls.__qualname__} changed unseen")
                    class_memo.DIGESTS.put(key, proof, fresh)
                    return fresh
                return got
    return compute()


def _class_digest_now(cls: type, within: _Digester | None, *, surface: bool = False) -> str:
    h = hashlib.blake2b(digest_size=12)
    d = _Digester(h)
    if within is not None:
        d.types = within.types
        d.types.add(id(cls))
        d.shared = within.shared
        d.open = within.open
    opened = id(cls) not in d.open
    d.open.add(id(cls))
    try:
        return _class_digest(cls, h, d, surface=surface)
    finally:
        if opened:
            d.open.discard(id(cls))


def _plain_method_kind(value: Any) -> str | None:
    """The kind of *value* when it is a plain method of a class: a function
    whose only free variable is ``__class__`` (``super()``), or a
    staticmethod, classmethod, property or ``cached_property`` holding only
    such functions; else None. Run, such a function is found again by its
    qualified name and checked by a code item of its own; one with a
    closure is not, so it is digested by its code and cells."""

    def plain(fn: Any) -> bool:
        return type(fn) is types.FunctionType and set(fn.__code__.co_freevars) <= {"__class__"}

    if plain(value):
        return "def"
    if isinstance(value, (staticmethod, classmethod)):
        return type(value).__name__ if plain(value.__func__) else None
    if type(value) is property:
        fns = [f for f in (value.fget, value.fset, value.fdel) if f is not None]
        return "property" if fns and all(map(plain, fns)) else None
    if type(value) is functools.cached_property:
        return "cached_property" if plain(value.func) else None
    return None


def _class_digest(cls: type, h: Any, d: _Digester, *, surface: bool = False) -> str:
    from ..decorator.loop_leftovers import leftover_value, set_loop_leftovers

    h.update(",".join(f"{c.__module__}.{c.__qualname__}" for c in cls.__mro__).encode())
    try:
        attrs = sorted((k, v) for k, v in vars(cls).items() if k not in _CLASS_SKIP)
    except TypeError:
        attrs = [(k, v) for k, v in vars(cls).items() if k not in _CLASS_SKIP]
    d.shared = d.shared | _shared_codes(attrs)
    leftovers = set_loop_leftovers(cls, tuple(k for k, v in attrs if isinstance(k, str) and not callable(v)))
    for name, value in attrs:
        h.update(b"\x00" + str(name).encode())
        kind = _plain_method_kind(value) if surface else None
        if kind is not None:
            # By name: the code items of the call's trace check the
            # methods that ran.
            h.update(b"method:" + kind.encode())
            continue
        if name in leftovers:
            # What a class-body loop over a set left: which set, not which
            # element its iteration order ended on.
            value = leftover_value(vars(cls), name, leftovers)
        # Each by a digest of its own: one that turns out opaque part way
        # through left what it had written so far, and a dict filled in a
        # different order in another process (from a set) made a class
        # that had not changed read as changed every time.
        got = _sub_digest(value, d)
        h.update(b"opaque:" + type(value).__qualname__.encode() if got is None else got.encode())
    return h.hexdigest()


_MISSING = object()


def _shared_codes(attrs: list) -> frozenset[types.CodeType]:
    """The codes two or more of a class's functions run: wrappers a loop
    made from one ``def`` and set on the class (a delegate per method of
    another class, each closing over the name it forwards)."""
    seen: set[types.CodeType] = set()
    shared: set[types.CodeType] = set()
    for _name, value in attrs:
        fn = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        code = getattr(fn, "__code__", None) if type(fn) is types.FunctionType else None
        if code is None or not fn.__closure__:
            continue
        (shared if code in seen else seen).add(code)
    return frozenset(shared)


def candidates(obj: Any) -> Iterable[types.FunctionType]:
    """The functions *obj* may stand for: itself, what a staticmethod,
    classmethod, bound method, property, ``cached_property``, partial or JIT
    dispatcher holds,
    and what a ``functools.wraps`` wrapper wraps. Read off known types
    only: no attribute lookup runs on an object of the user's."""
    from ..jit_functions import jit_python_function

    stack = [obj]
    seen = 0
    while stack and seen < 16:
        seen += 1
        o = stack.pop()
        if type(o) is types.FunctionType:
            yield o
            wrapped = o.__dict__.get("__wrapped__") if o.__dict__ else None
            if wrapped is not None:
                stack.append(wrapped)
            # A stub ``exec`` built (``boltons.funcutils.wraps``) stands for
            # what it calls through its private globals too: its
            # ``__wrapped__`` skips the decorator's own wrapper.
            stack.extend(f for _, f in private_forwards(o))
        elif issubclass(type(o), (types.MethodType, staticmethod, classmethod)):
            stack.append(o.__func__)
        elif isinstance(o, property):
            stack.extend(f for f in (o.fget, o.fset, o.fdel) if f is not None)
        elif type(o) is functools.cached_property:
            # A getter a factory made in a closure and set on a class: found
            # through the class that holds the property.
            stack.append(o.func)
        elif isinstance(o, functools.partial):
            stack.append(o.func)
        elif issubclass(type(o), functools._lru_cache_wrapper):
            # ``@functools.cache``: a C object, its function kept as set.
            stack.append(o.__wrapped__)
        elif o is not None and not issubclass(type(o), (type, types.ModuleType)):
            jitted = jit_python_function(o)
            if jitted is not None:
                stack.append(jitted)
            else:
                stack.extend(_held_functions(o))


def private_forwards(fn: types.FunctionType) -> list[tuple[str, types.FunctionType]]:
    """``(name, function)`` for each function *fn* calls through globals
    of its own: a namespace ``exec`` built for it alone, not a module's.
    A signature-preserving decorator (``boltons.funcutils.wraps``,
    ``makefun``) compiles a stub that forwards to the decorator's wrapper
    held there (``_call``). Empty for a function of a module."""
    g = fn.__globals__
    if g.get("__file__"):
        return []
    module = sys.modules.get(g.get("__name__") or "")
    if getattr(module, "__dict__", None) is g:
        return []
    out = []
    for name in sorted(set(fn.__code__.co_names)):
        held = g.get(name)
        if type(held) is types.FunctionType and held is not fn:
            out.append((name, held))
    return out


#: Where a descriptor or wrapper of a library's keeps the function it stands
#: for (sklearn's ``available_if`` keeps ``fn``).
_HELD_FUNCTION_ATTRS = ("fn", "func", "__wrapped__", "fget")


def _held_functions(obj: Any) -> list[types.FunctionType]:
    """The plain functions *obj* keeps in its own ``__dict__`` under one of
    `_HELD_FUNCTION_ATTRS`: read from the instance dict, so no ``__getattr__``
    or property of its class runs."""
    try:
        d = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - an object with no dict of its own holds none
        return []
    if type(d) is not dict:
        return []
    return [f for f in (d.get(a) for a in _HELD_FUNCTION_ATTRS) if type(f) is types.FunctionType]


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
    if issubclass(type(obj), types.ModuleType):
        found = obj.__dict__.get(name, _MISSING)
        if found is _MISSING:
            raise AttributeError(name)
        return found
    return getattr(obj, name)


def is_builtin_name(name: str) -> bool:
    return hasattr(builtins, name)
