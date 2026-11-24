"""The fast hit path of ``@cash.cache``: a call met before, served without
building its key again, while the watchers prove nothing the key reads moved.

A hit costs cash a key build (the function's code, helpers, closure,
defaults and globals folded, the arguments hashed) and a store lookup, about
a hundred microseconds even for ``f(3)``. ``functools.lru_cache`` answers in
a tenth of a microsecond because it trusts the function to stay what it was.
Cash cannot trust that, but with the C extension (`cash._native`) it can
check it: CPython's dict, type and function watchers report every change to
the things a key reads, and the extension counts them in one number, the
*world epoch* (``cash/_c/cwatch.c``).

How a call gets here (`HitMemo.offer`, after an ordinary hit):

* The function's world is scanned once (`_Scan`): its code, the globals,
  closures, defaults, decorators and classes that code names, the helpers
  and cached callees cash's own analysis found, transitively. Every value
  met must be one whose content cannot change unseen: an atom, a tuple of
  them, code (a function, a class, a module), or a plain dict, which is
  watched. Anything else -- a list, an instance, an array, a generator --
  and the function keeps the ordinary path. The dicts are watched, the
  classes' version tags taken, the cells' contents remembered.
* The call itself must be an ordinary one: not nested in another cached
  call, no file tracker, no notebook statement handing digests, no
  ``key=``/``ignore=``, no ``dynamic_depends_on=``, no declared files, no
  randomness, no environment reads, a result the RAM tier holds and hands
  back the way this path does.
* The entry remembers the key, the RAM tier's own entry object and the
  *cash epoch*, which cash moves after every call that is not a plain hit
  (a miss stores, learns and may evict) and on every registration, clear
  and setting change.

A later call with arguments of the same kind (atoms, tuples of atoms, or
one big plain value or list of records the watchers already prove
unchanged, keyed by its marker or its token, `value_watch.tag`) is served from
the memo while both epochs, every remembered cell and every class tag are
what they were and the RAM tier still holds that very entry. A ttl'd entry
is checked against the clock and an entry that read files has them checked
on every hit, exactly as the ordinary hit does. Statistics, the call log,
the net-loss accounting and the RAM tier's access counts are kept the same.

``CASH_HITPATH=0`` turns the path off; ``CASH_HITPATH_VERIFY=1`` builds the
key the ordinary way behind every fast hit and raises `HitMemoMismatch` when
it differs. Without the extension there is no proof and no fast path.
"""

from __future__ import annotations

import abc
import ast
import builtins
import collections
import contextlib
import contextvars
import dataclasses
import datetime
import decimal
import dis
import enum
import functools
import io
import operator
import os
import pathlib
import pickle
import re
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from typing import TYPE_CHECKING, Any

from .. import _native, _weak
from .._clock import perf_counter as _perf_counter
from .._memo import CODE_OBJECTS, LruMemo
from .._type_memo import class_flags, class_mro, class_name
from .._user_attrs import named_attr, plain_getattr
from ..analysis import text_store
from ..analysis.helper_code import own_code_is_user
from ..jit_functions import is_jit_type, jit_python_function
from ..source_reading import own_source
from ..tracking.tracker_context import active_tracker
from . import library_reads
from .call_state import NESTED_CASH_SECONDS
from .key_values import left_out_whole
from .user_code import (
    cash_wrapped,
    is_cash_wrapper,
    is_user_class,
    is_user_code_object,
    own_package,
    unregistered_namespace_code,
)

if TYPE_CHECKING:
    from .cached_function import CachedFunction

__all__ = ["ENABLED", "VERIFY", "HitMemo", "HitMemoMismatch", "bump", "flush_all", "status"]

_cw = _native.cwatch
if _cw is not None and not hasattr(_cw, "scoped_cells"):
    _cw = None


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


#: Is the fast path on? Needs the extension; ``CASH_HITPATH=0`` turns it off.
ENABLED = _cw is not None and _flag("CASH_HITPATH", "1")
#: ``CASH_HITPATH_VERIFY=1``: every fast hit builds the key again and compares.
VERIFY = ENABLED and _flag("CASH_HITPATH_VERIFY", "0")

_fast_lookup = _cw.fast_lookup if ENABLED else None
if ENABLED:
    from . import warning_state as _warning_state

    _warning_state.watch(_cw)
_world_epoch = _cw.world_epoch if ENABLED else None
_core_epoch = _cw.core_epoch if ENABLED else None
_cash_epoch = _cw.cash_epoch if ENABLED else None

#: World-dict keys libraries set and put back around nearly every call:
#: ``warnings.filters`` and the display hooks (each ``catch_warnings``,
#: which pandas runs in ``DataFrame.to_string``), ``sys.stderr`` and
#: ``sys.stdout`` (each ``contextlib.redirect_stderr``). Once any scanned
#: code named one, each such write moved the world for every memo of the
#: process. A change to one moves only the memos whose scan named it
#: (`scoped_cells`, compared with the cells), and putting back what it
#: replaced brings them back (see `scoped` in ``cwatch.c``).
_SCOPED_NAMES = ("filters", "showwarning", "_showwarnmsg_impl", "stderr", "stdout")
if ENABLED:
    _cw.world_scope(_SCOPED_NAMES)


class HitMemoMismatch(AssertionError):
    """``CASH_HITPATH_VERIFY=1``: the fast path served a key the ordinary
    key build no longer gives."""


def bump() -> None:
    """Something cash keeps changed in a way any key may read (a clear, a
    registration, a setting): no remembered hit is served until it is met
    again through the ordinary path. Nor is a remembered state segment
    (`state_memo`)."""
    from . import state_memo

    state_memo.moved()
    if _cw is not None:
        _cw.bump_cash()


#: A function's state -> the memos whose key reads it (callers that reached
#: it as a cached callee): its miss drops their remembered hits too.
_DEPENDENTS: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


#: A cached function's wrapper -> its state, for the scan of a caller.
_STATE_OF: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def own_state(wrapper: Any, state: list | None) -> None:
    """Note that *wrapper*'s remembered hits live in *state*."""
    if state is not None:
        _STATE_OF[wrapper] = state


def read_by_others(state: list | None) -> bool:
    """Does another function's memo read *state* (a cached callee's)?"""
    return state is not None and bool(_DEPENDENTS.get(state))


def forget(state: list | None) -> None:
    """A call of the function that owns *state* was not a plain hit: it
    stored, learned or evicted. Its remembered hits, and those of the
    functions whose key reads it, are served again only once met through
    the ordinary path. A miss anywhere else leaves them be: what it changes
    in the world the watchers see, and an entry it evicts or replaces the
    RAM check sees."""
    if state is None:
        return
    state[S_ENTRIES] = {}
    callers = _DEPENDENTS.get(state)
    if callers:
        for memo in list(callers):
            memo.state[S_ENTRIES] = {}


#: The numpy module whose number types the extension keys by their bytes
#: (`numpy_scalars_keyed`), or None while none is.
_SCALARS_OF: list = [None]


def numpy_scalars_keyed() -> int:
    """Hand the extension numpy's fixed-size number types, once per numpy
    module loaded, so a memo key holds ``np.int64(3)`` as its exact type and
    the bytes of its value (iterating an array gives such scalars). Imports
    nothing: while numpy is not loaded no argument is one. How many types
    the extension took."""
    numpy = sys.modules.get("numpy")
    if numpy is None or numpy is _SCALARS_OF[0] or _cw is None or not hasattr(_cw, "set_scalar_types"):
        return 0
    from .._plain_data import numpy_scalar_types

    rows = []
    for t in numpy_scalar_types():
        try:
            size = numpy.dtype(t).itemsize
            a, b = (
                numpy.frombuffer(bytes(p), dtype=t)[0]
                for p in (((0,), (1,)) if t is numpy.bool_ else (range(1, size + 1), range(0x81, 0x81 + size)))
            )
            rows.append((t, a, a.tobytes(), b, b.tobytes()))
        except Exception:  # noqa: BLE001 - a type that will not say its bytes keeps the ordinary path
            continue
    _SCALARS_OF[0] = numpy
    return _cw.set_scalar_types(tuple(rows))


def epochs(spec: CachedFunction, args: tuple, kwargs: dict) -> tuple | None:
    """``(world, cash, entries, (memo key, gen))`` now for *spec* called
    with *args*, or None when the path is off: an ordinary hit is remembered
    only if none of them moved while its key was built. The memo key is
    taken here, not after the call: another thread may change a watched
    argument and call again in between. Its gen too (`cwatch.keyed`): the
    key part of a list of plain values is a digest of its content, which
    comes back when the content is put back, while the gen of each arm is
    new, so a hit keyed from the content in between is not filed under it."""
    if _world_epoch is None:
        return None
    # The warnings state a key folds: a change since the last look moves
    # the cash epoch now, not at the next fast hit, so a hit keyed under
    # the new state is not remembered under the old (`warning_state`).
    _cw.warn_check()
    if _SCALARS_OF[0] is not sys.modules.get("numpy"):
        numpy_scalars_keyed()
    state = spec.hit_state
    return _world_epoch(), _cash_epoch(), state[S_ENTRIES] if state is not None else None, _cw.keyed(args, kwargs)


#: Every memo with fast hits not yet handed to the net-loss accounting.
_MEMOS: weakref.WeakSet[HitMemo] = _weak.WeakSet()


def flush_all() -> None:
    """Hand every memo's pending fast hits to the accounting (`HitMemo.flush`)."""
    for memo in list(_MEMOS):
        memo.flush()


def drain_log() -> None:
    """Write the fast hits' call-log records into their logs (`CallLog`)."""
    if _cw is None:
        return
    for owner, template, timestamp, seconds in _cw.take_log():
        owner.fast_log.append((template, timestamp, seconds))


def status() -> dict:
    """The fast path's counters, for tests and ``%cash_status``."""
    if _cw is None:
        return {"enabled": False}
    found = dict(_cw.world_info())
    found["enabled"] = ENABLED
    found["verify"] = VERIFY
    return found


# -- the scan --------------------------------------------------------------


class Ineligible(Exception):
    """The function's world holds something the watchers cannot vouch for."""


_ATOMS = frozenset(
    {type(None), bool, int, float, complex, str, bytes, range, slice, type(Ellipsis), type(NotImplemented)}
)

#: Objects of C types that cannot change after they are made, keyed by
#: their class or by what they were built with.
_FIXED_C = frozenset(
    {
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        types.ClassMethodDescriptorType,
        types.GetSetDescriptorType,
        types.MemberDescriptorType,
        operator.itemgetter,
        operator.attrgetter,
        operator.methodcaller,
        re.Pattern,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        datetime.timezone,
        decimal.Decimal,
    }
)

#: Locks carry no data: a key reads ``threading.Lock()`` and
#: ``threading.RLock()`` by their type alone (`key_values.SYNC_TYPES`), and
#: whether one is held is no part of any key. Which lock a name or an
#: attribute holds is watched as the dict holding it. Exact types: a
#: subclass of the user's own can carry anything.
_LOCK_TYPES = frozenset({type(threading.Lock()), type(threading.RLock())})

#: pathlib's own path classes: a path's parts are fixed when it is made
#: (what it caches besides is computed from them). A subclass of the
#: user's own can carry anything, so only these exact types.
_PATH_TYPES = frozenset(
    {
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.Path,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    }
)

#: The typing objects whose content is a few names fixed when they are
#: made (``Optional``, ``Any``, ``int | None``, ``Literal["a"]``,
#: ``list[int]``, ``TypeVar("T")``): what an alias holds besides
#: (`_TYPING_PARTS`) is looked into as any value. Not a ``ForwardRef`` (it
#: caches what it evaluates to) nor a ``type X = ...`` alias (its value is
#: evaluated lazily, from the names it reads then).
_TYPING_PARTS = ("__origin__", "__args__", "__metadata__", "__supertype__")
_TYPING_LAZY = frozenset(t for t in (typing.ForwardRef, getattr(typing, "TypeAliasType", None)) if isinstance(t, type))
_TYPING_C = frozenset(t for t in (getattr(types, "UnionType", None), types.GenericAlias) if isinstance(t, type))

_LRU_WRAPPER = type(functools.lru_cache(maxsize=1)(len))

#: ``abc``'s per-class registration record (``cls._abc_impl``): what it
#: holds moves ``abc.get_cache_token()`` on every registration.
_ABC_DATA = type(getattr(abc.ABC, "_abc_impl", None))

_HEAPTYPE = 1 << 9
_METACLASSES = (type, abc.ABCMeta)
#: More objects than this in one function's world: not worth watching.
_SCAN_LIMIT = 20_000
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A line of a ``def``'s source that is the def itself, not a decorator.
_DEF_LINE = re.compile(r"\s*(async\s+)?def\s")


def _is_cash(v: Any, t: type) -> bool:
    """An object of cash's own (`Cash`, its decorator, its modules)."""
    owner = v if t is types.FunctionType or isinstance(v, type) else t
    module = plain_getattr(owner, "__module__", None)
    return isinstance(module, str) and (module == "cash" or module.startswith("cash."))


def _code_names(code: types.CodeType, into: set[str]) -> None:
    """The names *code* and its nested scopes read by name: globals and
    attributes, and the identifier-like strings (``getattr(m, "x")``,
    ``globals()["x"]``). Not its locals: a module-level loop rebinding ``i``
    leaves a function whose own ``i`` is local alone."""
    into.update(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _code_names(const, into)
        elif type(const) is str and const.isidentifier():
            into.add(const)


#: `_signature_names` by source text: a scan of a function's world parsed
#: each function's source again, most of what a scan cost.
_SIGNATURE_NAMES: LruMemo[str, frozenset[str]] = LruMemo(CODE_OBJECTS)


#: A source this long or shorter is parsed (`_parse_signature_names`)
#: rather than looked up in the store while this process has not read the
#: store's file: a few lines parse in tens of microseconds, and the file, read
#: whole on the first lookup, held every answer of every earlier process.
_PARSED_FROM = 1000
#: How many characters of source a process parses so (`_PARSED_FROM`) before
#: it reads the store after all: ~4 ms of parsing.
_PARSED_BUDGET = [10_000]


def _signature_names(source: str) -> frozenset[str]:
    """The names the annotations in *source*'s ``def`` line read (a key may
    resolve them); every identifier of *source* when it does not parse."""
    found = _SIGNATURE_NAMES.get(source)
    if found is None and len(source) <= min(_PARSED_FROM, _PARSED_BUDGET[0]) and not text_store.held("signature-names"):
        # The same answer the store keeps: a function of the text alone.
        _PARSED_BUDGET[0] -= len(source)
        found = _SIGNATURE_NAMES[source] = frozenset(_parse_signature_names(source))
    if found is None:
        # A function of the text alone, kept across processes (`text_store`).
        found = _SIGNATURE_NAMES[source] = text_store.kept(
            "signature-names", source, "", lambda: frozenset(_parse_signature_names(source)), sorted, _names_decoded
        )
    return found


def _names_decoded(value: Any) -> frozenset[str]:
    if not isinstance(value, list) or not all(type(v) is str for v in value):
        raise ValueError("not a list of names")
    return frozenset(value)


def _parse_signature_names(source: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return set(_IDENT.findall(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            parts = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
            for part in (*(p.annotation for p in parts if p is not None), node.returns):
                if part is not None:
                    found.update(n.id for n in ast.walk(part) if isinstance(n, ast.Name))
                    found.update(
                        n.value for n in ast.walk(part) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    )
    return found


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


#: What `jit_functions.jit_options` reads of a dispatcher besides its class.
_JIT_READS = frozenset(
    {"targetoptions", "locals", "_dispatcher", "gufunc_builder", "kernel_ir", "mode", "options", "_sig"}
)

#: A class defining one of these reads or pickles its instances its own way.
_OWN_ACCESS = frozenset(
    {
        "__getattr__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
    }
)

#: Modules whose objects change as the collector runs (a
#: ``WeakValueDictionary`` loses entries, a ``WeakSet`` too): watched, each
#: change would move the world for every memo.
_VOLATILE_MODULES = frozenset({"weakref", "_weakrefset"})

#: Library classes whose own reduction pickles the instance's ``__dict__``
#: as ``object``'s does (numba's types: ``Type.__reduce__`` wraps it).
_PLAIN_REDUCE = frozenset({("numba.core.types.abstract", "Type")})

#: The instructions of a getter that only reads attributes off its argument.
_GETTER_OPS = frozenset(
    {"RESUME", "LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK", "LOAD_ATTR", "RETURN_VALUE", "NOP", "NOT_TAKEN"}
)
_GETTER_FLAGS = 0x04 | 0x08 | 0x20 | 0x80 | 0x200  # varargs, varkw, generator, coroutine, async generator


def _plain_getter(prop: Any) -> frozenset[str] | None:
    """The attribute names a property's getter reads, when all it does is
    return an attribute chain off the object it is read from
    (``return self._n_jobs``); None for any other descriptor or getter."""
    if type(prop) is not property:
        return None
    f = prop.fget
    if type(f) is not types.FunctionType or f.__closure__ or f.__defaults__ or f.__kwdefaults__:
        return None
    code = f.__code__
    if code.co_argcount != 1 or code.co_kwonlyargcount or code.co_flags & _GETTER_FLAGS or code.co_freevars:
        return None
    first = code.co_varnames[0]
    try:
        ins = [i for i in dis.get_instructions(code) if i.opname not in ("CACHE", "EXTENDED_ARG")]
    except Exception:  # noqa: BLE001 - code dis cannot read: no plain getter
        return None
    for op in ins:
        if op.opname not in _GETTER_OPS:
            return None
        if op.opname.startswith("LOAD_FAST") and op.argval != first:
            return None
        if op.opname == "LOAD_ATTR" and sys.version_info >= (3, 12) and (op.arg or 0) & 1:
            return None  # a method loaded to call
    return frozenset(code.co_names)


class _Scan:
    """What one cached function's key can read, made watched (see module doc)."""

    def __init__(self, root_module: str | None, root: Any = None) -> None:
        self.root_module = root_module
        #: `watch_helper`: what one helper's globals fold reads, not the
        #: world below the code it names (see `value`).
        self.shallow = False
        self.shallow_pkg: str | None = None
        #: In shallow mode, scanning a module's attributes: a function read
        #: as ``mod.f`` has the globals it reads folded one level deep
        #: (`ModuleAttrs._helper_constant_parts`).
        self.in_module = False
        #: The cached function itself: the names on its decorator lines
        #: are not read by its key (`GlobalReads.decorator_global_names`
        #: is folded for a function held inside a decorator only).
        self.root = root
        self.seen: set = set()
        self.keep: list = []
        self.cells: list = []
        self.callees: list = []
        self.types: dict[int, tuple[type, int]] = {}
        self.names: set[str] = set()
        self.dicts: list[tuple[dict, bool]] = []
        #: The user classes met: what an ABC registers and the classes
        #: defined below them move no watched dict (`family`).
        self.classes: list[type] = []
        #: The library classes met, MRO included: watched, and only the
        #: attributes the scanned code names looked into (`finish`).
        self.library_classes: list[type] = []
        #: ``(read, value)``: state no watcher sees that is watched by
        #: reading it again (`_Family.holds`): the abc cache token, sklearn's
        #: settings.
        self.probes: dict[Any, Any] = {}
        #: The state memo's scan: data no watcher sees (a list, a numpy
        #: array) is read again on every reuse instead (`ContentProbe`).
        self.content = False
        #: The state memo's scan: the names of the properties of a library
        #: metaclass met that answer from more than the class holds
        #: (`_metaclass_ok`); the scan fails if the scanned code reads one
        #: (`finish`).
        self.meta_props: set[str] = set()
        self.count = 0
        #: The state memo's scan: may a global the key leaves out whole be
        #: watched for that alone (`global_value`)? Not with a hasher
        #: registered.
        self.left_out_ok = False

    def _first(self, key: Any, obj: Any) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        self.keep.append(obj)
        self.count += 1
        if self.count > _SCAN_LIMIT:
            raise Ineligible("too much to watch")
        return True

    def global_value(self, v: Any, names: frozenset) -> None:
        """A value a global or a module attribute holds, as the key reads it
        (`GlobalValues.safe_global_hash`). The state memo's scan: one the
        key leaves out whole (`key_values.left_out_whole`) is watched for
        what keeps it left out, its class and its ``__dict__``, and not
        looked into. Rebinding the name moves the namespace watched already."""
        if self.left_out_ok and self.content and left_out_whole(v):
            if self._first(("left out", id(v)), v):
                self.watch(object.__getattribute__(v, "__dict__"), True)
                self._keeps_class_and_dict(v)
            return
        self.value(v, names)

    def watch(self, d: dict, unkeyed: bool) -> None:
        if type(d) is not dict:
            raise Ineligible("a namespace that is not a plain dict")
        self.dicts.append((d, unkeyed))

    def value(self, v: Any, names: frozenset) -> None:
        t = type(v)
        if t in _ATOMS:
            return
        if t is types.ModuleType:
            name = plain_getattr(v, "__name__", "")
            if name == "cash" or (isinstance(name, str) and name.startswith("cash.")):
                return
            if self._first(("module", id(v), id(names)), v):
                self.module(v, names)
            return
        if self.shallow and (t is types.FunctionType or (isinstance(v, type) and not self._folds_class(v))):
            # Once per way it is met: read as a module's attribute it is
            # also followed one level (`_named`), whichever way came first.
            if self._first(("named", id(v), self.in_module), v):
                self._named(v, t, names)
            return
        if not self._first(id(v), v):
            return
        if _is_cash(v, t):
            return  # cash's own objects: what of them a key reads moves the cash epoch
        shallow, self.shallow = self.shallow, False
        try:
            self._inside(v, t, names)
        finally:
            self.shallow = shallow

    def _inside(self, v: Any, t: type, names: frozenset) -> None:
        if t is tuple or t is frozenset:
            for item in v:
                self.value(item, names)
        elif t is dict:
            self.watch(v, True)
            for k, item in list(v.items()):
                self.value(k, names)
                self.value(item, names)
        elif t is types.FunctionType:
            self.function(v)
        elif isinstance(v, type):
            self.klass(v)
        elif t is types.BuiltinFunctionType:
            owner = v.__self__
            if self.content and _global_stream(owner):
                self.value(type(owner), names)
            elif owner is not None and not issubclass(type(owner), types.ModuleType):
                self.value(owner, names)
        elif t is types.MethodType:
            self.value(v.__func__, names)
            if self.content and _global_stream(v.__self__):
                self.value(type(v.__self__), names)
            else:
                self.value(v.__self__, names)
        elif self.content and _library_cython(v, t):
            return
        elif self.content and self._user_cython(v, t, names):
            return
        elif t is types.MethodWrapperType:
            self.value(v.__self__, names)
        elif t is staticmethod or t is classmethod:
            self.value(v.__func__, names)
        elif t is property:
            for part in (v.fget, v.fset, v.fdel):
                if part is not None:
                    self.value(part, names)
        elif t is functools.partial:
            if plain_getattr(v, "__dict__", None):
                raise Ineligible("a partial with attributes")
            self.value(v.func, names)
            self.value(v.args, names)
            self.watch(v.keywords, True)
            for item in v.keywords.values():
                self.value(item, names)
        elif t in _FIXED_C or t in _LOCK_TYPES or (t.__name__ == "ufunc" and t.__module__ == "numpy"):
            return
        elif t.__name__ == "_ArrayFunctionDispatcher" and t.__module__ == "numpy":
            # `np.sum`, `np.mean`: a C wrapper whose implementation is fixed
            # when it is made (`_implementation` cannot be set); what it
            # runs is that function.
            self.value(v._implementation, names)
        elif t in _PATH_TYPES:
            return
        elif t is _LRU_WRAPPER:
            # ``@functools.lru_cache``: the function it calls is fixed when
            # it is made; what it remembers is no part of any key. Its
            # attributes (``__wrapped__``, set by ``update_wrapper``) are.
            attrs = plain_getattr(v, "__dict__", None)
            if type(attrs) is not dict:
                raise Ineligible("an lru_cache without attributes")
            self.watch(attrs, True)
            for item in list(attrs.values()):
                self.value(item, names)
        elif t in _TYPING_C or (t.__module__ == "typing" and t not in _TYPING_LAZY):
            self._typing(v, names)
        elif t is dataclasses.Field:
            self._field(v, names)
        elif t is _ABC_DATA:
            self.probes[abc.get_cache_token] = abc.get_cache_token()
        elif self._sklearn_config(v):
            return
        elif self.content and self._fixed_kind(v, t, names):
            return
        elif self.content and self._jit_dispatcher(v, t, names):
            return
        elif self.content and self._fixed_value(v, t, names):
            return
        elif self.content and (probe := ContentProbe.of(v)) is not None:
            self.probes[probe] = probe.digest
        elif self.content and self._library_instance(v, t, names):
            return
        elif self.content and t is list:
            # A list of what is watched otherwise (sklearn's parameter
            # constraints): which items it holds is read again before each
            # reuse, by identity; each item is looked into as any value.
            probe = ContentProbe(v, _Items.of)
            for item in list(v):
                self.value(item, names)
            self.probes[probe] = probe.digest
        else:
            raise Ineligible(f"a {t.__module__}.{t.__qualname__}")

    def _fixed_kind(self, v: Any, t: type, names: frozenset) -> bool:
        """The state memo's scan: a member of an enum, or a number, string,
        bytes or tuple of a subclass without attributes of its own. True when *v*
        is one and is watched now.

        An enum member is its class's (`_klass` watches the class, the
        members' table among its attributes) and what it holds is its
        ``__dict__`` (``_value_``, ``_name_``), a plain dict watched like
        a function's attributes: setting anything on it moves the world.

        A subclass of ``int``, ``float``, ``str``, ``bytes`` or ``tuple`` (a
        named tuple) holds a value fixed when it is made, a tuple's items
        looked into as any value; what else a key can read of it is its class,
        scanned as any class, and attributes set on the instance, which no
        watcher sees: only one with none is taken, and a probe
        (`_attribute_count`) checks before every reuse that it still has
        none. Not one with ``__slots__`` of its own (what they hold no
        watcher sees either)."""
        if not t.__flags__ & _HEAPTYPE:
            return False
        if type(t) is enum.EnumMeta and isinstance(v, enum.Enum):
            held = plain_getattr(v, "__dict__", None)
            if type(held) is not dict:
                return False
            self.value(t, names)
            self.watch(held, True)
            for item in list(held.values()):
                self.value(item, names)
            return True
        if not issubclass(t, (int, float, str, bytes, tuple)) or issubclass(t, enum.Enum):
            return False
        for k in t.__mro__:
            slots = vars(k).get("__slots__")
            if isinstance(slots, str):
                slots = (slots,)
            if slots and any(name not in ("__dict__", "__weakref__") for name in slots):
                return False
        held = plain_getattr(v, "__dict__", None)
        if held is not None:
            if type(held) is not dict:
                return False
            if not held:
                self.probes[ContentProbe(v, _attribute_count)] = 0
            else:
                # Attributes set on it (scverse's ``Deprecation``, a ``str``
                # carrying the version it was deprecated in): watched as an
                # instance's (`_library_instance`).
                self.watch(held, True)
                for item in list(held.values()):
                    self.value(item, names)
                self._keeps_class_and_dict(v)
        self.value(t, names)
        if isinstance(v, tuple):
            for item in tuple.__iter__(v):
                self.value(item, names)
        return True

    def _user_cython(self, v: Any, t: type, names: frozenset) -> bool:
        """The state memo's scan: a Cython function of the user's own build
        (`_library_cython` takes a library's). True when *v* is one and is
        watched now.

        A key names it by the Cython code it can reach in its ``.pyx``, or
        by its whole built file where that cannot be trusted
        (`code_digest.compiled_identity`, `cython_function_digest`), worked
        out once per loaded module: a process cannot load a rebuilt file in
        place of the one it runs. What else the key reads of it is its
        module and that module's file, its names, its code and its
        attributes (``__dict__``: a mark of cash's, a JIT's). No watcher
        sees any of that nor an edit to its files, so a probe reads it
        again before every reuse (`_cython_reads`): the same module, file,
        names, code and ``__dict__`` by identity, and every file its key
        was worked out from (`cython_digest.keyed_files`) with the same
        size and times. The ``__dict__`` is watched as a function's
        attributes are, and what it holds looked into as any value."""
        from ..code_digest import extension_file_digest
        from ..cython_digest import is_cyfunction

        if not is_cyfunction(v) or extension_file_digest(v) is None:
            return False
        held = plain_getattr(v, "__dict__", None)
        if type(held) is not dict:
            return False
        probe = ContentProbe(v, _cython_reads)
        self.watch(held, True)
        for item in list(held.values()):
            self.value(item, names)
        self.probes[probe] = probe.digest
        return True

    def _jit_dispatcher(self, v: Any, t: type, names: frozenset) -> bool:
        """The state memo's scan: a numba dispatcher (``@njit``). True when
        *v* is one and is watched now.

        A key reads it as code (`jit_functions`): the Python function it
        compiles (``py_func``), walked as any function met, and its compile
        options (`jit_options`: ``targetoptions``, ``locals``, and what a
        ``_dispatcher``, ``gufunc_builder``, ``kernel_ir``, ``mode``,
        ``options`` or ``_sig`` would add, none of which a dispatcher
        has). All of it is in its ``__dict__``, a plain dict numba never
        writes to after the dispatcher is made (compiling fills tables the
        dict holds, not the dict): watched whole, so rebinding
        ``py_func`` or ``targetoptions`` moves the world, and the options'
        own dicts are watched as any dict met. Its class is watched as a
        library class (an attribute added to it is a type event), and a
        probe checks the instance keeps its class and ``__dict__``. Only a dispatcher whose
        class reads attributes as ``object`` does: no ``__getattr__`` or
        ``__getattribute__`` of its own, none of those names a class
        attribute. numba's other wrappers (``DUFunc``, ``GUFunc``,
        ``CFunc``, a stencil) are not taken."""
        if not t.__flags__ & _HEAPTYPE or not is_jit_type(t):
            return False
        fn = jit_python_function(v)
        held = plain_getattr(v, "__dict__", None)
        if fn is None or type(held) is not dict or held.get("py_func") is not fn or "targetoptions" not in held:
            return False
        for k in t.__mro__:
            ns = vars(k)
            if k is not object and ("__getattr__" in ns or "__getattribute__" in ns):
                return False
            if any(name in ns for name in _JIT_READS):
                return False
        self.value(t, names)
        self.watch(held, True)
        for name, item in list(held.items()):
            if type(item) is types.CodeType:
                continue  # ``__code__``: fixed, and ``py_func``'s is what a key reads
            if name in _JIT_READS or name == "py_func" or _is_dunder(name):
                self.value(item, names)
        self._keeps_class_and_dict(v)
        return True

    def _fixed_value(self, v: Any, t: type, names: frozenset) -> bool:
        """The state memo's scan: a value of a library that cannot change
        once made, or a context variable. True when *v* is one and is
        watched now.

        * a numpy scalar of numpy's own type (``np.float64(1.5)``,
          ``np.bool_(True)``), not a ``void`` (a view into a structured
          array) or an ``object_``;
        * pandas' ``NaT`` and its axis descriptors (``DataFrame.index``,
          an ``AxisProperty``: which axis, fixed);
        * a dataclass's parameters (``__dataclass_params__``: the flags
          the class was made with);
        * a ``ContextVar``: its name and default are fixed; what it holds
          in the context of a call is read again before each reuse, by
          identity, and looked into as any value."""
        if t is contextvars.ContextVar:
            probe = ContentProbe(v, _context_value)
            held = probe.digest.items[0]
            if held is not _NOTHING:
                self.value(held, names)
            self.probes[probe] = probe.digest
            return True
        if t is getattr(dataclasses, "_DataclassParams", None):
            return True
        module = t.__module__
        if module == "pandas._libs.properties" and t.__name__ == "AxisProperty":
            return True
        pd = sys.modules.get("pandas")
        if pd is not None and v is getattr(pd, "NaT", None):
            return True
        if module == "numpy" and not t.__flags__ & _HEAPTYPE:
            np = sys.modules.get("numpy")
            return np is not None and isinstance(v, np.generic) and not isinstance(v, (np.void, np.object_))
        return False

    def _library_instance(self, v: Any, t: type, names: frozenset) -> bool:
        """The state memo's scan: an instance of a library's class whose
        whole state is its ``__dict__`` -- a library's settings object.
        True when *v* is one and is watched now.

        A key reads such an object by the attribute chains a reader reads
        off it (`global_values.chain_end`: what its ``__dict__`` holds), or
        whole, by its class and its ``__dict__`` (a class keyed by name, a
        pickle of the attributes), never through a property. Its
        ``__dict__`` is watched whole, a write to any attribute moving the
        world -- a property's setter (``settings.n_jobs = 4`` storing
        ``_n_jobs``) as an assignment -- and every value it holds is
        looked into as any value met. Its classes are watched as library
        classes and a probe checks it keeps its class and its ``__dict__``.

        Taken only when that is all there is to it: a class of a library,
        made by ``type``, ``abc.ABCMeta`` or a library's metaclass that
        reads attributes as ``type`` does (`_metaclass_ok`), with no builtin base but
        ``object`` (whose storage no watcher sees), no ``__slots__`` but
        ``__dict__`` and ``__weakref__``, no ``__getattr__`` or
        ``__getattribute__`` that could answer a read from elsewhere, no
        reduction of its own (``__reduce__``, ``__getstate__``: a pickle of
        it is its ``__dict__``), and every property or other descriptor the
        scanned code names a getter that only reads attributes off the
        object (`_plain_getter`), or a method. Not a weak mapping or set
        (`_VOLATILE_MODULES`), which the collector changes."""
        if not t.__flags__ & _HEAPTYPE or is_user_code_object(t) or t.__module__ in _VOLATILE_MODULES:
            return False
        if type(t) not in _METACLASSES and not self._metaclass_ok(t):
            return False
        for k in t.__mro__:
            if k is object:
                continue
            if not k.__flags__ & _HEAPTYPE:
                return False
            ns = vars(k)
            if any(name in ns for name in _OWN_ACCESS) and (k.__module__, k.__qualname__) not in _PLAIN_REDUCE:
                return False
            slots = ns.get("__slots__")
            if isinstance(slots, str):
                slots = (slots,)
            if slots and any(name not in ("__dict__", "__weakref__") for name in slots):
                return False
            for name in names.intersection(ns) if len(names) < len(ns) else [n for n in ns if n in names]:
                attr = ns[name]
                if type(attr) is property:
                    if _plain_getter(attr) is None:
                        return False
                elif hasattr(type(attr), "__get__") and not isinstance(
                    attr, (types.FunctionType, staticmethod, classmethod)
                ):
                    return False
        try:
            held = object.__getattribute__(v, "__dict__")
        except AttributeError:
            return False
        if type(held) is not dict:
            return False
        self.value(t, names)
        self.watch(held, True)
        for item in list(held.values()):
            self.value(item, names)
        self._keeps_class_and_dict(v)
        return True

    def _keeps_class_and_dict(self, v: Any) -> None:
        """Check before each reuse that *v* keeps its class and its
        ``__dict__``: assigning either (``obj.__dict__ = {...}``) moves no
        watcher, and the dict watched would no longer be its."""
        probe = ContentProbe(v, _class_and_dict)
        self.probes[probe] = probe.digest

    def _typing(self, v: Any, names: frozenset) -> None:
        """A typing form or alias: fixed, but for what it was built from."""
        held = plain_getattr(v, "__dict__", None)
        for name in _TYPING_PARTS:
            # From the instance itself: an alias forwards other attributes
            # to its origin, and a type variable evaluates its bound lazily.
            part = held.get(name) if type(held) is dict else _c_part(v, name)
            if part is not None:
                self.value(part, names)

    def _field(self, f: dataclasses.Field, names: frozenset) -> None:
        """A dataclass field (``__dataclass_fields__``): set while its class
        is made and not after; what it holds is looked into as any value."""
        for part in (f.type, f.default, f.default_factory):
            if part is not dataclasses.MISSING:
                self.value(part, names)
        if f.metadata:
            self.value(f.metadata, names)  # a mapping proxy over a dict no watcher sees

    def _sklearn_config(self, v: Any) -> bool:
        """sklearn's settings (``sklearn._config._threadlocal``, read by
        ``get_config``): watched by reading them again."""
        config = sys.modules.get("sklearn._config")
        if config is None or v is not getattr(config, "_threadlocal", None):
            return False
        get_config = getattr(config, "get_config", None)
        if get_config is None:
            return False
        self.probes[get_config] = get_config()
        return True

    def module(self, mod: types.ModuleType, names: frozenset) -> None:
        md = plain_getattr(mod, "__dict__", None)
        if type(md) is not dict:
            raise Ineligible("a module without a plain namespace")
        self.watch(md, False)
        # Both scans: a helper's globals fold is part of the key too.
        library = self._library_module(mod)
        in_module, self.in_module = self.in_module, self.shallow
        try:
            for name in names:
                if name in md:
                    if mod is os and name in _OS_ENVIRON:
                        # ``os.environ.get("X")``: a key folds the variables
                        # read by name, never the mapping read through ``os``
                        # (`ModuleAttrs`: a library's object is no data). A
                        # write to another variable moves nothing it reads.
                        continue
                    if mod is sys and name == "modules":
                        # ``sys.modules``: never plain data (it holds
                        # ``sys``), so a key folds nothing of it
                        # (`module_attrs._library_attr_parts`); which
                        # module a name is bound to is watched apart
                        # (`watch_modules`). Walked, it is every module
                        # loaded, and one of a class of its own
                        # (``six.moves``) refused the whole world.
                        continue
                    if library and _unread_object(md[name]):
                        # ``logging._handlers``: read through a library's
                        # module, an object of the library's is no data a
                        # key folds (`module_attrs._library_attr_parts`),
                        # whatever it holds; rebinding the name moves the
                        # module's watched namespace.
                        continue
                    self.global_value(md[name], names)
        finally:
            self.in_module = in_module

    def _library_module(self, mod: types.ModuleType) -> bool:
        """Is *mod* a library's module for every reader the key can have:
        not the user's, and not in the package of the cached function (a
        library's own function decorated reads its package as its own)."""
        from ..install_paths import is_user_code_module

        root_pkg = (self.root_module or "").partition(".")[0]
        name = plain_getattr(mod, "__name__", None)
        if not isinstance(name, str) or (root_pkg and name.partition(".")[0] == root_pkg):
            return False
        return not is_user_code_module(mod)

    def _one_level(self, f: types.FunctionType) -> None:
        """The globals *f*'s code names, at the depth a helper's own are."""
        own: set[str] = set()
        _code_names(f.__code__, own)
        self.names.update(own)
        names = frozenset(own)
        g = f.__globals__
        self.watch(g, False)
        # A library's function is keyed by its name (`code_ref_part`): an
        # object of the library's its code names is no data a key folds, as
        # read through its module (`module`). Python 3.14's ``warnings``
        # reads its context from a ``ContextVar`` and a settings object.
        library = not _users(f, self.root_module)
        for name in names:
            if name in g:
                if library and _unread_object(g[name]):
                    continue
                self.value(g[name], names)

    def _named(self, v: Any, t: type, names: frozenset) -> None:
        """Shallow mode: a function or library class a helper's global or
        module attribute names. Its globals fold keys it by name
        (`GlobalValues.code_ref_part`), what that code reads being the
        code's own fold; but it also reads, of a function, its names, the
        attributes stored on it (``scale.k``) and, read as ``mod.f``, the
        globals it reads; of a library class, the attributes the helper
        names (``lib.Class.ATTR``) and its name. Those are watched; the
        rest of its world is not."""
        shallow, in_module, self.shallow, self.in_module = self.shallow, self.in_module, False, False
        try:
            if t is types.FunctionType:
                self.cells.append((v, (v.__name__, v.__qualname__)))
                attrs = v.__dict__
                self.watch(attrs, True)
                for name in names:
                    if name in attrs:
                        self.value(attrs[name], names)
                if in_module:
                    target = cash_wrapped(v)
                    target = jit_python_function(target) or target
                    if type(target) is types.FunctionType and self._first(("one level", id(target)), target):
                        self.shallow = True
                        self._one_level(target)
                return
            if not (v.__flags__ & _HEAPTYPE):
                return  # a builtin class: nothing in it changes
            for k in v.__mro__:
                if not (k.__flags__ & _HEAPTYPE):
                    continue
                if id(k) not in self.types:
                    tag = _cw.world_watch_type(k)
                    if tag is None:
                        raise Ineligible("a class without a version tag")
                    self.types[id(k)] = (k, tag)
                held = vars(k)
                for name in names:
                    if name in held:
                        self.value(held[name], names)
        finally:
            self.shallow, self.in_module = shallow, in_module

    def function(self, f: types.FunctionType) -> None:
        if is_cash_wrapper(f):
            # Another cached function, or this one called by name: its own
            # state is cash's (a registration, a setting) or its code.
            callee = _STATE_OF.get(plain_getattr(f, "__func__", f))
            if callee is not None:
                self.callees.append(callee)
            inner = plain_getattr(f, "__wrapped__", None)
            if inner is not None:
                self.value(inner, frozenset())
            return
        if not _users(f, self.root_module):
            # Library code: keyed by its name, which the module watch covers.
            return
        # Its names: setting ``__name__`` moves the key (`renamed`) but no
        # function watcher, so a fast hit checks them as it does the cells.
        self.cells.append((f, (f.__name__, f.__qualname__)))
        code = f.__code__
        own: set[str] = set()
        _code_names(code, own)
        decorated: set[str] = set()
        try:
            source = own_source(f)
        except Exception:  # noqa: BLE001 - no source: the bytecode names it reads
            source = ""
        # The root's decorators ran once, at its definition: what they name
        # now changes nothing its key reads. ``@cache(level=10)`` over a
        # library function kept every call of it off this path, for a
        # counter deep in the decorator's own machinery.
        if source and f is not self.root:
            for line in source.splitlines():
                if _DEF_LINE.match(line):
                    break
                decorated.update(_IDENT.findall(line))
        if source:
            self.names.update(_signature_names(source))
        self.names.update(own)
        self.names.update(decorated)
        names = frozenset(own | decorated)
        g = f.__globals__
        self.watch(g, False)
        b = f.__builtins__
        if type(b) is dict:
            self.watch(b, False)
        for name in names:
            if name in g:
                if g is _OS_GLOBALS and name in _OS_ENVIRON:
                    continue  # ``os.getenv``'s ``environ``: as in `module`
                self.global_value(g[name], names)
            elif type(b) is dict and name in b:
                self.value(b[name], names)
        # Defaults, closure and attributes are read in full: a default
        # function's own reads are folded with the function's (`_default_parts`).
        shallow, self.shallow = self.shallow, False
        try:
            self._held_by(f, names)
        finally:
            self.shallow = shallow

    def _held_by(self, f: types.FunctionType, names: frozenset) -> None:
        for default in f.__defaults__ or ():
            self.value(default, names)
        kw = f.__kwdefaults__
        if kw:
            self.watch(kw, True)
            for default in kw.values():
                self.value(default, names)
        for cell in f.__closure__ or ():
            try:
                held = cell.cell_contents
            except ValueError:
                raise Ineligible("an empty closure cell") from None
            self.cells.append((cell, held))
            self.value(held, names)
        attrs = f.__dict__
        self.watch(attrs, True)
        for item in list(attrs.values()):
            self.value(item, names)

    def _folds_class(self, cls: type) -> bool:
        """Does a helper's globals fold read into *cls* (its data and its
        methods' reads), rather than key it by name?"""
        return is_user_class(cls, self.shallow_pkg) or is_user_code_object(cls)

    def klass(self, cls: type) -> None:
        shallow, self.shallow = self.shallow, False
        try:
            self._klass(cls)
        finally:
            self.shallow = shallow

    def _klass(self, cls: type) -> None:
        if not (class_flags(cls) & _HEAPTYPE) or cls is typing.Any:
            return  # a builtin class, or ``typing.Any``: nothing in it changes
        if type(cls) not in _METACLASSES and not (self.content and self._metaclass_ok(cls)):
            raise Ineligible("a class with a metaclass of its own")
        for k in class_mro(cls):
            if not (k.__flags__ & _HEAPTYPE) or id(k) in self.types:
                continue
            tag = _cw.world_watch_type(k)
            if tag is None:
                raise Ineligible("a class without a version tag")
            self.types[id(k)] = (k, tag)
            if not is_user_code_object(k):
                # A library's class is keyed by its name, its code not read
                # (`_library_class_attr_parts`): what a key reads of it is
                # an attribute the code names (``pd.Timestamp.max``), and a
                # rebinding of any is a type event, which the watch sees.
                self.library_classes.append(k)
                continue
            self.classes.append(k)
            for name, attr in list(vars(k).items()):
                if _is_dunder(name) or name.startswith("_abc_"):
                    # Machinery, never data the key reads; but code it runs.
                    if issubclass(type(attr), (types.FunctionType, staticmethod, classmethod, property)):
                        self.value(attr, frozenset())
                    continue
                self.value(attr, frozenset())

    def _metaclass_ok(self, cls: type) -> bool:
        """The state memo's scan: *cls*'s metaclass is ``enum``'s own (an
        enum class: its members are its attributes, watched with it), or a
        library's for a class of that library, which a key keys by name
        and the library's version (`_library_class_attr_parts`). Such a
        metaclass is watched as a class too, and only one that reads a class's attributes as
        ``type`` does: no ``__getattr__`` or ``__getattribute__`` of its
        own, and no property but one returning what the class holds
        (`_plain_getter`); any other could answer a read from state no
        watcher sees. Except a library metaclass's property that is not a
        dunder (bokeh's ``MetaHasProps.model_class_reverse_map``, answering
        from a module global): a key reads a library class's attributes
        from ``vars()`` along its MRO (`_library_class_attr_parts`), never
        through its metaclass, so such a property only runs when the
        scanned code reads it by name; its name is noted and the scan
        fails then (`finish`)."""
        meta = type(cls)
        if meta is enum.EnumMeta:
            return True
        if is_user_code_object(cls) or is_user_code_object(meta):
            return False
        for k in meta.__mro__:
            if not (k.__flags__ & _HEAPTYPE):
                continue
            for name, attr in vars(k).items():
                if name in ("__getattr__", "__getattribute__"):
                    return False
                if (
                    not callable(attr)
                    and hasattr(type(attr), "__get__")
                    and type(attr) not in (staticmethod, classmethod)
                ):
                    # A property that returns what the class holds
                    # (``settings.n_jobs`` reading ``cls._n_jobs``) answers
                    # from the class, which is watched: its setter writes
                    # the class, a type event. The attributes it reads are
                    # looked into as named ones (`finish`).
                    if _is_dunder(name):
                        return False
                    read = _plain_getter(attr)
                    if read is None:
                        self.meta_props.add(name)
                    else:
                        self.names.update(read)
            if id(k) not in self.types:
                tag = _cw.world_watch_type(k)
                if tag is None:
                    return False
                self.types[id(k)] = (k, tag)
        return True

    def finish(self) -> None:
        """Look into the attributes of the library classes met that the
        scanned code names, until no new name is met. Then fail if the
        scanned code names a library metaclass's property that answers
        from more than its class holds (`_metaclass_ok`)."""
        from .key_values import plain_data_kind

        self._library_class_attrs(plain_data_kind)
        read = self.meta_props & self.names
        if read:
            raise Ineligible(f"a metaclass property read by name ({min(read)})")

    def _library_class_attrs(self, plain_data_kind: Any) -> None:
        """`finish`'s look into the library classes' named attributes."""
        done: set[tuple[int, str]] = set()
        while True:
            pending = []
            names = self.names
            for k in list(self.library_classes):
                ns = vars(k)
                for name in names.intersection(ns) if len(names) < len(ns) else [n for n in ns if n in names]:
                    if (id(k), name) not in done:
                        pending.append((k, name))
            if not pending:
                return
            for k, name in pending:
                done.add((id(k), name))
                attr = vars(k)[name]
                # Only data is keyed of a library's class: a descriptor or
                # a method of its own is its code, keyed by its name.
                if plain_data_kind(attr) is not None:
                    self.value(attr, frozenset())

    def family(self) -> _Family | None:
        """What a fast hit checks of the user classes met that no watcher
        sees (`_Family`); None for none."""
        from .class_data import SUBCLASSES

        if not self.classes and not self.probes:
            return None
        return _Family(tuple(self.classes), SUBCLASSES in self.names, tuple(self.probes.items()))

    def commit(self) -> None:
        """Watch what was found (after the whole scan passed)."""
        from .class_data import ANNOTATION_READS

        if ANNOTATION_READS & self.names:
            # The key folds the annotations of the classes met
            # (`class_data.class_annotations`); changed in place they move
            # no class tag.
            for k, _ in self.types.values():
                ns = vars(k)
                held = ns.get("__annotations__", ns.get("__annotations_cache__"))
                if type(held) is dict:
                    self.dicts.append((held, True))
        for d, unkeyed in self.dicts:
            _cw.world_watch_dict(d, unkeyed)
        _cw.world_add_names(self.names)
        _quiet_warnings()


#: ``id(module dict)`` of every module namespace watched so far.
_WATCHED_MODULES: set[int] = set()
_WATCH_LOCK = threading.Lock()


def _users(f: types.FunctionType, root_module: str | None) -> bool:
    """Is *f* code the key reads into: the user's by the helper walk's
    rule, or ``exec``'d into a namespace no module holds, whose code and
    reads the globals fold keys (`unregistered_namespace_code`)."""
    return own_code_is_user(f, root_module) or unregistered_namespace_code(f)


def world_epochs() -> tuple[int, int] | None:
    """``(world epoch, cash epoch)``, or None without the fast path."""
    if not ENABLED:
        return None
    return _world_epoch(), _cash_epoch()


def watch_helper(helper: Any, root_module: str | None) -> tuple:
    """Watch everything *helper*'s code names, as `scan` watches a cached
    function's world, so that while the world epoch stands still nothing it
    reads has moved; returns its cells (closure cells and what they held,
    the function and its names), which no watcher sees and `cells_hold`
    compares. Raises `Ineligible` for a helper of library code (its reads
    are not watched here) or one whose world holds what no watcher sees."""
    if type(helper) is not types.FunctionType or not _users(helper, root_module):
        raise Ineligible("a helper the scan does not watch")
    s = _Scan(root_module)
    s.shallow, s.shallow_pkg = True, own_package(helper)
    s._first(id(helper), helper)
    s.function(helper)
    s.commit()
    return tuple(s.cells)


def cells_hold(cells: tuple) -> bool:
    """Does every cell `watch_helper` returned still hold what it held?"""
    for owner, held in cells:
        if type(owner) is types.CellType:
            try:
                if owner.cell_contents is not held:
                    return False
            except ValueError:
                return False
        elif type(owner) is types.FunctionType:
            if owner.__name__ is not held[0] or owner.__qualname__ is not held[1]:
                return False
        else:
            return False
    return True


def _c_part(v: Any, name: str) -> Any:
    """*name* of a C-made typing object (``int | None``, ``list[int]``)."""
    try:
        return getattr(v, name, None)
    except Exception:  # noqa: BLE001 - nothing to look into
        return None


#: ``os.environ`` and ``os.environb``, met as attributes of ``os``.
_OS_ENVIRON = frozenset({"environ", "environb"})
_OS_GLOBALS = os.__dict__

#: What ``warnings.catch_warnings(record=True)`` sets and puts back: the
#: display hooks, swapped for a recorder's ``list.append`` and back. pandas
#: does it around every ``align`` (``rewrite_warning``), so a pvlib body
#: that clips a series moved the world on each call, and every miss built
#: its state segment again (`state_memo`). No key reads them unless scanned
#: code names them (``world_names``), and then a change still counts.
_WARNING_HOOKS = ("showwarning", "_showwarnmsg_impl")

#: Module -> the names of callables it holds that a library sets and puts
#: back around nearly every call, which no key reads unless scanned code
#: names them (`_WARNING_HOOKS`). pytest's assertion plugin binds
#: ``_reprcompare`` (and ``_assertion_pass`` when a plugin implements that
#: hook) around every test item and puts ``None`` back after it: each test
#: moved the world twice, so a function a test suite called once or twice
#: per test was never served from its memo (PyThaiNLP's ``word_dict_trie``,
#: 1673 hits across a test walk, none fast).
_QUIET_HOOKS = {
    "warnings": _WARNING_HOOKS,
    "_pytest.assertion.util": ("_reprcompare", "_assertion_pass"),
}


def _quiet_warnings() -> None:
    """Make `_QUIET_HOOKS` quiet in their modules' namespaces, for every
    scan: `watch_helper` watches the ``warnings`` namespace as soon as a
    helper names ``warnings``, while only `watch_modules` made the hooks
    quiet, and a function whose own scan stopped early (`Ineligible`) never
    reaches it. Every ``catch_warnings(record=True)`` its body ran then moved
    the world twice, and no helper fold was served again. A module not yet
    loaded is made quiet once it is."""
    if len(_QUIET_DONE) == len(_QUIET_HOOKS) or _cw is None or not hasattr(_cw, "world_quiet"):
        return
    for name, hooks in _QUIET_HOOKS.items():
        if name in _QUIET_DONE:
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        md = plain_getattr(module, "__dict__", None)
        if type(md) is dict:
            _cw.world_quiet(md, hooks)
            _QUIET_DONE.add(name)


#: The modules of `_QUIET_HOOKS` made quiet so far.
_QUIET_DONE: set[str] = set()


def watch_modules() -> None:
    """Watch every module's namespace (keyed) and ``sys.modules`` (every
    change): a helper cash finds by name can live in any of them."""
    with _WATCH_LOCK:
        modules = sys.modules
        _cw.world_watch_dict(modules, True)
        for mod in list(modules.values()):
            md = plain_getattr(mod, "__dict__", None)
            if type(md) is dict and id(md) not in _WATCHED_MODULES:
                _cw.world_watch_dict(md, False)
                _WATCHED_MODULES.add(id(md))
        _quiet_warnings()
        _cw.world_watch_dict(builtins.__dict__, False)


def _resolve(mod_name: str, chain: tuple) -> Any:
    obj: Any = sys.modules.get(mod_name)
    for attr in chain:
        if obj is None:
            return None
        obj = named_attr(obj, attr, None)
    return obj


def scan(
    spec: CachedFunction,
    registry: Any,
    *,
    environment_ok: bool = False,
    content: bool = False,
    reports: list | None = None,
) -> tuple[tuple, tuple, list, _Family | None]:
    """Watch the world of *spec*'s key. Returns the cells and the class
    tags a fast hit compares, the states of the cached functions it
    reaches, and the classes whose family it checks (`_Scan.family`);
    raises `Ineligible`. With *environment_ok* (the state memo, which checks
    each variable a segment read before serving it) an analysis that keys
    an environment read does not make it ineligible. With *content* (the
    state memo too) a list or an array met is read again on every reuse
    (`ContentProbe`) instead of making it ineligible. *reports*, when
    given, collects the analyses the scan resolved helpers by."""
    func, name = spec.func, spec.name
    s = _Scan(plain_getattr(func, "__module__", None), func)
    s.content = content
    hashers = plain_getattr(registry, "arg_hasher", None)
    # A registered hasher may key what a value holds, lock and all.
    s.left_out_ok = content and not (hashers is not None and (hashers.type_hashers or hashers.override_hashers))
    s.value(func, frozenset())
    graph = registry.graph
    stack, visited = [name], {name}
    while stack:
        node = stack.pop()
        if node in registry.data_sources:
            raise Ineligible("a data source")
        report = registry.report_for(func, name) if node == name else registry.purity_reports.get(node)
        if report is not None:
            if reports is not None:
                reports.append(report)
            if report.unwalkable or report.unkeyable or (report.environment_reads and not environment_ok):
                raise Ineligible("its analysis keys an environment read or could not finish")
            for mod_name, chain in report.helper_resolution_paths.values():
                s.value(_resolve(mod_name, chain), frozenset(chain))
            for ref in report.helper_objects.values():
                s.value(ref(), frozenset())
            for ref in report.cached_callees:
                s.value(ref() if callable(ref) else ref, frozenset())
            for mod_name, chain, _ref in report.helper_bindings:
                s.value(_resolve(mod_name, chain), frozenset(chain))
        dep_func = registry.functions.get(node)
        if dep_func is not None:
            s.value(dep_func, frozenset())
        dep_cf = registry.cached.get(node) if node != name else None
        if dep_cf is not None and dep_cf.hit_state is not None:
            s.callees.append(dep_cf.hit_state)
        for dep in graph.get_dependencies(node):
            if dep not in visited:
                visited.add(dep)
                stack.append(dep)
    s.finish()
    watch_modules()
    s.commit()
    cells = tuple(x for pair in s.cells for x in pair) + _cw.scoped_cells(s.names)
    tags = tuple(x for pair in s.types.values() for x in pair)
    return cells, tags, s.callees, s.family()


def scan_trace(spec: CachedFunction, registry: Any, items: list) -> tuple[tuple, tuple, list, _Family | None]:
    """Watch what a stored trace (`code_trace`) names, as `scan` watches the
    key's world: while none of it moves, the trace still holds, and a fast
    hit needs no check of its own. Raises `Ineligible`."""
    s = _Scan(plain_getattr(spec.func, "__module__", None))
    for item in items:
        kind = item[0]
        if kind in ("code", "class", "surface", "global", "held", "attr"):
            # From the module the item names, through the names it reads:
            # a function scanned is its code, globals, cells and defaults; a
            # class every attribute of it.
            mod = sys.modules.get(item[1])
            if mod is None:
                raise Ineligible("a traced module that is not loaded")
            path = item[2].split(".")
            names = frozenset(path if kind == "attr" else path[:1])
            s.names.update(names)
            s.value(mod, names)
        elif kind in ("import", "first_import") and item[1] in sys.modules:
            # A module loaded: the check (`code_trace.verify`) has nothing to
            # do for it, and sys.modules is watched, so unloading or
            # replacing it moves the guard. A first import's file was
            # checked by the hit that armed this path; the module that runs
            # is the one loaded.
            continue
        elif kind == "cached":
            cf = registry.cached.get(item[1])
            if cf is None or cf.hit_state is None:
                raise Ineligible("a cached function served inside without a fast path")
            s.callees.append(cf.hit_state)
        else:
            # A lookup by name or an import by name: checked on each hit.
            raise Ineligible(f"a traced {kind}")
    s.finish()
    watch_modules()
    s.commit()
    cells = tuple(x for pair in s.cells for x in pair) + _cw.scoped_cells(s.names)
    tags = tuple(x for pair in s.types.values() for x in pair)
    return cells, tags, s.callees, s.family()


class ContentProbe:
    """Data a key folds by its content that no watcher sees changed in
    place: an exact list, set or bytearray of plain data
    (`key_values.plain_data_kind`), or an exact numpy array of a dtype
    without objects. Called, it reads the whole value again and gives a
    digest that moves with any change the fold could see (`_Family.holds`
    compares it): the pickle of the plain data, which spells out every
    item, its type and the sharing between them; the array's dtype, shape,
    strides and bytes. As costly as the fold's own hash of the value, which
    it stands in for; no costlier.

    Rebinding the name that holds the value moves a watched dict; this
    covers what is done to the value itself."""

    __slots__ = ("digest", "read", "value")

    def __init__(self, value: Any, read: Any) -> None:
        self.value = value
        self.read = read
        self.digest = _read_in_round(value, read)

    def __call__(self) -> Any:
        return self.read(self.value)

    @classmethod
    def of(cls, v: Any) -> ContentProbe | None:
        """A probe of *v*, or None when it is no data a probe reads whole."""
        t = type(v)
        if t is list or t is set or t is bytearray:
            from .key_values import plain_data_kind

            if t is not bytearray and plain_data_kind(v) != "plain":
                return None
            try:
                return cls(v, _plain_content)
            except Exception:  # noqa: BLE001 - cannot be read whole: not probed
                return None
        if t is collections.OrderedDict:
            from .key_values import plain_data_kind

            if plain_data_kind(dict(v)) is None:
                return None
            try:
                return cls(v, _plain_content)
            except Exception:  # noqa: BLE001 - cannot be read whole: not probed
                return None
        if t.__name__ == "ndarray" and t.__module__ == "numpy":
            try:
                if v.dtype.hasobject:
                    return None
                return cls(v, _array_content)
            except Exception:  # noqa: BLE001 - cannot be read whole: not probed
                return None
        pd = sys.modules.get("pandas")
        if pd is not None and (t is pd.DataFrame or t is pd.Series):
            try:
                if not _plain_frame(v, pd):
                    return None
                return cls(v, _frame_content)
            except Exception:  # noqa: BLE001 - cannot be read whole: not probed
                return None
        return None


_NOTHING: Any = object()

#: ``(world epoch, {(id of the value, read): (value, what it read)})`` while
#: a round of checks is open on this thread (`probe_round`).
_ROUND: contextvars.ContextVar[list | None] = contextvars.ContextVar("cash_probe_round", default=None)


@contextlib.contextmanager
def probe_round() -> typing.Iterator[None]:
    """One round of checks (a key build, the check of a stored trace) in
    which each value a `ContentProbe` reads whole is read once, however
    many watches hold a probe of it: lmfit's module of line shapes holds
    four scipy distributions, which each of a dozen class watches read
    again, 5 ms a call. Cash's own code runs between the checks of a
    round, never the user's body; what the round read is forgotten when
    the world epoch moves (an import, a class edited), and a probe made
    since reads its value itself."""
    if _ROUND.get() is not None:
        yield
        return
    token = _ROUND.set([_cw.world_epoch() if _cw is not None else 0, {}])
    try:
        yield
    finally:
        _ROUND.reset(token)


def _probe_round() -> dict | None:
    """The reads of the open round (`probe_round`), or None."""
    r = _ROUND.get()
    if r is None:
        return None
    epoch = _cw.world_epoch() if _cw is not None else 0
    if r[0] != epoch:
        r[0] = epoch
        r[1] = {}
    return r[1]


def round_checks() -> dict | None:
    """What the open round of checks (`probe_round`) found so far, for a
    check that the round's other checks may ask again (`class_memo`), or
    None outside one. Emptied when the world epoch moves."""
    return _probe_round()


def _read_in_round(value: Any, read: Any) -> Any:
    """``read(value)``, once a round (`probe_round`) when one is open: a
    watch made in it starts from what the watches checked in it read."""
    seen = _probe_round()
    if seen is None:
        return read(value)
    k = (id(value), read)
    got = seen.get(k, _NOTHING)
    if got is _NOTHING or got[0] is not value:
        got = seen[k] = (value, read(value))
    return got[1]


def _class_and_dict(v: Any) -> _Items:
    """*v*'s class and ``__dict__``, compared by identity."""
    return _Items((type(v), object.__getattribute__(v, "__dict__")))


def _context_value(var: contextvars.ContextVar) -> _Items:
    """What *var* holds in the current context, compared by identity."""
    return _Items((var.get(_NOTHING),))


def _unread_object(v: Any) -> bool:
    """Is *v* an object a key never folds when read as a library module's
    attribute: not plain data (`key_values.plain_data_kind`, which
    looks only into exact builtin containers and leaves, so no change to
    *v* makes it plain), and not a module, class or callable, whose name a
    key reads."""
    from ..value_types import IMMUTABLE_LEAF_TYPES

    t = type(v)
    if t in IMMUTABLE_LEAF_TYPES or t in (tuple, frozenset, list, set, dict):
        return False
    return not (isinstance(v, (type, types.ModuleType)) or callable(v))


class _Items:
    """The items of a list, compared by identity: equal while it holds the
    very same objects in the same order (`_Scan._inside`)."""

    __slots__ = ("items",)

    def __init__(self, items: tuple) -> None:
        self.items = items

    @classmethod
    def of(cls, v: list) -> _Items:
        return cls(tuple(v))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Items) or len(self.items) != len(other.items):
            return False
        return all(a is b for a, b in zip(self.items, other.items))

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]


def _plain_content(v: Any) -> bytes:
    import hashlib
    import pickle

    return hashlib.blake2b(pickle.dumps(v, protocol=5), digest_size=32).digest()


def _global_stream(owner: Any) -> bool:
    """Is *owner* the process's global random stream, numpy's
    (``np.random.normal`` is a method of ``numpy.random.mtrand._rand``) or
    the standard library's (``random.random``, of ``random._inst``)?

    What a key reads of it is its seed epoch, folded after the part the
    state memo keeps (`KeyBuilder._code_state_post`), on every call; the
    part kept reads the method by its library's name only, never the
    stream's position, which every draw moves and no watcher sees. So the
    state memo's scan watches the stream's class, not the stream."""
    mtrand = sys.modules.get("numpy.random.mtrand")
    if mtrand is not None and owner is getattr(mtrand, "_rand", None):
        return type(owner) is getattr(mtrand, "RandomState", None)
    stdlib = sys.modules.get("random")
    return stdlib is not None and owner is getattr(stdlib, "_inst", None) and type(owner) is stdlib.Random


def _library_cython(v: Any, t: type) -> bool:
    """Is *v* a Cython-compiled function of a library (``np.random.default_rng``,
    pandas' ``is_scalar``)? A key names it as it names a library's Python
    function, by its module and name, its code keyed by the library's
    version: like one (`_Scan.function`), there is nothing in it to watch,
    and rebinding the name is a module event. One the user compiled is
    keyed by its source, which no watcher sees: not this (`_Scan._user_cython`)."""
    if t.__name__ not in ("cython_function_or_method", "fused_cython_function"):
        return False
    if not t.__module__.startswith("_cython_"):
        return False
    from ..install_paths import is_user_code_module

    name = plain_getattr(v, "__module__", None)
    mod = sys.modules.get(name) if isinstance(name, str) else None
    if mod is None or name == "cash" or name.startswith("cash."):
        return False
    return not is_user_code_module(mod)


def _cython_reads(v: Any) -> tuple:
    """What a key reads of *v*, a Cython function of the user's build
    (`_Scan._user_cython`): its module, the module's file, its names, code
    and ``__dict__``, compared by identity, and the size and times of every
    file its key was worked out from."""
    from .._user_attrs import module_file
    from ..cython_digest import keyed_files

    name = plain_getattr(v, "__module__", None)
    module = sys.modules.get(name) if isinstance(name, str) else None
    parts = (
        type(v),
        name,
        module,
        module_file(module),
        plain_getattr(v, "__qualname__", None),
        plain_getattr(v, "__name__", None),
        plain_getattr(v, "__code__", None),
        plain_getattr(v, "__dict__", None),
    )
    paths = keyed_files(v)
    stats = []
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            stats.append(None)
        else:
            stats.append((st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino))
    return (_Items(parts), paths, tuple(stats))


def _attribute_count(v: Any) -> int:
    """How many attributes *v* (a number or string of a subclass) holds."""
    return len(v.__dict__)


def _plain_frame(v: Any, pd: Any) -> bool:
    """Is *v* (an exact frame or series) one whose content hash
    (`content_hashers.hash_pandas`) is all a key reads of it and moves
    with any change to it: no ``attrs``, columns of numbers, booleans,
    dates or strings, labels and axis names of immutable scalars. A
    column of objects could hold one changed in place, which no hash of
    the column's references sees."""
    from ..content_hashers import immutable_labels, pandas_value_route

    if v.attrs:
        return False
    dtypes = list(v.dtypes) if _is_frame(v, pd) else [v.dtype]
    for dtype in dtypes:
        if not _plain_dtype(dtype, pd, pandas_value_route):
            return False
    axes = [v.index, v.columns] if _is_frame(v, pd) else [v.index]
    names = [] if _is_frame(v, pd) else [v.name]
    for axis in axes:
        names.extend(axis.names)
        levels = axis.levels if isinstance(axis, pd.MultiIndex) else [axis]
        for level in levels:
            if _plain_dtype(level.dtype, pd, pandas_value_route):
                continue
            if level.dtype != object or not immutable_labels(level._values):
                return False
    return all(type(name) in _ATOMS for name in names)


def _is_frame(v: Any, pd: Any) -> bool:
    return type(v) is pd.DataFrame


def _plain_dtype(dtype: Any, pd: Any, route_of: Any) -> bool:
    route = route_of(dtype, pd)
    return route in ("raw", "datetimetz") or isinstance(dtype, pd.StringDtype)


def _frame_content(v: Any) -> str:
    """The content hash of a frame or series, as a key folds it: through
    the copy-on-write memo (`arg_hashing.cow_frame_digest`), read again
    only when the frame changed, else in full."""
    from ..content_hashers import hash_pandas
    from .arg_hashing import cow_frame_digest

    digest = cow_frame_digest(v) or hash_pandas(v)
    if digest is None:
        raise TypeError("a frame that cannot be hashed")
    return digest


def _array_content(v: Any) -> tuple:
    import hashlib

    import numpy as np

    if v.dtype.hasobject:
        raise TypeError("an array of objects")
    data = v if v.flags.c_contiguous else np.ascontiguousarray(v)
    digest = hashlib.blake2b(data.reshape(-1).view(np.uint8), digest_size=32).digest() if data.size else b""
    return (v.dtype.str, v.dtype.descr if v.dtype.names else None, v.shape, v.strides, digest)


def watch_class(cls: type, reports: Any = (), reached: list | None = None) -> tuple[tuple, tuple, _Family | None]:
    """Watch what *cls* holds, as `scan` watches a cached function's world:
    every attribute along its MRO, the code of its methods and what that
    code names, and the helpers *reports* (the analyses of its methods)
    resolved. Returns the cells, the class tags and the family a check
    compares. Data no watcher sees and no probe of `ContentProbe` reads
    whole (an instance, a list of them: sklearn's parameter constraints) is
    read whole again by pickle on every check (`_deep_content`), and the
    user code that pickle names in it (an instance's class, a function in a
    list) is watched as any code met. For the per-class memos of `class_memo`. Raises
    `Ineligible`, also for a class whose world reaches a cached function,
    unless *reached* is given: the cached functions' states met are then
    added to it, and the caller answers for them."""
    reports = tuple(reports)
    s = _ProbingScan(plain_getattr(cls, "__module__", None))
    s.content = True
    s.value(cls, frozenset())
    for report in reports:
        if report.unwalkable or report.unkeyable:
            raise Ineligible("an analysis that could not finish")
        for mod_name, chain in report.helper_resolution_paths.values():
            s.value(_resolve(mod_name, chain), frozenset(chain))
        for ref in report.helper_objects.values():
            s.value(ref(), frozenset())
        for ref in report.cached_callees:
            s.value(ref() if callable(ref) else ref, frozenset())
        for mod_name, chain, _ref in report.helper_bindings:
            s.value(_resolve(mod_name, chain), frozenset(chain))
    if s.callees or any(report.cached_callees for report in reports):
        if reached is None:
            raise Ineligible("a cached function reached")
        reached.extend(s.callees)
        reached.extend(ref for report in reports for ref in report.cached_callees)
    s.finish()
    if _probes_main_variables(s.probes, (x for pair in s.cells for x in pair)):
        raise Ineligible("a probe would hold a variable of __main__")
    watch_modules()
    s.commit()
    cells = tuple(x for pair in s.cells for x in pair)
    tags = tuple(x for pair in s.types.values() for x in pair)
    family = s.family()
    if family is not None:
        # Held by a memo of the process, it must not keep a class alive:
        # one defined anew under the same name (a notebook cell run again)
        # would stay among its base's ``__subclasses__``.
        weak = _WeakFamily.__new__(_WeakFamily)
        weak.refs = tuple(map(_weak.ref, family.classes))
        weak.probes, weak.guard = family.probes, family.guard
        weak.registrations, weak.below = family.registrations, family.below
        family = weak
    return cells, tags, family


def _probes_main_variables(probes: dict, met: typing.Iterable[Any]) -> bool:
    """Would a probe of *probes* hold a value a variable of the notebook is
    bound to: of ``__main__``, or of a namespace the code *met* runs in
    that is no imported module's own (a notebook shell's, a script's)? A
    memo of the process would then be one more holder of it, and the
    notebook, which counts a variable's holders by reference count to tell
    whether a restored copy could stand in for it, refuses to store any
    statement that changes it (``calls += [...]`` after a decorated call
    met a class whose method appends to ``calls``)."""
    if not any(type(probe) is ContentProbe for probe in probes):
        return False
    spaces = {}
    main = getattr(sys.modules.get("__main__"), "__dict__", None)
    if type(main) is dict:
        spaces[id(main)] = main
    for f in met:
        if type(f) is types.FunctionType:
            g = f.__globals__
            name = g.get("__name__")
            if name == "__main__" or getattr(sys.modules.get(name), "__dict__", None) is not g:
                spaces[id(g)] = g
    bound = {id(v) for g in spaces.values() for v in list(g.values())}
    return any(type(probe) is ContentProbe and id(probe.value) in bound for probe in probes)


#: What `_ProbingScan` never reads whole: code, whose own watch is the proof.
_CODE_TYPES = (type, types.FunctionType, types.ModuleType, types.MethodType, types.BuiltinFunctionType, types.CellType)


class _ProbingScan(_Scan):
    """`_Scan`, but a data value it would refuse is read whole again on
    every check instead (`_deep_content`), and the code it names watched."""

    def value(self, v: Any, names: frozenset) -> None:
        if type(v) in _ATOMS:
            return
        try:
            super().value(v, names)
        except Ineligible:
            if isinstance(v, _CODE_TYPES) or self.count > _SCAN_LIMIT:
                raise
            met: list = []
            try:
                probe = ContentProbe(v, lambda value: _deep_content(value, met))
            except Exception as e:  # noqa: BLE001 - what cannot be pickled cannot be probed
                raise Ineligible(f"data that cannot be read whole ({class_name(type(e))})") from None
            self.probes[probe] = probe.digest
            # A pickle names code: what that code is and reads is watched.
            for code in met:
                self.value(code, frozenset())
            met.clear()
            probe.read = _deep_content


class _CodeNamingPickler(pickle.Pickler):
    """Notes the user code met inside the data it pickles (*met*): a pickle
    names it, never what it is or reads."""

    def __init__(self, file: Any, met: list | None) -> None:
        super().__init__(file, protocol=5)
        self.met = met

    def reducer_override(self, obj: Any) -> Any:
        if self.met is not None and isinstance(obj, _CODE_TYPES) and is_user_code_object(obj):
            self.met.append(obj)
        return NotImplemented


def _deep_content(v: Any, met: list | None = None) -> bytes:
    """A digest of all *v* holds, as pickle spells it out; the user code
    it names joins *met*."""
    import hashlib

    buf = io.BytesIO()
    _CodeNamingPickler(buf, met).dump(v)
    return hashlib.blake2b(buf.getbuffer(), digest_size=32).digest()


class _Family:
    """What the key reads of the user classes met that moves no watcher
    (`class_data.class_family`): the ABCs that registered them, what the
    ABCs among them registered, and, when the world's code reads
    ``__subclasses__``, the classes below them.

    A registration anywhere moves ``abc.get_cache_token()``, which
    `_cwatch.fast_call` compares on every hit (``state[S_GUARD]``); only
    then are the registrations read again here, and a fast hit goes on
    when they are what they were. A class defined below one moves nothing
    at all: those are held by weakref (a collected class's address may be
    a new one's) and compared on every hit, in Python. A class below
    edited in place is watched like any class met.
    """

    __slots__ = ("below", "classes", "guard", "probes", "registrations")

    def __init__(self, classes: tuple[type, ...], subclasses: bool, probes: tuple = ()) -> None:
        from .class_data import classes_below

        self.classes = classes
        #: ``(read, value)``: read again on every fast hit and reuse, which
        #: then is served in Python (`_Scan.probes`).
        self.probes = probes
        self.guard = (abc.get_cache_token, abc.get_cache_token())
        self.registrations = self._registrations()
        self.below: tuple[tuple[weakref.ref, ...], ...] | None = None
        if subclasses:
            below = [classes_below(k) for k in classes]
            for k in (k for ks in below for k in ks):
                if k.__flags__ & _HEAPTYPE and _cw.world_watch_type(k) is None:
                    raise Ineligible("a class without a version tag")
            self.below = tuple(tuple(map(_weak.ref, ks)) for ks in below)

    def _registrations(self) -> tuple:
        from .class_data import abc_registries, registered_with

        return tuple((registered_with(k), tuple(abc_registries(k))) for k in self.classes)

    def joined(self, older: _Family | None) -> _Family | None:
        """*older*'s classes and this family's new ones, as one: each as
        it was when first met, under *older*'s guard, so a registration
        between the two is still seen. None when one watches the classes
        below and the other does not: the older entries' keys did not."""
        if older is None:
            return self
        if (self.below is None) != (older.below is None):
            return None
        known = {id(k) for k in older.classes}
        fresh = [i for i, k in enumerate(self.classes) if id(k) not in known]
        if not fresh and all(p in older.probes for p in self.probes):
            return older
        joined = _Family.__new__(_Family)
        joined.probes = (*older.probes, *(p for p in self.probes if p not in older.probes))
        joined.classes = (*older.classes, *(self.classes[i] for i in fresh))
        joined.guard = older.guard
        joined.registrations = (*older.registrations, *(self.registrations[i] for i in fresh))
        joined.below = None
        if self.below is not None and older.below is not None:
            joined.below = (*older.below, *(self.below[i] for i in fresh))
        return joined

    def holds(self, state: list) -> bool:
        """Is what the key read of the classes still so? Moves the guard
        on when a registration elsewhere left theirs as they were."""
        seen = _probe_round()
        for read, value in self.probes:
            try:
                if seen is not None and type(read) is ContentProbe:
                    # Read once a round (`probe_round`): a dozen class
                    # watches can share one module's data.
                    if _read_in_round(read.value, read.read) != value:
                        return False
                elif read() != value:
                    return False
            except Exception:  # noqa: BLE001 - cannot tell: not so
                return False
        token = abc.get_cache_token()
        if token != self.guard[1]:
            if self._registrations() != self.registrations:
                return False
            self.guard = (abc.get_cache_token, token)
            state[S_GUARD] = self.guard
        if self.below is not None:
            from .class_data import classes_below

            for k, refs in zip(self.classes, self.below):
                now = classes_below(k)
                if len(now) != len(refs) or any(r() is not c for r, c in zip(refs, now)):
                    return False
        return True


class _WeakFamily(_Family):
    """A `_Family` holding its classes by weak reference (`watch_class`):
    one that is gone no longer holds."""

    __slots__ = ("refs",)

    @property  # type: ignore[override]
    def classes(self) -> tuple[type, ...]:
        found = tuple(ref() for ref in self.refs)
        if any(k is None for k in found):
            raise Ineligible("a class that is gone")
        return found

    def holds(self, state: list) -> bool:
        try:
            return super().holds(state)
        except Ineligible:
            return False


# -- the memo --------------------------------------------------------------

#: What `_cwatch.fast_call` hands back for "take the ordinary path".
MISS = object()

# The slots of a memo's state list, as `_cwatch.fast_call` reads them.
S_WORLD, S_ENTRIES, S_CELLS, S_TYPES, S_SERVE, S_ACC, S_SPEC, S_LOG, S_GUARD, S_CLEAR_AT, S_FLAGS = range(11)
# The slots of an entry tuple.
E_CASH, E_STORE, E_KEY, E_HELD, E_HIT, E_VALUE, E_TOUCH, E_SAVED, E_LOG, E_BODY = range(10)

fast_call = _cw.fast_call if ENABLED else None
if ENABLED:
    from .call_state import CALL_ENTRY

    _cw.hit_configure(MISS, CALL_ENTRY, active_tracker, NESTED_CASH_SECONDS)


class _State(list):
    """The slots `_cwatch.fast_call` reads; a wrapper closes over it, so it
    refuses pickling by value like the rest of what a wrapper holds."""

    __slots__ = ("__weakref__",)

    # One function's state, keyed by identity (`_DEPENDENTS`).
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    def __reduce__(self):
        from .script_pickling import refuse_pickling_by_value

        refuse_pickling_by_value()


def new_state() -> list | None:
    """A state list for a cached function's wrapper to hand `_cwatch.fast_call`
    (empty: every call takes the ordinary path until `HitMemo` fills it)."""
    if not ENABLED:
        return None
    return _State([-1, {}, (), (), None, _cw.Acc(), None, None, None, 0.0, VERIFY])


class _Hit:
    """What a fast hit needs besides what `_cwatch.fast_call` reads."""

    __slots__ = ("metadata", "deadline", "files", "lib", "ttl", "ram", "backend", "copy", "value", "md")


def _scoped_part(cells: tuple) -> tuple:
    """The ``(name, epoch)`` pairs of a scan's flat *cells* (`scoped_cells`)."""
    return tuple(x for i in range(0, len(cells) - 1, 2) if type(cells[i]) is str for x in cells[i : i + 2])


class HitMemo:
    """One cached function's fast hits (see the module doc)."""

    #: Remembered calls per function at most.
    MAX_ENTRIES = 4096
    #: A scan is due this long after the last one, whatever the backoff. The
    #: backoff counts calls: 1,024 of a function whose ordinary hit checks a
    #: file took seconds on a Windows runner, so a world that moved now and
    #: then kept it off the memo for good. A scan takes ~0.5-1 ms, so this
    #: costs at most ~0.4% of a run whose world never stays still.
    RESCAN_AFTER_S = 0.25

    def __init__(self, spec: CachedFunction, runner: Any) -> None:
        self.spec = spec
        self.runner = runner
        state = spec.hit_state
        if state is None:
            state = spec.hit_state = new_state()
        #: Read by `_cwatch.fast_call`; world epoch -1 until a scan passed.
        self.state: list = state
        state[S_SERVE] = self.serve
        state[S_SPEC] = spec
        state[S_LOG] = runner._calls
        self.failed_epoch = -1
        #: Traces watched (`scan_trace`) -> the world epoch they were at.
        self.traces: dict[tuple, int] = {}
        self.slow_since_scan = 0
        self.backoff = 1
        self.scanned_at = 0.0
        self.served_since_scan = 0
        #: What the key read of the classes met that no watcher sees.
        self.family: _Family | None = None
        #: The ``(name, epoch)`` cells of the scoped names the scans met.
        self.scoped: tuple = ()
        _MEMOS.add(self)

    # -- served ----------------------------------------------------------

    def serve(self, entry: tuple, args: tuple, kwargs: dict, t0: float) -> Any:
        """A fast hit `_cwatch.fast_call` found but leaves to Python: a ttl,
        files to check, a copy to make, the clear check due, verify mode, or
        accounting to hand over. Returns the result, or `MISS`."""
        state = self.state
        self.flush()
        family = self.family
        if family is not None and not family.holds(state):
            # An ABC registered a class, or a class was defined below one:
            # the key may differ; scanned again at the next ordinary hit.
            state[S_WORLD] = -1
            return MISS
        hit: _Hit = entry[E_HIT]
        runner = self.runner
        now = time.time()
        if hit.deadline is not None and now >= hit.deadline:
            return MISS
        backend = hit.backend
        clear_watch = getattr(backend, "_clear_watch", None)
        if clear_watch is not None:
            if time.monotonic() - clear_watch._checked_at >= clear_watch.CHECK_EVERY:
                backend._drop_ram_if_cleared()
                if entry[E_STORE].get(entry[E_KEY]) is not entry[E_HELD]:
                    return MISS
            state[S_CLEAR_AT] = clear_watch._checked_at + clear_watch.CHECK_EVERY
        else:
            state[S_CLEAR_AT] = float("inf")
        if hit.files and not runner._files.auto_file_deps_fresh(hit.metadata):
            return MISS
        if hit.lib and library_reads.changed(hit.lib) is not None:
            return MISS
        if VERIFY:
            self._verify(entry, args, kwargs)
        ram = hit.ram
        key = entry[E_KEY]
        if hit.copy:
            md, value = ram.get(key)
            if md is None:
                return MISS
            store = runner._store
            store.attach_lineage(value, key, hit.metadata.auto_file_deps, ttl=hit.ttl, func_name=self.spec.name)
            store.carry_digest(hit.metadata, value)
        else:
            value = hit.value
            with ram._lock:
                md = hit.md
                md["last_access"] = now
                md["access_count"] = md.get("access_count", 0) + 1
                ram._touch(key)
        spec = self.spec
        spec.last_key = key
        dt = _perf_counter() - t0
        stats = spec.stats
        stats["hits"] += 1
        stats["total_time_saved"] += entry[E_SAVED]
        stats["lookup_seconds"] += dt
        runner._calls.fast_log.append((entry[E_LOG], now, dt))
        body = entry[E_BODY]
        if body is not None:
            runner._calls.note_fast_effectiveness(spec.name, 1, dt, body, body)
        self.served_since_scan += 1
        if hit.copy:
            runner._rng.warn_unseeded_estimator_result(spec.name, value, spec.allow_random)
        return value

    def flush(self) -> None:
        """Hand the fast hits not yet accounted to the net-loss ledger, as
        that many ordinary hits would have been."""
        taken = self.state[S_ACC].take()
        if taken is None:
            return
        hits, saved_total, lookup, last_key, n, overhead, saved, body = taken
        self.served_since_scan += hits
        spec = self.spec
        stats = spec.stats
        if hits:
            stats["hits"] += hits
            stats["total_time_saved"] += saved_total
            stats["lookup_seconds"] += lookup
        if last_key is not None:
            spec.last_key = last_key
        if n:
            self.runner._calls.note_fast_effectiveness(spec.name, n, overhead, saved, body)

    def _ordinary_key(self, args: tuple, kwargs: dict) -> Any:
        """The key the ordinary path builds for *args*, building nothing else."""
        from .._active import ACTIVE_CONFIG, EXPLAINING

        token = EXPLAINING.set(True)
        config_token = ACTIVE_CONFIG.set(self.runner._calls._config)
        try:
            return self.runner._keys.build(self.spec, args, kwargs)
        finally:
            ACTIVE_CONFIG.reset(config_token)
            EXPLAINING.reset(token)

    def _verify(self, entry: tuple, args: tuple, kwargs: dict) -> None:
        try:
            built = self._ordinary_key(args, kwargs)
        except Exception as e:
            raise HitMemoMismatch(f"fast hit of {self.spec.name}: the ordinary key build raised {e!r}") from e
        VERIFIED[0] += 1
        if built.cache_key != entry[E_KEY]:
            raise HitMemoMismatch(
                f"fast hit of {self.spec.name} served {entry[E_KEY]!r}, the ordinary key is {built.cache_key!r}"
            )

    # -- learned ---------------------------------------------------------

    def offer(self, call: Any, args: tuple, kwargs: dict, hit: Any, before: tuple) -> None:
        """Remember the ordinary hit *call* just made, when it may be served
        again from here (see the module doc)."""
        spec, runner = self.spec, self.runner
        _cw.warn_check()  # the warnings state moved since `epochs`: the cash epoch did
        world_now = _world_epoch()
        state = self.state
        if world_now != before[0] or _cash_epoch() != before[1] or state[S_ENTRIES] is not before[2]:
            return  # something moved while the key was built
        # The memo is at the core epoch, and at the epochs of the scoped
        # names its scans met (`_SCOPED_NAMES`), kept among its cells.
        core_now = _core_epoch()
        if state[S_WORLD] != core_now or (self.scoped and not _cw.world_holds(self.scoped, ())):
            if self.failed_epoch == world_now:
                return
            self.slow_since_scan += 1
            now = time.monotonic()
            if self.slow_since_scan < self.backoff and now - self.scanned_at < self.RESCAN_AFTER_S:
                return
            self.backoff = 1 if self.served_since_scan else min(self.backoff * 2, 1024)
            self.slow_since_scan = self.served_since_scan = 0
            self.scanned_at = now
            state[S_WORLD] = -1
            state[S_ENTRIES] = {}
            try:
                cells, tags, callees, family = scan(spec, runner._registry)
            except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: the ordinary path
                self.failed_epoch = world_now
                return
            self.family = family
            self.scoped = _scoped_part(cells)
            state[S_GUARD] = family.guard if family is not None else None
            for callee in callees:
                if callee is not state:
                    _DEPENDENTS.setdefault(callee, _weak.WeakSet()).add(self)
            # What changed between the key build and the watch is not
            # proven: this call is not remembered, the next one is.
            state[S_CELLS], state[S_TYPES] = cells, tags
            state[S_WORLD] = core_now
            self.traces = {}
            return
        trace = call.metadata.code_trace if call.metadata is not None else None
        if trace and not self._watch_trace(trace, core_now):
            return
        entry = self._entry(call, hit)
        if entry is None:
            return
        keyed = before[3]
        if keyed is None or _cw.keyed(args, kwargs) != keyed:
            return  # not watched before the key build, or armed again since
        key = keyed[0]
        if type(key[0]) is bytes and (b"[" in key[0] or b"{" in key[0]):
            # A small list or dict is in the key by value, and no arm
            # numbers its content: another thread may have changed it while
            # the ordinary key was built and put it back since, and may have
            # ended by now. Once per entry remembered, not per hit.
            key = self._key_of_a_copy(call, args, kwargs, keyed[1])
            if key is None:
                return
        entries = state[S_ENTRIES]
        if len(entries) >= self.MAX_ENTRIES:
            entries.clear()
        entries[key] = (before[1], *entry)

    def _key_of_a_copy(self, call: Any, args: tuple, kwargs: dict, gen: int) -> Any:
        """The memo key of a copy of *args* whose ordinary key is the one
        *call* was served under, or None. The copy is this call's own, so
        what its two keys are built from is one content, whatever another
        thread does to the arguments meanwhile."""
        copied = tuple(_plain_copy(a, 0) for a in args)
        copied_kw = {k: _plain_copy(v, 0) for k, v in kwargs.items()}
        again = _cw.keyed(copied, copied_kw)
        if again is None or again[1] != gen:
            return None
        try:
            built = self._ordinary_key(copied, copied_kw)
        except Exception:  # noqa: BLE001 - no key for the copy: not remembered
            return None
        if getattr(built, "cache_key", None) != call.cache_key:
            return None
        return again[0]

    def _watch_trace(self, trace: list, world_now: int) -> bool:
        """Is *trace* watched since this world epoch? Watches it when not
        (and the call is not remembered: what moved before the watch is not
        proven), adding its cells and class tags to those every fast hit
        compares."""
        try:
            key = tuple(tuple(item) for item in trace)
            hash(key)
        except TypeError:
            return False
        if self.traces.get(key) == world_now:
            return True
        if key in self.traces:
            return False  # watched before, and something moved since: Ineligible then
        state = self.state
        try:
            cells, tags, callees, family = scan_trace(self.spec, self.runner._registry, trace)
        except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: the ordinary path
            self.traces[key] = -1
            return False
        if family is not None:
            joined = family.joined(self.family)
            if joined is None:
                self.traces[key] = -1
                return False
            self.family = joined
            state[S_GUARD] = joined.guard
        for callee in callees:
            if callee is not state:
                _DEPENDENTS.setdefault(callee, _weak.WeakSet()).add(self)
        state[S_CELLS] = state[S_CELLS] + cells
        state[S_TYPES] = state[S_TYPES] + tags
        self.scoped += _scoped_part(cells)
        self.traces[key] = world_now
        return False

    def _entry(self, call: Any, hit: Any) -> tuple | None:
        """``(store, key, held, _Hit, value, md, saved, log, body)``, or None."""
        runner = self.runner
        metadata = call.metadata
        # `call.capture_watch` (globals, class data, helpers the key folded)
        # is read by a miss only; the scan watched or refused each of them.
        if metadata is None:
            return None
        if (
            metadata.iterator_storage == "chunked"
            or metadata.dynamic_sources
            or metadata.dynamic_resolvers
            or metadata.result_ref
            or metadata.held_refs
            or metadata.rng_replay
            or metadata.read_only
            or metadata.env_reads
            # A hit writes the state the call left its arguments in back
            # into them: never served without that write.
            or metadata.args_left
        ):
            return None
        ledger = call.ledger or {}
        chain = ledger.get("@chain") or []
        # The stages after the globals: rng epoch, then the environment,
        # whose parts the labels below name: an environment read refuses it,
        # a library's installed version (read once per process) does not.
        if len(chain) < 7 or chain[5] != chain[4]:
            return None
        for label in ledger:
            if label in ("@chain", "source"):
                continue
            if not (isinstance(label, tuple) and label[0] in _LEDGER_KINDS):
                return None
        backend = runner._backend_slot.backend
        tiers = getattr(backend, "backends", None)
        ram = tiers[0] if isinstance(tiers, list) and tiers else backend
        store = getattr(ram, "_store", None)
        if type(store) is not dict or not hasattr(ram, "_touch") or not hasattr(ram, "_lock"):
            return None
        held = store.get(call.cache_key)
        if held is None or type(held) is not tuple or len(held) != 2:
            return None
        md, stored = held
        log = runner._calls.fast_log_template(call.args_hash, call.cache_key, self.spec.name, metadata)
        if log is None:
            return None
        h = _Hit()
        h.metadata = metadata
        h.ram = ram
        h.backend = backend
        h.files = bool(metadata.auto_file_deps)
        h.lib = metadata.lib_reads
        h.ttl = call.ttl
        ttl = runner._backend_slot.entry_ttl(call.ttl, metadata)
        h.deadline = None
        if ttl is not None:
            from ..backends._base import written_at

            started = written_at(metadata)
            if started is None:
                return None
            h.deadline = started + ttl
        # An immutable result is handed back as the object the tier holds,
        # not a new copy of it: nothing about it can be changed through it.
        direct = (
            type(stored) is type(hit)
            and _deep_atoms(hit, 0)
            and (stored is hit or stored == hit)
            and not metadata.content_digest
            and (type(hit) is not tuple or not self.spec.frozen)
        )
        h.copy = not direct
        h.value = stored
        h.md = md
        # fast_call serves in C only what needs nothing more than it does.
        in_c = (
            direct
            and h.deadline is None
            and not h.files
            and (not h.lib or library_reads.classes_watched(h.lib, _cw.world_watch_type))
            and (self.family is None or (self.family.below is None and not self.family.probes))
        )
        # Never the hit credited with a warm-up: the ordinary hit before it was.
        saved, body = runner._calls.credit.credit(self.spec.name, metadata, steady=True)
        return (
            store,
            call.cache_key,
            held,
            h,
            stored if in_c else MISS,
            _cw.Touch(ram, md, call.cache_key),
            float(saved),
            log,
            float(body) if body is not None else None,
        )


#: Fast hits checked by ``CASH_HITPATH_VERIFY=1`` in this process.
VERIFIED = [0]

#: The ledger parts the scan covers: code, helpers, cached callees, globals
#: and their captures and defaults; and the installed version of a library
#: the call reaches, which `library_versions` reads once per process (the
#: library itself is reached by a global name the scan watches).
_LEDGER_KINDS = frozenset(
    {"calls", "helper", "globals", "captures and defaults of cached function", "installed library"}
)

#: Results a hit hands back as the very object the RAM tier holds.
_DIRECT = frozenset({str, int, float, bool, type(None), bytes, complex})


def _deep_atoms(value: Any, depth: int) -> bool:
    """Is *value* an atom of `_DIRECT`, or a tuple of such, at any depth?"""
    t = type(value)
    if t in _DIRECT:
        return True
    if t is tuple and depth < 32:
        return all(_deep_atoms(item, depth + 1) for item in value)
    return False


def _plain_copy(value: Any, depth: int) -> Any:
    """*value* with each list, dict and tuple in it copied, as the memo key
    reads them by value (`cwatch.memo_key`); anything else as it is."""
    t = type(value)
    if depth > 32:
        return value
    if t is list:
        return [_plain_copy(v, depth + 1) for v in list(value)]
    if t is tuple:
        return tuple(_plain_copy(v, depth + 1) for v in value)
    if t is dict:
        return {k: _plain_copy(v, depth + 1) for k, v in list(value.items())}
    return value


def memo_for(spec: CachedFunction, runner: Any) -> HitMemo | None:
    """*spec*'s memo, made on first use; None when the path is off or the
    function cannot take one (async, ``key=``, declared files, ...)."""
    memo = spec.hit_memo
    if memo is not None:
        return memo
    if not ENABLED or spec.hit_state is None or not plain_spec(spec):
        return None
    memo = HitMemo(spec, runner)
    spec.hit_memo = memo
    return memo


def plain_spec(spec: CachedFunction) -> bool:
    import inspect

    return not (
        inspect.iscoroutinefunction(spec.func)
        or spec.arg_key is not None
        or spec.dynamic_depends_on
        or spec.declared_files
        or spec.seed_params
        or spec.rng_modules
    )
