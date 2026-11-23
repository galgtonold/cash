"""Purity reports kept on disk, so a fresh process does not walk again the
helpers of a cached function whose code has not changed.

The helper walk (`helper_walk.HelperWalk`) reads, parses and audits every
helper a ``@cash.cache`` function reaches, in every new process: for a
library like nilearn, over a thousand bodies before the first call returns,
most of the cost of a run whose results are all cached. `tree_memo` and
`fact_store` keep what follows from a function's text alone; what the walk
decides from what the names are bound to was still worked out again.

So the whole report is stored, beside the module scans (`scan_store`), with
what it was worked out from (`_guards`), and used only when all of that is
the same in the process asking:

* the raw source text of the root and of every callable the walk visited,
  each found again from where its module holds it (an edit, a comment, a
  waiver, a revert, a function moved or rebound is a different text or a
  different object), and its key digest (`callable_identity`);
* of each file those callables are defined in, what the walk took from it
  beyond their own text (`_plan`): the file with its function bodies cut
  out (its imports, its globals, the class bodies, the conditions a
  definition sits under), the whole text where the walk read more of it
  than what it visited (`cash.source_reads`), which globals its functions
  change where the walk asked. A finding's line is kept from the start of
  the function holding it, so lines added above it move it with it. An
  edit to a function no part of the walk read leaves the report as it is;
* what every global, closure cell and default each visited function names
  is bound to now (`_token`: a module by name, a function or class by
  module, name and file and its ``@pure`` / ``@stateful`` mark, any other
  value by its type), and every call-site binding the report records;
* which modules are loaded where a body imports one, and whether a module
  not loaded yet can be read from its source;
* the version of each library the visited functions reach
  (`library_versions`), the Python and cash (the store folder), and
  whether calls are traced (``CASH_TRACE``).

Anything the report holds that cannot be found again by a name in a loaded
module (a closure made by a factory, an object's method, a mock) leaves the
report out of the store: that root is walked in every process, as before.
A report read back holds the live objects its names lead to now, so
`bindings_changed` checks it in this process as it checks a walked one.

``CASH_ANALYSIS_STORE=0`` turns it off with the module scans;
``CASH_ANALYSIS_MEMO_VERIFY=1`` walks every stored report's function again
and raises `StoredReportMismatch` when the report differs.
"""

from __future__ import annotations

import ast
import atexit
import dataclasses
import functools
import hashlib
import inspect
import json
import re
import sys
import types
import weakref
from collections.abc import Callable
from typing import Any

from .. import _weak, source_reads
from .._clock import fine_thread_clock
from .._memo import CODE_OBJECTS, MODULE_ANALYSES, LruMemo
from .._paths import is_pseudo_filename
from ..code_digest import callable_identity
from ..effects import EffectKind
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..install_paths import top_package
from ..loaded_code import code_objects_of
from ..purity import is_pure, is_stateful
from ..source_reading import own_source, read_code_file
from . import scan_store, tree_memo
from .helper_bindings import (
    UNRESOLVED,
    held_ref,
    helper_ref,
    resolve_binding,
    resolve_local_import,
    unloaded_local_imports,
)
from .helper_code import UnwalkableLayers, callable_layers, is_mock
from .purity_report import PurityIssue, PurityReport
from .unloaded_modules import source_digest

__all__ = ["STATS", "StoredReportMismatch", "flush", "load", "save", "status"]

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 8
_KIND = "purity-report"
#: Reports kept per function: its recent texts, and the worlds they ran in.
_MAX_ENTRIES = 8
#: How many ``__wrapped__`` / ``__func__`` steps lead from where a module
#: holds a callable to the one the walk visited.
_MAX_STEPS = 8

STATS = {"loaded": 0, "stored": 0, "missed": 0, "stale": 0, "refused": 0, "cheap": 0, "failed": 0, "verified": 0}


class StoredReportMismatch(AssertionError):
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: a stored report is not what walking
    the function again gives."""


class _Unstorable(Exception):
    """Something the report or the walk holds cannot be found again by name."""


def status() -> dict:
    return dict(STATS)


# --- finding an object again ---


def _wrapped(obj: Any) -> Any:
    try:
        return inspect.getattr_static(obj, "__wrapped__", None)
    except Exception:  # noqa: BLE001 - an object whose type answers oddly
        return None


def _descend(live: Any, target: Any) -> str | None:
    """The steps from *live* to *target*: ``f`` for ``__func__`` (a method
    read off its class), ``p`` for a property's getter or a
    ``cached_property``'s function (read off its class, the descriptor
    itself), ``w`` for ``__wrapped__``, ``r(<module>|<qualname>)`` for the
    implementation a ``functools.singledispatch`` function's ``registry``
    holds for that class; None when none lead."""
    steps = ""
    for _ in range(_MAX_STEPS):
        if live is target:
            return steps
        if type(live) is types.MethodType and type(target) is types.MethodType:
            # Read off an instance, a method is a new object every time:
            # the same function bound to the same object is the same method.
            if live.__func__ is target.__func__ and live.__self__ is target.__self__:
                return steps
        dispatched = _registry_step(live, target)
        if dispatched is not None:
            return steps + dispatched
        if type(live) is types.MethodType:
            live, steps = live.__func__, steps + "f"
            continue
        getter = _getter(live)
        if getter is not None:
            live, steps = getter, steps + "p"
            continue
        inner = _wrapped(live)
        if inner is None:
            return None
        live, steps = inner, steps + "w"
    return None


def _registry(live: Any) -> Any:
    """A ``functools.singledispatch`` function's ``registry`` (class ->
    implementation), or None."""
    if type(live) is not types.FunctionType:
        return None
    registry = live.__dict__.get("registry")
    return registry if isinstance(registry, types.MappingProxyType) else None


def _registry_step(live: Any, target: Any) -> str | None:
    """``r(<module>|<qualname>)`` when *live*'s singledispatch registry
    holds *target* for a class a name leads to: an implementation
    registered under a throwaway name (``def _``), which no module holds
    by its own name. Read back, the registry is asked for that class again
    (`_follow`), and what it holds then is checked like any other visited
    callable."""
    registry = _registry(live)
    if registry is None or type(target) is not types.FunctionType:
        return None
    position = None
    for at, (cls, impl) in enumerate(list(registry.items())):
        if impl is not target:
            continue
        home = _home(cls)
        if home is None or resolve_binding(*home) is not cls:
            # A class no name leads to (a stand-in for a library that is
            # not installed): its place in the registry, which the order
            # the module registers in decides.
            position = at if position is None else position
            continue
        module, chain = home
        if any(c in part for part in (module, *chain) for c in "()|#"):
            continue
        return f"r({module}|{'.'.join(chain)})"
    return None if position is None else f"r(#{position})"


def _getter(live: Any) -> Any:
    """The function a property (its getter) or a ``cached_property`` runs,
    or None: a factory's getter (``scanpy``'s ``preset_property``) has no
    name of its own, and the class attribute holding it is the way back."""
    kind = type(live)
    if kind is property:
        return live.fget if type(live.fget) is types.FunctionType else None
    if kind is functools.cached_property:
        return live.func if type(live.func) is types.FunctionType else None
    return None


def _follow(live: Any, steps: str) -> Any:
    at = 0
    while at < len(steps):
        step = steps[at]
        at += 1
        if step == "f":
            if type(live) is not types.MethodType:
                return UNRESOLVED
            live = live.__func__
        elif step == "p":
            live = _getter(live)
            if live is None:
                return UNRESOLVED
        elif step == "r":
            end = steps.find(")", at)
            if steps[at : at + 1] != "(" or end < 0:
                return UNRESOLVED
            live = _dispatched(_registry(live), steps[at + 1 : end])
            at = end + 1
            if live is UNRESOLVED:
                return UNRESOLVED
        else:
            live = _wrapped(live)
            if live is None:
                return UNRESOLVED
    return live


def _dispatched(registry: Any, spelled: str) -> Any:
    """What *registry* holds for the class *spelled* names (``module|qualname``)
    or at the place it spells (``#<n>``), or ``UNRESOLVED``."""
    if registry is None:
        return UNRESOLVED
    if spelled.startswith("#"):
        try:
            position = int(spelled[1:])
        except ValueError:
            return UNRESOLVED
        impls = list(registry.values())
        return impls[position] if 0 <= position < len(impls) else UNRESOLVED
    module, _, dotted = spelled.partition("|")
    cls = resolve_binding(module, tuple(dotted.split(".")))
    if cls is UNRESOLVED:
        return UNRESOLVED
    try:
        return registry.get(cls, UNRESOLVED)
    except TypeError:  # an unhashable object where a class was
        return UNRESOLVED


def _chain(qualname: str) -> tuple[str, ...]:
    """The attributes that lead to what *qualname* names from its module.
    A private name in a class body (``Parameter.__set_expression``) is held
    under its mangled name, ``_Parameter__set_expression``: read as
    written, no name led to such a method, and every report whose walk
    visited one was walked again in each process."""
    parts = qualname.split(".")
    for i in range(1, len(parts)):
        name, owner = parts[i], parts[i - 1].lstrip("_")
        if name.startswith("__") and not name.endswith("__") and owner:
            parts[i] = f"_{owner}{name}"
    return tuple(parts)


def _home(obj: Any) -> tuple[str, tuple[str, ...]] | None:
    base = obj.__func__ if type(obj) is types.MethodType else obj
    module = getattr(base, "__module__", None) if isinstance(base, (type, types.FunctionType)) else None
    qualname = getattr(base, "__qualname__", None) if module else None
    if not isinstance(module, str) or not isinstance(qualname, str) or "<" in qualname:
        return None
    return module, _chain(qualname)


def _defining_home(obj: Any) -> tuple[str, tuple[str, ...]] | None:
    """The module whose namespace a function was defined in, with its
    qualified name: where it is found when its ``__module__`` was set to
    a package that does not hold it (NumPy's ``set_module`` names
    ``numpy._core.numerictypes.issubclass_`` as ``numpy.issubclass_``,
    which NumPy 2 no longer exports)."""
    base = obj.__func__ if type(obj) is types.MethodType else obj
    if type(base) is not types.FunctionType:
        return None
    module = base.__globals__.get("__name__")
    if not isinstance(module, str) or module == getattr(base, "__module__", None):
        return None
    qualname = base.__qualname__
    if "<" in qualname:
        return None
    return module, _chain(qualname)


def _owner_homes(obj: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Where a class of *obj*'s module holds it under its own name, for a
    function whose qualified name leads nowhere: a method a ``dataclass``
    generated (``__create_fn__.<locals>.__repr__``, held as ``Mount.__repr__``
    behind ``reprlib.recursive_repr``'s wrapper). `_descend` follows such a
    wrapper's ``__wrapped__`` from the class attribute to *obj*. Also a
    function made in a factory whose names ``functools.wraps`` copied from
    another (scanpy's ``_LegacyRng`` sets one per ``Generator`` method, each
    named ``Generator.<method>``): its code still says where it was made."""
    base = obj.__func__ if type(obj) is types.MethodType else obj
    if type(base) is not types.FunctionType:
        return []
    code = base.__code__
    # Before Python 3.11 a code object has no ``co_qualname``: a name that
    # differs from the code's own (``wraps`` copied it) says the same there.
    made_elsewhere = "<" in code.co_qualname if hasattr(code, "co_qualname") else code.co_name != base.__name__
    if "<" not in base.__qualname__ and not made_elsewhere:
        return []
    name = base.__name__
    found = []
    for module in {base.__module__, base.__globals__.get("__name__")}:
        namespace = getattr(sys.modules.get(module) if isinstance(module, str) else None, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for held in list(namespace.values()):
            if not isinstance(held, type) or held.__module__ != module or name not in held.__dict__:
                continue
            qualname = held.__qualname__
            if isinstance(qualname, str) and "<" not in qualname:
                found.append((module, (*qualname.split("."), name)))
    return found


def _dispatcher_homes(obj: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Where the module that defined *obj* holds a ``functools.singledispatch``
    function whose registry holds it: an implementation registered as
    ``def _``, whose own name holds whichever was defined last. `_descend`
    takes the registry step from there."""
    base = obj.__func__ if type(obj) is types.MethodType else obj
    if type(base) is not types.FunctionType:
        return []
    found = []
    for module in dict.fromkeys((base.__module__, base.__globals__.get("__name__"))):
        namespace = getattr(sys.modules.get(module) if isinstance(module, str) else None, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, held in list(namespace.items()):
            registry = _registry(held)
            if registry is not None and any(impl is base for impl in list(registry.values())):
                found.append((module, (name,)))
    return found


def _instance_homes(obj: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Where the module that defined a method's function holds the object
    the method is bound to (``model = Model()``), by the method's name."""
    if type(obj) is not types.MethodType or isinstance(obj.__self__, type):
        return []
    owner, name = obj.__self__, getattr(obj.__func__, "__name__", None)
    module = getattr(obj.__func__, "__module__", None)
    namespace = getattr(sys.modules.get(module) if isinstance(module, str) else None, "__dict__", None)
    if not isinstance(name, str) or not isinstance(namespace, dict):
        return []
    return [(module, (held, name)) for held, value in list(namespace.items()) if value is owner]


def _holder_homes(obj: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Where a module that defines the class a method names as its owner
    holds it: a method a library made for a class of the user's
    (``namedtuple``'s ``P._make``, ``P._replace``, ``P._asdict``), whose
    module is the library's and whose class lives in the user's module.
    Asked only when no other name leads to it: one lookup per loaded
    module."""
    base = obj.__func__ if type(obj) is types.MethodType else obj
    qualname = getattr(base, "__qualname__", None) if type(base) is types.FunctionType else None
    if not isinstance(qualname, str) or "<" in qualname or "." not in qualname:
        return []
    chain = _chain(qualname)
    return [(name, chain) for name in _class_holders(chain[0]) if _holds_own_class(name, chain[0])]


#: Class name -> (how many modules were loaded, the modules that held a
#: class of that name defined in them then): `_class_holders`.
_HOLDERS: dict[str, tuple[int, tuple[str, ...]]] = {}


def _holds_own_class(name: str, attr: str) -> bool:
    namespace = getattr(sys.modules.get(name), "__dict__", None)
    held = namespace.get(attr) if type(namespace) is dict else None
    return isinstance(held, type) and getattr(held, "__module__", None) == name


def _class_holders(attr: str) -> tuple[str, ...]:
    """The loaded modules that hold a class named *attr* defined in them,
    looked up in every module again only once another module was loaded:
    the methods of one class reach `_holder_homes` one after another, and
    each lookup read thousands of namespaces. The caller checks each module
    again; a module that came to hold such a class without a module being
    loaded since is left out, which leaves a report unstored, never wrong."""
    count = len(sys.modules)
    kept = _HOLDERS.get(attr)
    if kept is not None and kept[0] == count:
        return kept[1]
    found = []
    for name, module in list(sys.modules.items()):
        # `_holds_own_class`, without looking the module up again: this
        # reads every loaded module.
        namespace = getattr(module, "__dict__", None)
        held = namespace.get(attr) if type(namespace) is dict else None
        if held is not None and isinstance(held, type) and getattr(held, "__module__", None) == name:
            found.append(name)
    if len(_HOLDERS) > 4096:
        _HOLDERS.clear()
    _HOLDERS[attr] = (count, tuple(found))
    return tuple(found)


def _path_of(obj: Any, *also: tuple[str, tuple[str, ...]] | None) -> list:
    """``[module, chain, steps]`` that lead back to *obj* now: its home
    (module and qualified name), one of *also* (where a caller names it),
    the module it was defined in, a class there that holds it, the object
    a method is bound to, or the class a library made a method for."""
    # The class holders last: they are looked up in every loaded module.
    for paths in (
        (_home(obj), *also, _defining_home(obj), *_owner_homes(obj), *_dispatcher_homes(obj), *_instance_homes(obj)),
        _holder_homes,
    ):
        for path in paths if isinstance(paths, tuple) else paths(obj):
            if path is None:
                continue
            live = resolve_binding(*path)
            if live is UNRESOLVED:
                continue
            steps = _descend(live, obj)
            if steps is not None:
                return [path[0], list(path[1]), steps]
    raise _Unstorable(f"no name leads to {obj!r}")


def _owner_path(method: types.MethodType, named: tuple[str, tuple[str, ...]] | None) -> list:
    """``[module, chain]`` of the object a visited method is bound to: a
    module-level instance (``model = Model()``, ``model.fit`` called), read
    as the name its caller uses, less the method's own name. The walk read
    its methods off that very object, so the report holds while the name
    leads to an object of the same kind defined by the same text
    (`_owners_now`)."""
    if named is None or len(named[1]) < 2:
        raise _Unstorable("a method of an object no name leads to")
    module, chain = named[0], tuple(named[1][:-1])
    if resolve_binding(module, chain) is not method.__self__:
        raise _Unstorable("a method of an object no name leads to")
    return [module, list(chain)]


def _owners_now(owners: list) -> list:
    """What each object a visited method was bound to is now: its kind and
    the text of the module whose name holds it (rebinding the name to
    another object is an edit of that text)."""
    found = []
    for module, chain in owners:
        obj = resolve_binding(module, tuple(chain))
        mod_file = getattr(sys.modules.get(module), "__file__", None)
        text = _file_digest(mod_file) if isinstance(mod_file, str) and not is_pseudo_filename(mod_file) else None
        found.append(None if obj is UNRESOLVED else [_type_name(type(obj)), text])
    return found


def _find(path: Any) -> Any:
    module, chain, steps = path
    live = resolve_binding(module, tuple(chain))
    return UNRESOLVED if live is UNRESOLVED else _follow(live, steps)


# --- what an object is, across processes ---


def _static_str(obj: Any, name: str) -> str | None:
    try:
        value = inspect.getattr_static(obj, name, None)
    except Exception:  # noqa: BLE001 - a user class's own lookup can raise anything; unknown means walk
        return None
    return value if isinstance(value, str) else None


def _token(value: Any) -> list:
    """What *value* is, as another process can tell: what the walk's
    decisions about a name bound to it rest on, short of its code (checked
    for every visited callable on its own, `_object_guard`)."""
    kind = type(value)
    if issubclass(kind, types.ModuleType):
        return ["m", getattr(value, "__name__", None)]
    if is_mock(value):
        raise _Unstorable("a mock")
    if kind is types.MethodType:
        owner = value.__self__
        return ["bm", _token(value.__func__), _token(owner) if isinstance(owner, type) else _type_name(type(owner))]
    if kind in (types.FunctionType, types.BuiltinFunctionType) or isinstance(value, type):
        code = getattr(value, "__code__", None) if kind is types.FunctionType else None
        return [
            "f",
            _type_name(kind),
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", None),
            code.co_filename if code is not None else None,
            is_pure(value),
            is_stateful(value),
            getattr(value, "_cash_cached", False) is True,
        ]
    if isinstance(value, functools.partial):
        return ["p", _token(value.func)]
    token = ["v", _type_name(kind), _static_str(value, "__module__"), _static_str(value, "__qualname__")]
    if callable(value):
        # The functions a callable object runs (`callable_layers`: the
        # ``pyfunc`` of an ``np.vectorize``, a class-based decorator's
        # ``self.fn``), which the walk follows as helpers: by its type
        # alone, ``VEC = np.vectorize(double)`` read the same as
        # ``np.vectorize(triple)``, and a later process served the report
        # walked through ``double``, with its helpers and findings.
        try:
            token.append([_token(layer) for layer in callable_layers(value)])
        except UnwalkableLayers as e:
            raise _Unstorable(str(e)) from e
    return token


def _type_name(kind: type) -> list:
    return [getattr(kind, "__module__", None), getattr(kind, "__qualname__", None)]


def _codes_below(code: types.CodeType) -> list[types.CodeType]:
    found = [code]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found.extend(_codes_below(const))
    return found


_ABSENT = ["absent"]


def _names_guard(fn: types.FunctionType) -> list:
    """What each global, closure cell and default *fn* names is bound to."""
    g = fn.__globals__
    builtins = g.get("__builtins__")
    if isinstance(builtins, types.ModuleType):
        builtins = vars(builtins)
    if not isinstance(builtins, dict):
        builtins = {}
    names: dict[str, list] = {}
    for code in _codes_below(fn.__code__):
        for name in code.co_names:
            if name in names:
                continue
            if name in g:
                names[name] = _token(g[name])
            elif name in builtins:
                names[name] = ["builtin", *_token(builtins[name])]
            else:
                names[name] = _ABSENT
    cells = []
    for cell in fn.__closure__ or ():
        try:
            cells.append(_token(cell.cell_contents))
        except ValueError:  # an empty cell
            cells.append(["empty"])
    defaults = [_token(v) for v in fn.__defaults__ or ()]
    kwdefaults = sorted([k, _token(v)] for k, v in (fn.__kwdefaults__ or {}).items())
    return [sorted(names.items()), cells, defaults, kwdefaults]


def _raw_digest(obj: Any) -> str | None:
    try:
        text = own_source(obj)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=20).hexdigest()


def _file_of(obj: Any) -> str | None:
    if type(obj) is types.FunctionType:
        filename = obj.__code__.co_filename
    else:
        module = sys.modules.get(getattr(obj, "__module__", None) or "")
        filename = getattr(module, "__file__", None)
    if not isinstance(filename, str) or is_pseudo_filename(filename):
        return None
    return filename


def _file_digest(filename: str) -> str | None:
    try:
        return hashlib.blake2b(read_code_file(filename), digest_size=20).hexdigest()
    except OSError:
        return None


def _object_guard(obj: Any) -> list:
    """What *obj*, a callable the walk visited, is now: its token, raw text
    and, for a function, what its names are bound to."""
    names = _names_guard(obj) if type(obj) is types.FunctionType else None
    # Its key digest is checked with the report's helper digests (`_check`).
    return [_token(obj), _raw_digest(obj), names]


def _digest(value: Any) -> str:
    """A short digest of *value*, a JSON-able guard: what a report keeps of
    it (tuples digest as lists, as they read back)."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


# --- one object's guard, shared by every report that visits it ---
#
# A library's cached functions reach the same helpers: forty roots of one
# package visit a few hundred functions some five thousand times. Each
# object's guard (`_object_guard`) is worked out once per process and kept
# with what it was worked out from (`_stamp`): the object, its code, its
# marks, the digest of its file's text, and for each name, cell and default
# the module, function or class it holds (by identity, as `bindings_changed`
# holds an in-process report) or the token of any other value. It is served
# again only while all of that is the same, so a rebind, a new code object,
# a patched global or an edited file works it out again. What depends on
# more than the object (imports, unbound names, loaded modules, library
# versions) is worked out for every report.
#
# The memo holds every object only by a weak reference (`_kept`): a class
# or function the user drops goes, and an entry whose object has gone
# matches nothing, though another object take its id.

_MISSING = object()
#: id(object) -> (weak reference to it, kept stamp, digest of its guard).
_OBJECT_GUARDS: LruMemo[int, tuple[Any, tuple, str]] = LruMemo(CODE_OBJECTS)
#: id(code object) -> (weak reference to it, the names it and the code
#: below it read, in order).
_CODE_NAMES: LruMemo[int, tuple[Any, tuple[str, ...]]] = LruMemo(CODE_OBJECTS)


def _code_names(code: types.CodeType) -> tuple[str, ...]:
    kept = _CODE_NAMES.get(id(code))
    if kept is not None and kept[0]() is code:
        return kept[1]
    found: dict[str, None] = {}
    for below in _codes_below(code):
        found.update(dict.fromkeys(below.co_names))
    names = tuple(found)
    _CODE_NAMES[id(code)] = (_weak.ref(code), names)
    return names


#: Kinds whose values a stamp holds as themselves, compared by identity
#: (and kept by a weak reference). A builtin function is held by its token:
#: a bound one (``items.append``) holds the object it is bound to.
_HELD_KINDS = frozenset({types.FunctionType, types.ModuleType, type})
#: Built-in kinds with no instance dict, which no code can change: the
#: token of each of their values is the kind's (`_token` reads only the
#: kind and what it holds), worked out once.
_PLAIN_KINDS = frozenset(
    {int, float, complex, str, bytes, bool, type(None), tuple, frozenset, list, dict, set, bytearray, range, slice}
)
_PLAIN_TOKENS: dict[type, Any] = {}


def _value_stamp(value: Any) -> Any:
    """*value* as a stamp holds it: a module, function or class itself
    (compared by identity, kept by a weak reference), anything else by its
    token (a value the memo held could not be dropped, and many cannot be
    referenced weakly)."""
    kind = type(value)
    if kind in _HELD_KINDS:
        return value
    if kind in _PLAIN_KINDS:
        token = _PLAIN_TOKENS.get(kind)
        if token is None:
            token = _PLAIN_TOKENS[kind] = _Token(json.dumps(_token(value)))
        return token
    if issubclass(kind, types.ModuleType) or isinstance(value, type):
        return value
    return _Token(json.dumps(_token(value)))


def _kept(stamp: tuple) -> tuple | None:
    """*stamp* as the memo keeps it: each object in it by a weak reference.
    None when one cannot be referenced weakly (it is then not kept)."""
    out: list[Any] = []
    for item in stamp:
        kind = type(item)
        if kind is tuple:
            item = _kept(item)
            if item is None:
                return None
        elif kind is not _Token and item is not _MISSING:
            try:
                item = _weak.ref(item)
            except TypeError:
                return None
        out.append(item)
    return tuple(out)


class _Token(str):
    """A token spelled as text in a stamp: compared by equality."""

    __slots__ = ()


def _stamp(obj: Any, file_digest: str | None) -> tuple:
    """What *obj*'s guard is worked out from (see above)."""
    if type(obj) is not types.FunctionType:
        # A class or builtin: its token is cheap to read, its text is not.
        return (_Token(file_digest or ""), _Token(json.dumps(_token(obj))))
    code = obj.__code__
    # Its globals need not be in the stamp: a function's ``__globals__``
    # cannot be rebound, and what its names hold now is.
    g = obj.__globals__
    builtins = g.get("__builtins__")
    if isinstance(builtins, types.ModuleType):
        builtins = vars(builtins)
    if not isinstance(builtins, dict):
        builtins = {}
    reads: list[Any] = []
    for name in _code_names(code):
        value = g.get(name, _MISSING)
        if value is _MISSING:
            reads.append(_MISSING)  # read from the builtins, or absent
            value = builtins.get(name, _MISSING)
        reads.append(_MISSING if value is _MISSING else _value_stamp(value))
    cells: list[Any] = []
    for cell in obj.__closure__ or ():
        try:
            cells.append(_value_stamp(cell.cell_contents))
        except ValueError:  # an empty cell
            cells.append(_MISSING)
    marks = obj.__dict__
    return (
        _Token(file_digest or ""),
        code,
        _Token(obj.__module__ or ""),
        _Token(obj.__qualname__),
        _Token(repr((marks.get("_cash_pure"), marks.get("_cash_stateful"), marks.get("_cash_cached")))),
        tuple(reads),
        tuple(cells),
        tuple(_value_stamp(v) for v in obj.__defaults__ or ()),
        tuple((_Token(k), _value_stamp(v)) for k, v in sorted((obj.__kwdefaults__ or {}).items())),
    )


def _same_stamp(kept: tuple, now: tuple) -> bool:
    """Is *now*, a stamp, what *kept* (`_kept`) was made from: the same
    objects, alive, and the same tokens?"""
    if len(kept) != len(now):
        return False
    for x, y in zip(kept, now):
        kind = type(x)
        if kind is weakref.ReferenceType:
            if x() is y and y is not None:
                continue
        elif x is y:  # the missing mark
            continue
        elif kind is _Token:
            if type(y) is _Token and x == y:
                continue
        elif kind is tuple and type(y) is tuple and _same_stamp(x, y):
            continue
        return False
    return True


def _object_digest(obj: Any, file_digest: str | None) -> str:
    """The digest of `_object_guard` (*obj*), worked out again only when
    what it rests on is not the same objects (see above)."""
    stamp = _stamp(obj, file_digest)
    key = id(obj)
    kept = _OBJECT_GUARDS.get(key)
    if kept is not None and kept[0]() is obj and _same_stamp(kept[1], stamp):
        return kept[2]
    digest = _digest(_object_guard(obj))
    keep = _kept((obj, stamp))
    if keep is not None:
        _OBJECT_GUARDS[key] = (keep[0], keep[1], digest)
    return digest


#: id(function) -> (weak references to it and to its code, its file's
#: digest, `callable_identity`).
_IDENTITIES: LruMemo[int, tuple[Any, types.CodeType, str, str]] = LruMemo(CODE_OBJECTS)


def _identity(live: Any, files: dict[str, str | None]) -> str:
    """`callable_identity` (*live*), the key digest a report keeps for each
    helper, kept per process for a plain function while its code and the
    text of its file (digested in *files*, the file digests of this check)
    are the same: its own source text is what it digests, or failing that
    its code. A wrapper (``__wrapped__``), anything else, or a function
    without a file is digested anew."""
    if type(live) is not types.FunctionType or "__wrapped__" in live.__dict__:
        return callable_identity(live)
    filename = _file_of(live)
    if filename is None:
        return callable_identity(live)
    if filename not in files:
        files[filename] = _file_digest(filename)
    file_digest = files[filename]
    if file_digest is None:
        return callable_identity(live)
    code = live.__code__
    kept = _IDENTITIES.get(id(live))
    if kept is not None and kept[0]() is live and kept[1]() is code and kept[2] == file_digest:
        return kept[3]
    identity = callable_identity(live)
    _IDENTITIES[id(live)] = (_weak.ref(live), _weak.ref(code), file_digest, identity)
    return identity


# --- what a report checks of the files it read ---
#
# A report is not checked against the whole text of each file its helpers
# are in: an edit to a function the walk never read would make every report
# that visits any other function of that file stale, and a commit touching
# one module had every cached function reaching it walked again. What the
# walk took from a file is checked instead:
#
# * the text of each callable it visited, on its own (`_object_guard`);
# * the file's outline (`_outline_of`): its text with the body of every
#   function cut out, which holds its imports, its globals, the class
#   bodies, the decorators, the signatures and defaults, every ``if`` a
#   definition sits under, and a module ``__getattr__`` whole;
# * the whole text of a file the walk read whole, or of one holding a
#   callable whose text it read without visiting it (`cash.source_reads`
#   notes both), or a visited callable whose own text cannot be checked
#   (`_plan`);
# * which globals each module whose global-change scan it read has its
#   functions change (`mutable_globals.module_modified_globals`), the one
#   answer it takes from bodies it does not visit;
# * nothing of where a function starts: a finding's line is kept as an
#   offset into the visited function holding it, and numbered again from
#   where it starts when read back (`_anchored`).

_OUTLINE_KIND = "file-outline"
#: (path, digest of its text) -> digest of its outline.
_OUTLINES: LruMemo[tuple[str, str], str] = LruMemo(MODULE_ANALYSES)
#: Functions kept whole in an outline: what a name a module or class does
#: not bind reads as.
_WHOLE_FUNCTIONS = frozenset({"__getattr__", "__dir__"})
#: Comments that say something to cash, wherever they are.
_CASH_COMMENT = re.compile(rb"#[^\n]*@cash[^\n]*")


def _cut_bodies(tree: ast.Module) -> None:
    """Replace the body of every function in *tree* by ``...``, but a
    `_WHOLE_FUNCTIONS` one's. A signature, a default or a decorator holds
    no body (a lambda's is an expression) and stays."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name not in _WHOLE_FUNCTIONS:
                    child.body = [ast.Expr(ast.Constant(...))]
                continue
            stack.append(child)


def _outline_of(text: bytes) -> str:
    """The digest of *text*'s outline (see above): its tree with every
    function body cut, without positions, and every ``@cash`` comment.
    Text that does not parse is its own outline."""
    try:
        tree = ast.parse(text)
        _cut_bodies(tree)
        dumped = ast.dump(tree).encode("utf-8", "surrogatepass")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "text:" + hashlib.blake2b(text, digest_size=20).hexdigest()
    h = hashlib.blake2b(dumped, digest_size=20)
    for comment in _CASH_COMMENT.findall(text):
        h.update(b"\0")
        h.update(comment)
    return h.hexdigest()


def _outline_digest(filename: str, digest: str) -> str | None:
    """The outline digest of the file at *filename* while its text digests
    to *digest*; worked out once per text, across processes too
    (`scan_store.save_record`). None when the file holds another text now."""
    key = (filename, digest)
    known = _OUTLINES.get(key)
    if known is not None:
        return known
    record = scan_store.load_record(_OUTLINE_KIND, filename, digest)
    outline = record.get("outline") if isinstance(record, dict) else None
    if not isinstance(outline, str) or tree_memo.VERIFY:
        try:
            text = read_code_file(filename)
        except OSError:
            return None
        if hashlib.blake2b(text, digest_size=20).hexdigest() != digest:
            return None
        fresh = _outline_of(text)
        if isinstance(outline, str) and outline != fresh:
            raise StoredReportMismatch(f"stored outline of {filename} differs from working it out again")
        if not isinstance(outline, str):
            scan_store.save_record(_OUTLINE_KIND, filename, digest, {"outline": fresh})
        outline = fresh
    _OUTLINES[key] = outline
    return outline


def _modified_globals(module_name: str) -> list[str] | None:
    """The globals the functions of the module named *module_name* change,
    as the walk's global-change scan finds them now; None when it is not loaded."""
    from .mutable_globals import module_modified_globals

    module = sys.modules.get(module_name)
    if module is None:
        return None
    return sorted(module_modified_globals(module))


def _read_file(obj: Any) -> str | None:
    """The file whose text was read as *obj*'s (`cash.source_reads`)."""
    if type(obj) is types.MethodType:
        obj = obj.__func__
    code = obj if type(obj) is types.CodeType else getattr(obj, "__code__", None)
    if type(code) is types.CodeType:
        filename = code.co_filename
        return None if is_pseudo_filename(filename) else filename
    try:
        filename = inspect.getsourcefile(obj)
    except (TypeError, OSError):
        filename = None
    return filename if filename and not is_pseudo_filename(filename) else _file_of(obj)


def _spans(fn: types.FunctionType, line: int) -> bool:
    """Is *line* of *fn*'s file one of the lines its own text holds?"""
    first = fn.__code__.co_firstlineno
    last = max(
        (ln for code in _codes_below(fn.__code__) for _, _, ln in code.co_lines() if ln is not None), default=first
    )
    return first <= line <= last


def _plan(objects: list[Any], paths: list[list], report: PurityReport, reads: source_reads.Reads) -> dict:
    """Which files a report is checked against whole, which modules'
    global-change scans, and where each finding's line is anchored (see
    above). A function or class whose text the walk read without visiting it, and
    that a name leads to, is added to *objects* (at its path in *paths*) to
    be checked by its own text; any other read checks its file whole."""
    whole: set[str] = {f for f in reads.files if not is_pseudo_filename(f)}
    covered: set[int] = set()
    for obj in objects:
        covered.add(id(obj))
        if type(obj) is types.FunctionType:
            covered.add(id(obj.__code__))
        own = type(obj) is types.FunctionType or isinstance(obj, type)
        if not own or _raw_digest(obj) is None:
            filename = _file_of(obj)
            if filename is not None:
                whole.add(filename)
    for obj in reads.objects.values():
        base = obj.__func__ if type(obj) is types.MethodType else obj
        if id(base) in covered or id(getattr(base, "__code__", None)) in covered:
            continue
        if type(base) is types.FunctionType or isinstance(base, type):
            try:
                path = _path_of(base)
            except _Unstorable:
                path = None
            if path is not None and _raw_digest(base) is not None:
                covered.add(id(base))
                objects.append(base)
                paths.append(path)
                continue
        filename = _read_file(base)
        if filename is None:
            if type(base) is types.CodeType or isinstance(base, (types.FunctionType, type)):
                continue  # code with no file: nothing in a file decides it
            raise _Unstorable("a text read from no file")
        whole.add(filename)
    # Each finding's line, as an offset into a visited function that holds
    # it: read back, it is numbered from where that function starts then,
    # which an edit above it moves. A line no visited function holds
    # checks its file whole.
    anchors = []
    for at, issue in enumerate(report.issues):
        for field, line in enumerate((issue.line, issue.waiver_line)):
            if not line or not issue.filename:
                continue
            holder = next(
                (
                    i
                    for i, f in enumerate(objects)
                    if type(f) is types.FunctionType and _file_of(f) == issue.filename and _spans(f, line)
                ),
                None,
            )
            if holder is None:
                whole.add(issue.filename)
            else:
                anchors.append([at, field, holder, line - objects[holder].__code__.co_firstlineno])
    return {"whole": sorted(whole), "scans": sorted(reads.scans), "anchors": anchors}


# --- the guards: what a report was worked out from ---


def _guards(
    objects: list[Any],
    bindings: list[Any],
    cached: list[Any],
    local_imports: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
    unbound: tuple[tuple[str, str], ...],
    unloaded: tuple[str, ...],
    root_module: str | None,
    files: dict[str, str | None],
    plan: dict,
) -> dict:
    """What the report was worked out from, now. *files* (filled here) maps
    each file read to the digest of its text, for the helper digests.
    *plan* (`_plan`) says which files are checked whole (the others by
    their outline) and which modules' global-change scans."""
    from ..decorator.library_versions import library_versions

    digests = []
    visited_files = set()
    for obj in objects:
        filename = _file_of(obj)
        if filename is not None:
            visited_files.add(filename)
            if filename not in files:
                files[filename] = _file_digest(filename)
        digests.append(_object_digest(obj, None if filename is None else files[filename]))
    whole = set(plan["whole"])
    texts = []
    for filename in sorted(visited_files | whole):
        if filename not in files:
            files[filename] = _file_digest(filename)
        digest = files[filename]
        if digest is None or filename in whole:
            texts.append([filename, "whole", digest])
        else:
            texts.append([filename, "outline", _outline_digest(filename, digest)])
    imports = []
    for module_name, prefix, dotted in local_imports:
        found = unloaded_local_imports(module_name, prefix, dotted, root_module)
        obj = resolve_local_import(module_name, prefix)
        imports.append(
            [
                module_name,
                list(prefix),
                list(dotted),
                None if obj is None else _token(obj),
                [[name, source_digest(name, root_module) is None] for name in found],
            ]
        )
    from .helper_walk import untraced

    functions = [o for o in objects if type(o) is types.FunctionType]
    return {
        "objects": digests,
        "bindings": [None if b is UNRESOLVED else _token(b) for b in bindings],
        "cached": [None if c is UNRESOLVED else _token(c) for c in cached],
        "files": texts,
        "scans": [[m, _modified_globals(m)] for m in plan["scans"]],
        "imports": imports,
        "unbound": [[m, n, m in sys.modules and n in vars(sys.modules[m])] for m, n in unbound],
        "unloaded": [[n, n in sys.modules] for n in unloaded],
        "libraries": [list(v) for v in library_versions(functions, top_package(root_module))],
        "untraced": untraced(),
    }


# --- a report spelled without live objects, and read back ---


def _spell_issue(issue: PurityIssue) -> list:
    return [
        issue.kind,
        issue.description,
        issue.where,
        issue.line,
        issue.filename,
        issue.subject,
        None if issue.effect_kind is None else issue.effect_kind.value,
        issue.waiver_line,
    ]


def _read_issue(spelled: list) -> PurityIssue:
    kind, description, where, line, filename, subject, effect, waiver_line = spelled
    return PurityIssue(
        kind=kind,
        description=description,
        where=where,
        line=line,
        filename=filename,
        subject=subject,
        effect_kind=None if effect is None else EffectKind(effect),
        waiver_line=waiver_line,
    )


def _spell(report: PurityReport, objects: list[Any], paths: list[list], files: dict[str, str | None]) -> dict:
    """*report* as JSON: each live object it holds as the index of its path
    in *paths* (*objects* are the visited ones, found at those paths)."""
    if report.unwalkable or report.unkeyable:
        raise _Unstorable("a walk that did not finish, or a mock")
    index = {id(o): i for i, o in enumerate(objects)}

    def at(obj: Any) -> int:
        if type(obj) is types.MethodType and not isinstance(obj.__self__, type):
            # A method of a module-level instance (`_owner_path`): read
            # back as its function, the code it runs.
            obj = obj.__func__
        i = index.get(id(obj))
        if i is None:
            index[id(obj)] = i = len(paths)
            objects.append(obj)
            paths.append(_path_of(obj))
        return i

    walked = []
    for ref, codes in report.walked_code:
        owner = ref()
        if owner is None or code_objects_of(owner) != codes:
            raise _Unstorable("walked code that changed")
        walked.append(at(owner))
    cached = []
    for ref in report.cached_callees:
        callee = ref()
        if callee is None:
            raise _Unstorable("a cached callee that is gone")
        cached.append(_path_of(callee))
    whole = []
    for ref in report.audited_whole:
        fn = ref()
        if fn is None:
            raise _Unstorable("an audited function that is gone")
        whole.append(at(fn))
    bindings = []
    for module_name, chain, ref in report.helper_bindings:
        live = resolve_binding(module_name, chain)
        held = ref()
        if live is UNRESOLVED or held is None or not _same(live, held):
            raise _Unstorable("a binding that moved")
        bindings.append([module_name, list(chain)])
    # A helper no caller's name or home leads to as itself (the function an
    # ``lru_cache`` or ``functools.wraps`` wrapper holds): found again
    # through the wrapper its home holds.
    held = []
    for qualname, ref in report.helper_objects.items():
        obj = ref()
        if obj is None:
            raise _Unstorable(f"helper {qualname} is gone")
        held.append([qualname, _path_of(obj, report.helper_resolution_paths.get(qualname))])
    found = {q: _find(p) for q, p in held}
    for qualname, digest in report.helper_source_hashes.items():
        path = report.helper_resolution_paths.get(qualname)
        if path is not None:
            live = resolve_binding(*path)
        else:
            live = found.get(qualname, objects[0])  # the root has no path
        if live is UNRESOLVED or _identity(live, files) != digest:
            raise _Unstorable(f"helper {qualname} is not found again by its path")
    return {
        "issues": [_spell_issue(i) for i in report.issues],
        "hashes": list(report.helper_source_hashes.items()),
        "paths": [[q, m, list(c)] for q, (m, c) in report.helper_resolution_paths.items()],
        "held": held,
        "opaque": list(report.opaque_callees),
        "bindings": bindings,
        "unbound": [list(u) for u in report.unbound_names],
        "unloaded": list(report.unloaded_modules),
        "walked": walked,
        "waived": sorted([m, list(c)] for m, c in report.waived_bindings),
        "environment": sorted(list(e) for e in report.environment_reads),
        "cached": cached,
        "whole": whole,
        # Which classes are only tested against: without it a report read
        # back folds what their methods read again (`check_targets`).
        "check_only": sorted(report.check_only_classes),
        "state_changing": sorted(report.state_changing_helpers),
        "writes_in_place": report.writes_in_place,
    }


def _found(path: list) -> Any:
    obj = _find(path)
    if obj is UNRESOLVED:
        raise _Unstorable("a helper that is gone")
    return obj


def _same(live: Any, held: Any) -> bool:
    if type(live) is types.MethodType and type(held) is types.MethodType:
        return live.__func__ is held.__func__ and live.__self__ is held.__self__
    return live is held


def _read(spelled: dict, objects: list[Any]) -> PurityReport:
    """The report *spelled* stands for, holding the live *objects*."""
    bindings = []
    for module_name, chain in spelled["bindings"]:
        chain = tuple(chain)
        live = resolve_binding(module_name, chain)
        if live is UNRESOLVED:
            raise _Unstorable("a binding that is gone")
        bindings.append((module_name, chain, held_ref(live)))
    walked = []
    for i in spelled["walked"]:
        owner = objects[i]
        walked.append((_weak.ref(owner), code_objects_of(owner)))
    cached = []
    for path in spelled["cached"]:
        callee = _find(path)
        if callee is UNRESOLVED:
            raise _Unstorable("a cached callee that is gone")
        cached.append(held_ref(callee))
    return PurityReport(
        issues=tuple(_read_issue(i) for i in spelled["issues"]),
        helper_source_hashes=dict(spelled["hashes"]),
        helper_resolution_paths={q: (m, tuple(c)) for q, m, c in spelled["paths"]},
        helper_objects={q: helper_ref(_found(p)) for q, p in spelled["held"]},
        opaque_callees=tuple(spelled["opaque"]),
        helper_bindings=tuple(bindings),
        unbound_names=tuple(tuple(u) for u in spelled["unbound"]),
        unloaded_modules=tuple(spelled["unloaded"]),
        waived_bindings=frozenset((m, tuple(c)) for m, c in spelled["waived"]),
        unkeyable=(),
        walked_code=tuple(walked),
        environment_reads=frozenset(tuple(e) for e in spelled["environment"]),
        cached_callees=tuple(cached),
        unwalkable="",
        audited_whole=tuple(_weak.ref(objects[i]) for i in spelled["whole"]),
        check_only_classes=frozenset(spelled["check_only"]),
        state_changing_helpers=frozenset(spelled["state_changing"]),
        writes_in_place=bool(spelled["writes_in_place"]),
    )


# --- the root, and where its reports are kept ---


def _root_path(func: Any) -> tuple[str, tuple[str, ...], str] | None:
    """Where *func*'s reports are kept: where its code is written, its
    module, qualified name, file and first line (`_place`). Only a plain
    function is stored (a function an ``lru_cache`` or ``functools.wraps``
    wrapper holds is found through the wrapper).

    Only the store's slot: an entry proves it is about *func* by the path
    it records for it (`_root_found`). Named by where the code is, not by a
    name that leads to it, because finding a name can read every loaded
    module (`_holder_homes`), in every process, for each root."""
    if type(func) is not types.FunctionType:
        return None
    place = _place(func)
    if place is None:
        return None
    # Not by its first line, which an edit above it moves (`_text`).
    module, (qualname, filename, _first), _steps = place
    return module, (qualname, filename), _SLOT


#: The steps of a root's slot (`_root_path`): its module, name and file.
_SLOT = "@slot"
#: The steps of a root path `_root_spelled` names by `_place`: no name leads to it.
_CODE_ROOT = "@"


def _place(func: types.FunctionType) -> tuple[str, tuple[str, ...], str] | None:
    """*func*'s module, qualified name, file and first line, or None for code
    with no file of its own."""
    code = func.__code__
    module = func.__module__
    qualname = func.__qualname__
    if not isinstance(module, str) or not isinstance(qualname, str) or is_pseudo_filename(code.co_filename):
        return None
    return module, (qualname, code.co_filename, str(code.co_firstlineno)), _CODE_ROOT


def _code_root(func: types.FunctionType) -> tuple[str, tuple[str, ...], str] | None:
    """`_place`, for a root no name leads to: a function a library decorates
    into another object (``typing.Literal``, whose name holds the
    ``_SpecialForm`` built around it) is analysed when a call reaches it as
    data, in every process, and nothing but where it is written finds it
    again.

    Only a function defined at a module's top level or in a class there,
    running in that module's namespace, with no closure: nothing but its
    text (`_text`) and what its names are bound to (the guards) tells two
    such functions apart. A closure is not: the report of one belongs to
    that function object alone, which another process cannot name."""
    if func.__closure__ or "<" in func.__qualname__:
        return None
    if func.__globals__ is not getattr(sys.modules.get(func.__module__), "__dict__", None):
        return None
    return _place(func)


def _root_found(path: list, func: Any) -> Any:
    """What the root path *path* of a stored entry leads to, checked by
    `_check`: *func* when `_code_root` spells it so, as `_find` otherwise."""
    if path[2] != _CODE_ROOT:
        return _find(path)
    root = _code_root(func) if type(func) is types.FunctionType else None
    return func if root is not None and _spelled_root(root) == path else UNRESOLVED


def _root_spelled(func: Any) -> list:
    """The path `_entry` and `verify` record for the root: a name that leads
    to it, or where `_code_root` says it is written."""
    try:
        return _path_of(func)
    except _Unstorable:
        root = _code_root(func) if type(func) is types.FunctionType else None
        if root is None:
            raise
        return _spelled_root(root)


def _spelled_root(root: tuple[str, tuple[str, ...], str]) -> list:
    return [root[0], list(root[1]), root[2]]


def _file(root: tuple[str, tuple[str, ...], str], directory: Any = None) -> Any:
    """The file of *root*'s reports in *directory*, by default `store_dir`."""
    if directory is None:
        directory = scan_store.store_dir()
    if directory is None:
        return None
    name = f"{_KIND}\0{root[0]}\0{'.'.join(root[1])}\0{root[2]}"
    slot = hashlib.blake2b(name.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
    return directory / f"{slot}.json"


def _entries(file: Any, root: tuple[str, tuple[str, ...], str]) -> list[dict]:
    data = scan_store.read_json(file)
    if not isinstance(data, dict) or data.get("kind") != _KIND or data.get("format") != _FORMAT:
        return []
    if data.get("root") != _spelled_root(root):
        return []
    entries = data.get("entries")
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


def _text(func: Any) -> str | None:
    """The root's own text and file. Not its first line: an edit above it
    moves that and nothing the report holds, but the lines of its findings
    (`_plan`)."""
    digest = _raw_digest(func)
    return None if digest is None else f"{digest}:{func.__code__.co_filename}"


def load(func: Callable[..., Any]) -> PurityReport | None:
    """The report an earlier process stored for *func*, if everything it was
    worked out from is the same now; else None."""
    root = _root_path(func)
    if root is None:
        return None
    file = _file(root)
    if file is None:
        return None
    text = _text(func)
    if text is None:
        return None
    entries = [e for e in _entries(file, root) if e.get("text") == text]
    if not entries:
        STATS["missed"] += 1
        return None
    for entry in reversed(entries):
        try:
            report = _check(entry, func)
        except Exception:  # noqa: BLE001 - an entry that cannot be checked is stale: walk, and store anew
            report = None
        if report is not None:
            STATS["loaded"] += 1
            return report
    STATS["stale"] += 1
    return None


def _check(entry: dict, func: Any) -> PurityReport | None:
    objects = []
    for at, path in enumerate(entry["paths"]):
        obj = _root_found(path, func) if at == 0 else _find(path)
        if obj is UNRESOLVED:
            return None
        objects.append(obj)
    if not objects or objects[0] is not func:
        return None
    spelled = entry["report"]
    bindings = [resolve_binding(m, tuple(c)) for m, c in spelled["bindings"]]
    cached = [_find(p) for p in spelled["cached"]]
    imports = [(m, tuple(p), tuple(d)) for m, p, d in entry["imports"]]
    files: dict[str, str | None] = {}
    now = _guards(
        objects,
        bindings,
        cached,
        imports,
        tuple(tuple(u) for u in spelled["unbound"]),
        tuple(spelled["unloaded"]),
        func.__module__,
        files,
        entry["plan"],
    )
    now["owners"] = _owners_now(entry.get("owners", []))
    if _digest(now) != entry["guards"]:
        return None
    # Each helper as the decorator finds it on every call: through the
    # name its caller uses, which may not be its home.
    paths = {q: (m, tuple(c)) for q, m, c in spelled["paths"]}
    held = {q: _find(p) for q, p in spelled["held"]}
    for qualname, digest in spelled["hashes"]:
        path = paths.get(qualname)
        live = held.get(qualname, func) if path is None else resolve_binding(*path)  # the root has no path
        if live is UNRESOLVED or _identity(live, files) != digest:
            return None
    return _anchored(_read(spelled, objects), entry["plan"]["anchors"], objects)


def _anchored(report: PurityReport, anchors: list, objects: list[Any]) -> PurityReport:
    """*report* with each anchored finding line (`_plan`) numbered from
    where the function holding it starts now."""
    if not anchors:
        return report
    issues = list(report.issues)
    for at, field, holder, offset in anchors:
        line = objects[holder].__code__.co_firstlineno + offset
        name = "line" if field == 0 else "waiver_line"
        issues[at] = dataclasses.replace(issues[at], **{name: line})
    return dataclasses.replace(report, issues=tuple(issues))


def _as_json(value: Any) -> Any:
    """*value* as a JSON round trip gives it back: tuples as lists."""
    if isinstance(value, dict):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


#: Records waiting to be written: a process that never makes a cache dir
#: keeps only the most recent.
_MAX_PENDING = 256
#: Records worked out in this process and not written yet: (the store
#: folder of the call that worked it out, root, entry).
_PENDING: list[tuple[Any, tuple[str, tuple[str, ...], str], dict]] = []
#: Records worked out before they are written together (and the rest at
#: exit): a file written per report cost more than the walks it spared.
_FLUSH_EVERY = 64
#: A walk that took less CPU time than working out its guards (which
#: reading the report back does again) and reading its file is not stored:
#: walking it again costs less than checking it would. CPU time of this
#: thread, so a busy machine does not tip it -- where that clock is fine:
#: on Windows it moves in 15.6 ms ticks, read every walk as 0 s and so as
#: too cheap to store, and nothing was ever stored (`fine_thread_clock`).
_READ_COST = 0.0005


def save(func: Callable[..., Any], report: PurityReport, walk: Any, seconds: float | None = None) -> None:
    """Keep *report*, just walked from *func* by *walk* in *seconds* (CPU
    time of this thread where that clock is fine, `fine_thread_clock`), for
    later processes. Worked
    out now, written in batches and at exit (`flush`), once the cache dir
    exists."""
    root = _root_path(func)
    if root is None:
        return
    # The folder of the call walking *func*, taken now: when the record is
    # written (at exit above all) no call runs (`scan_store.target_dir`).
    directory = scan_store.target_dir()
    if directory is None:
        return
    text = _text(func)
    if text is None:
        return
    clock = fine_thread_clock()
    start = clock()
    try:
        entry = _entry(func, report, walk, text)
    except Exception:  # noqa: BLE001 - a report that cannot be spelled is not stored; the walk stands
        STATS["refused"] += 1
        return
    if seconds is not None and seconds < clock() - start + _READ_COST:
        STATS["cheap"] += 1
        return
    _PENDING.append((directory, root, entry))
    del _PENDING[:-_MAX_PENDING]
    if len(_PENDING) >= _FLUSH_EVERY:
        flush()


def _entry(func: Any, report: PurityReport, walk: Any, text: str) -> dict:
    objects: list[Any] = [func]
    paths: list[list] = [_root_spelled(func)]
    seen = {id(func)}
    names = walk.visited_names()
    owners: list[list] = []
    for obj in walk.visited():
        base = obj.__func__ if type(obj) is types.MethodType else obj
        if id(base) in seen:
            continue
        qualname = names.get(id(obj))
        named = report.helper_resolution_paths.get(qualname) if qualname else None
        if type(obj) is types.MethodType and not isinstance(obj.__self__, type):
            owners.append(_owner_path(obj, named))
        seen.add(id(base))
        objects.append(base)
        paths.append(_path_of(base, named))
    files: dict[str, str | None] = {}
    spelled = _spell(report, objects, paths, files)
    bindings = [resolve_binding(m, tuple(c)) for m, c in spelled["bindings"]]
    cached = [_find(p) for p in spelled["cached"]]
    imports = walk.local_imports_read()
    plan = _plan(objects, paths, report, walk.source_read())
    guards = _guards(
        objects,
        bindings,
        cached,
        imports,
        report.unbound_names,
        report.unloaded_modules,
        func.__module__,
        files,
        plan,
    )
    guards["owners"] = _owners_now(owners)
    return {
        "text": text,
        "paths": paths,
        "owners": owners,
        "report": spelled,
        "imports": [[m, list(p), list(d)] for m, p, d in imports],
        "plan": plan,
        # Only the digest of the guards: they are worked out again to check.
        "guards": _digest(guards),
    }


def flush(*, at_exit: bool = False) -> None:
    """Write the records worked out so far, each into the folder of the call
    that worked it out, once its cache dir exists; the others stay held
    (`scan_store.written_to`)."""
    if not _PENDING:
        return
    try:
        where = {d: scan_store.written_to(d, at_exit=at_exit) for d in {d for d, _, _ in _PENDING}}
    except Exception:  # noqa: BLE001 - at exit: a store that cannot be found keeps nothing
        return
    pending = [p for p in _PENDING if where[p[0]] is not None]
    _PENDING[:] = [p for p in _PENDING if where[p[0]] is None]
    by_root: dict[tuple, list[dict]] = {}
    for directory, root, entry in pending:
        by_root.setdefault((where[directory], root), []).append(entry)
    for (directory, root), new in by_root.items():
        file = _file(root, directory)
        # Most recently worked out last: a full file drops the oldest.
        entries = [e for e in _entries(file, root) if e not in new]
        entries = (entries + new)[-_MAX_ENTRIES:]
        data = {"kind": _KIND, "format": _FORMAT, "root": _spelled_root(root), "entries": entries}
        if scan_store.write_json(file.parent, file, data):
            STATS["stored"] += 1


def verify(func: Any, loaded: PurityReport, walked: PurityReport) -> None:
    """``CASH_ANALYSIS_MEMO_VERIFY=1``: *loaded*, read from the store, must
    be the report walking *func* again gave."""
    STATS["verified"] += 1
    try:
        objects: list[Any] = [func]
        paths: list[list] = [_root_spelled(func)]
        want = _spell(walked, objects, paths, {})
        objects2: list[Any] = [func]
        paths2: list[list] = [_root_spelled(func)]
        got = _spell(loaded, objects2, paths2, {})
    except _Unstorable as e:
        raise StoredReportMismatch(f"stored report of {func.__qualname__}: {e}") from None
    if _as_json(want) != _as_json(got) or paths != paths2:
        raise StoredReportMismatch(f"stored report of {func.__qualname__} differs from walking it again")


atexit.register(flush, at_exit=True)
