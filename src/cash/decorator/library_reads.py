"""The settings of a library a cached body read as it ran, checked on lookup.

A library keeps settings outside any argument: ``sc.settings.n_jobs`` (a
property of the settings class's metaclass returning ``cls._n_jobs``),
``sklearn.set_config(transform_output="pandas")`` (a dict in a thread-local),
``pd.set_option("display.precision", 2)`` (a module's dict), NumPy's print
options (a ``ContextVar``), a ``ContextVar`` of the user's. The key reads the
user's code, and a library only by its installed version, so a call made
after a setting changed was served the result made under the old one.

What a body reads of them is found as it runs: the code it ran, the user's
and the libraries' (`code_trace` records both: library code is armed for
it on first sight), each read by the global names its bytecode loads and the
attribute chains it follows from them (``settings.n_jobs``,
``_threadlocal.global_config``, ``VAR.get``). Each chain is followed again
after the body, and what it ends on is kept with the entry (`build`):

``("v", module, "a.b", fingerprint)``
    plain data: an atom, an enum member, or lists, tuples, sets and dicts of
    them (a library's ``_global_config``);
``("c", module, "VAR", fingerprint)``
    a ``ContextVar``: the value it holds in the context the body ran in;
``("s", name, "", fingerprint)``
    state kept in C that no chain reaches: NumPy's error handling
    (``np.errstate``, ``np.seterr``) for a call that reached NumPy,
    ``decimal``'s context for one that reached ``decimal``.

A lookup follows each chain again and serves the entry only while every one
ends on the same fingerprint (`changed`). A chain ending on anything else (a
function, a class, a cache of objects) is not kept: code is keyed by the
library's version, and objects are not settings. Nor is what a method of an
object reads (``options.get("x")``): an object's ``__dict__`` holds the
caches it fills lazily too (numba's types cache their ``str``), and a new
process, which has not filled them yet, would never be served. A ``ContextVar`` holding a
value that is not plain data cannot be checked: the call is not stored
(`build` says why), and runs with a warning. The standard library's own
module state is not followed, except ``decimal``'s context.

Names the code that ran binds itself (``global _cache; _cache = ...``) are
left out: what the call set, a hit would find set the same.
"""

from __future__ import annotations

import copy
import dis
import enum
import hashlib
import importlib
import inspect
import sys
import threading
import types
import weakref
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any

__all__ = ["build", "changed", "counts_file", "note_served"]

_MISSING = object()
_ATOMS = (type(None), bool, int, float, complex, str, bytes)
#: How many items one fingerprint looks through before giving up.
_BUDGET = 20000
#: How many items one call keeps at most: more, and the call is not stored.
_MAX_ITEMS = 2000

_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"builtins", "__future__"}
#: Top-level packages whose state lives in C, read by `_SPECIALS`.
_SPECIAL_TOPS = frozenset({"numpy", "decimal", "_decimal", "_pydecimal"})

_lock = threading.Lock()
#: code -> ``(chains, globals it binds)`` of its bytecode.
_CHAINS: weakref.WeakKeyDictionary[types.CodeType, tuple[tuple[tuple[str, ...], ...], frozenset[str]]] = (
    weakref.WeakKeyDictionary()
)


class CodeSet:
    """Code objects, by identity. Two code objects compare equal when their
    bytecode, names and constants do, whatever file they come from: a
    module imported again from another folder (or a function of the user's
    written like a library's) is other code, and a ``WeakSet`` took it for
    the code it had."""

    __slots__ = ("_refs",)

    def __init__(self) -> None:
        self._refs: dict[int, Any] = {}

    def __contains__(self, code: Any) -> bool:
        ref = self._refs.get(id(code))
        return ref is not None and ref() is code

    def __len__(self) -> int:
        return len(self._refs)

    def add(self, code: types.CodeType) -> None:
        key = id(code)
        refs = self._refs

        def gone(ref: Any, key: int = key) -> None:
            if refs.get(key) is ref:
                del refs[key]

        refs[key] = weakref.ref(code, gone)

    def discard(self, code: types.CodeType) -> None:
        if code in self:
            del self._refs[id(code)]


#: A library's code that reads no setting: every chain it follows ends on
#: code or an object. The recorder stops arming it (`code_trace`), which
#: is most of what watching a library costs a miss.
QUIET = CodeSet()
#: filename -> whether code there is a library's (`counts_file`).
_FILES: dict[str, bool] = {}


# -- what code counts -----------------------------------------------------------


def counts_file(filename: str) -> bool:
    """Is code in *filename* a library's: installed, not the standard
    library's, not cash's, not the user's?"""
    known = _FILES.get(filename)
    if known is not None:
        return known
    from ..install_paths import interpreter_roots, is_cash_path, is_installed_path, normcase_path

    verdict = False
    if filename and not filename.startswith("<"):
        try:
            path = normcase_path(filename)
            verdict = (
                is_installed_path(filename)
                and not is_cash_path(filename)
                and not (path.startswith(interpreter_roots()) and "site-packages" not in path)
            )
        except Exception:  # noqa: BLE001 - a path that cannot be judged is not a library's
            verdict = False
    if len(_FILES) > 8192:
        _FILES.clear()
    _FILES[filename] = verdict
    return verdict


def _top(name: Any) -> str:
    return name.partition(".")[0] if isinstance(name, str) else ""


def _owner(v: Any) -> str:
    """The top-level package that owns *v*: a module's, a class's or
    function's ``__module__``, else its type's."""
    if isinstance(v, types.ModuleType):
        return _top(v.__name__)
    if isinstance(v, (type, types.FunctionType, types.BuiltinFunctionType)):
        return _top(getattr(v, "__module__", None))
    return _top(type(v).__module__)


def _library_owner(top: str) -> bool:
    if not top or top in _STDLIB or top == "cash":
        return False
    mod = sys.modules.get(top)
    if mod is None:
        return False
    from ..install_paths import is_user_module

    try:
        return not is_user_module(mod)
    except Exception:  # noqa: BLE001 - a module that cannot be judged is not followed
        return False


# -- the chains a code reads ----------------------------------------------------


_OP = dis.opmap
_LOAD_GLOBAL, _LOAD_NAME, _LOAD_ATTR = _OP["LOAD_GLOBAL"], _OP["LOAD_NAME"], _OP["LOAD_ATTR"]
_STORES = frozenset({_OP["STORE_GLOBAL"], _OP["DELETE_GLOBAL"]})
_ATTR_STORES = frozenset({_OP["STORE_ATTR"], _OP["DELETE_ATTR"]})
_EXTENDED_ARG, _CACHE = _OP["EXTENDED_ARG"], _OP.get("CACHE", -1)
_NOP = _OP.get("NOP", -1)
_LOAD_DEREF = _OP.get("LOAD_DEREF", -1)
#: A chain from a closure's variable, not a global, starts with this mark.
DEREF = "@"
#: LOAD_GLOBAL and LOAD_ATTR keep a flag in the low bit of their argument
#: (3.11 and 3.12 on): the name index is the rest.
_SHIFT_GLOBAL = 1 if sys.version_info >= (3, 11) else 0
_SHIFT_ATTR = 1 if sys.version_info >= (3, 12) else 0


def _chains(code: types.CodeType) -> tuple[tuple[tuple[str, ...], ...], frozenset[str]]:
    """The attribute chains *code* follows from a global name, and the
    globals it binds (and, as ``".name"``, the attributes it sets): read
    off its bytecode, once."""
    held = _CHAINS.get(code)
    if held is not None:
        return held
    names = code.co_names
    raw = code.co_code
    chains: set[tuple[str, ...]] = set()
    stores: set[str] = set()
    cur: list[str] | None = None
    ext = 0
    for i in range(0, len(raw), 2):
        op = raw[i]
        arg = raw[i + 1] | ext
        if op == _EXTENDED_ARG:
            ext = arg << 8
            continue
        ext = 0
        if op == _CACHE or op == _NOP:
            continue
        if op == _LOAD_GLOBAL or op == _LOAD_NAME:
            if cur:
                chains.add(tuple(cur))
            cur = [names[arg >> _SHIFT_GLOBAL if op == _LOAD_GLOBAL else arg]]
            continue
        if op == _LOAD_DEREF:
            if cur:
                chains.add(tuple(cur))
            cur = None
            try:
                name = code._varname_from_oparg(arg)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - a Python without it: closures are not followed
                continue
            if name in code.co_freevars:
                cur = [DEREF + name]
            continue
        if op == _LOAD_ATTR and cur is not None:
            cur.append(names[arg >> _SHIFT_ATTR])
            continue
        if cur:
            chains.add(tuple(cur))
        cur = None
        if op in _STORES:
            stores.add(names[arg])
        elif op in _ATTR_STORES:
            stores.add("." + names[arg])
    if cur:
        chains.add(tuple(cur))
    held = (tuple(sorted(chains)), frozenset(stores))
    with _lock:
        _CHAINS[code] = held
    return held


# -- fingerprints ---------------------------------------------------------------


class _TooBig(Exception):
    pass


def _canon(v: Any, out: list[str], budget: list[int]) -> bool:
    """Append *v*'s canonical text to *out*; False when it is not plain data."""
    budget[0] -= 1
    if budget[0] < 0:
        raise _TooBig
    t = type(v)
    if t in _ATOMS:
        out.append(f"{t.__name__}:{v!r}")
        return True
    if isinstance(v, enum.Enum):
        out.append(f"E:{t.__module__}.{t.__qualname__}.{v.name}")
        return True
    if t is tuple or t is list or (isinstance(v, tuple) and hasattr(t, "_fields")):
        out.append(f"{t.__name__}(")
        for x in v:
            if not _canon(x, out, budget):
                return False
        out.append(")")
        return True
    if t is set or t is frozenset:
        parts = []
        for x in v:
            one: list[str] = []
            if not _canon(x, one, budget):
                return False
            parts.append("".join(one))
        out.append(f"{t.__name__}{{{','.join(sorted(parts))}}}")
        return True
    if t is dict:
        parts = []
        for k, x in list(v.items()):
            one = []
            if not (_canon(k, one, budget) and _canon(x, one, budget)):
                return False
            parts.append("\0".join(one))
        out.append(f"dict{{{','.join(sorted(parts))}}}")
        return True
    return False


def fingerprint(v: Any) -> str | None:
    """*v*'s fingerprint as plain data, or None when it is not."""
    t = type(v)
    if t in _ATOMS:
        text = f"{t.__name__}:{v!r}"
        return text if len(text) <= 96 else "#" + hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
    out: list[str] = []
    try:
        if not _canon(v, out, [_BUDGET]):
            return None
    except (_TooBig, RecursionError):
        return None
    except Exception:  # noqa: BLE001 - a container changed while read: not plain enough
        return None
    text = "".join(out)
    return text if len(text) <= 96 else "#" + hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _plain(v: Any) -> bool:
    t = type(v)
    return t in _ATOMS or t in (tuple, list, set, frozenset, dict) or isinstance(v, (enum.Enum, tuple))


#: ``[extobj, state]`` of the last `_numpy_state`: NumPy 2 keeps the error
#: handling in one object a ContextVar holds, replaced on every change.
_NUMPY_LAST: list[Any] = [_MISSING, ""]


def _numpy_state() -> str:
    np = sys.modules.get("numpy")
    if np is None:
        return "absent"
    config = sys.modules.get("numpy._core._ufunc_config")
    var = getattr(config, "_extobj_contextvar", None)
    if type(var) is not ContextVar:
        return _numpy_compute(np)
    # Never set in this context: the default, the same object each time.
    ext = var.get(None)
    if ext is _NUMPY_LAST[0]:
        return _NUMPY_LAST[1]
    state = _numpy_compute(np)
    _NUMPY_LAST[:] = [ext, state]
    return state


def _numpy_compute(np: Any) -> str:
    try:
        call = np.geterrcall()
        name = None if call is None else f"{getattr(call, '__module__', '?')}.{getattr(call, '__qualname__', '?')}"
        return repr((sorted(np.geterr().items()), np.getbufsize(), name))
    except Exception:  # noqa: BLE001 - a numpy without them
        return "unreadable"


def _decimal_state() -> str:
    mod = sys.modules.get("decimal")
    if mod is None:
        return "absent"
    try:
        ctx = mod.getcontext()
        traps = sorted(k.__name__ for k, on in ctx.traps.items() if on)
        return repr((ctx.prec, ctx.rounding, ctx.Emin, ctx.Emax, ctx.capitals, ctx.clamp, traps))
    except Exception:  # noqa: BLE001 - unreadable context
        return "unreadable"


_SPECIALS = {"numpy": _numpy_state, "decimal": _decimal_state}
_SPECIAL_OF = {"numpy": "numpy", "decimal": "decimal", "_decimal": "decimal", "_pydecimal": "decimal"}
#: The ContextVars whose value is C state a special reads instead.
_SPECIAL_VARS = {"numpy.ufunc.extobj": "numpy"}


# -- following a chain ------------------------------------------------------------

_PLAIN_DESCRIPTORS = (
    property,
    types.MemberDescriptorType,
    types.GetSetDescriptorType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    classmethod,
    staticmethod,
)


def _step(v: Any, name: str) -> Any:
    """``getattr(v, name)``, for an attribute a plain lookup or a property
    gives; `_MISSING` for one only ``__getattr__`` or another descriptor
    would."""
    if isinstance(v, types.ModuleType):
        return v.__dict__.get(name, _MISSING)
    found = inspect.getattr_static(v, name, _MISSING)
    if found is _MISSING:
        return _MISSING
    if hasattr(type(found), "__get__") and not isinstance(found, _PLAIN_DESCRIPTORS):
        return _MISSING
    try:
        return getattr(v, name)
    except Exception:  # noqa: BLE001 - an attribute that raises is not read
        return _MISSING


def deref_roots(code: types.CodeType) -> tuple[str, ...]:
    """The closure variables of *code* a chain starts from."""
    return tuple(sorted({chain[0][1:] for chain in _chains(code)[0] if chain[0].startswith(DEREF)}))


def _follow(g: Mapping[str, Any], chain: tuple[str, ...], user: bool) -> tuple[str, tuple[str, ...], Any] | None:
    """Where *chain* ends in *g*: ``(kind, the chain up to there, value)``,
    or None when it ends on nothing kept (see the module doc)."""
    v = g.get(chain[0], _MISSING)
    if v is _MISSING:
        return None
    if type(v) is not ContextVar and (user or not _plain(v)):
        top = _owner(v)
        if user or top in _STDLIB:
            if not _library_owner(top) and top not in _SPECIAL_TOPS:
                return None
    i = 1
    while True:
        if type(v) is ContextVar:
            return "c", chain[:i], v
        if _plain(v) and not isinstance(v, types.ModuleType):
            return ("v", chain[:i], v) if (i > 1 or not user) else None
        if i >= len(chain):
            return None
        nv = _step(v, chain[i])
        if nv is _MISSING:
            return None
        v = nv
        i += 1


def _resolve(module: str, names: tuple[str, ...]) -> Any:
    """The value at *names* in *module* now, as `_follow` reached it."""
    mod = sys.modules.get(module)
    if mod is None:
        mod = importlib.import_module(module)
    v = mod.__dict__[names[0]]
    for name in names[1:]:
        v = v.__dict__[name] if isinstance(v, types.ModuleType) else getattr(v, name)
    return v


#: The types whose value cannot change while the same object: a chain that
#: ends on the very object it ended on last time ends on the same value.
_FIXED = frozenset({type(None), bool, int, float, complex, str, bytes})


class _Check:
    """One item, ready to be checked again: its chain split, and the object
    it ended on last (when that object cannot change in place)."""

    __slots__ = ("dotted", "fp", "kind", "last", "module", "names", "snap")

    def __init__(self, item: Any) -> None:
        self.kind, self.module, self.dotted, self.fp = item
        self.names = tuple(self.dotted.split(".")) if self.dotted else ()
        self.last: Any = _MISSING
        #: A copy of the container it ended on last, which matched *fp*.
        self.snap: Any = _MISSING

    def holds(self) -> bool:
        kind = self.kind
        if kind == "s":
            return _SPECIALS[self.module]() == self.fp
        v = _resolve(self.module, self.names)
        if kind == "c":
            v = v.get(_MISSING)
            if v is _MISSING:
                v = "unset"
        if v is self.last:
            return True
        snap = self.snap
        if snap is not _MISSING and type(v) is type(snap):
            # Compared in C: a container equal to the copy taken when it
            # last matched holds the same data. (Equal numbers of another
            # type, ``1`` for ``True``, count as the same.)
            try:
                if v == snap:
                    return True
            except Exception:  # noqa: BLE001 - not comparable: fingerprinted below
                pass
        if fingerprint(v) != self.fp:
            return False
        if type(v) in _FIXED or isinstance(v, enum.Enum):
            self.last = v
        elif type(v) in (dict, list, tuple, set, frozenset):
            try:
                self.snap = copy.deepcopy(v)
            except Exception:  # noqa: BLE001 - not copied: fingerprinted each time
                self.snap = _MISSING
        return True

    def text(self) -> str:
        if self.kind == "s":
            return f"{self.module}'s {'error handling' if self.module == 'numpy' else 'context'}"
        return f"{self.module}.{self.dotted}"


#: ``id(items)`` -> ``(items, their checks)``: an entry's list, checked on
#: every hit, is split once.
_COMPILED: dict[int, tuple[Any, tuple[_Check, ...]]] = {}


def _checks(items: Any) -> tuple[_Check, ...]:
    held = _COMPILED.get(id(items))
    if held is not None and held[0] is items:
        return held[1]
    checks = tuple(_Check(item) for item in items)
    with _lock:
        if len(_COMPILED) > 4096:
            _COMPILED.clear()
        _COMPILED[id(items)] = (items, checks)
    return checks


# -- recording --------------------------------------------------------------------

#: code -> ``(globals, user, roots, kept chains, specials)``: what `_plan`
#: found, standing while every root name holds the object it held.
_PLANS: weakref.WeakKeyDictionary[types.CodeType, tuple] = weakref.WeakKeyDictionary()


def _plan(
    code: types.CodeType, g: Mapping[str, Any], user: bool, *, memo: bool = True
) -> tuple[tuple[tuple[str, ...], ...], frozenset[str]]:
    """The chains of *code* that end on something kept, and the specials
    its reads reach. A chain that ends on code or an object is passed
    over until a name it starts from is bound to another object."""
    held = _PLANS.get(code)
    if held is not None and held[0] is g and held[1] == user:
        get = g.get
        if all(get(name, _MISSING) is value for name, value in held[2]):
            return held[3], held[4]
    chains, _bound = _chains(code)
    kept: list[tuple[str, ...]] = []
    specials: set[str] = set()
    roots: dict[str, Any] = {}
    for chain in chains:
        if chain[0].startswith("__"):
            continue  # a module's own __name__, __file__, __spec__
        root = g.get(chain[0], _MISSING)
        roots[chain[0]] = root
        if root is _MISSING:
            continue
        if _SPECIAL_OF.get(_owner(root)):
            specials.add(_SPECIAL_OF[_owner(root)])
        try:
            hit = _follow(g, chain, user)
        except Exception:  # noqa: BLE001 - a chain that cannot be followed now is followed next time
            kept.append(chain)
            continue
        if hit is not None:
            kept.append(chain)
    plan = (tuple(kept), frozenset(specials))
    if not memo:
        return plan
    with _lock:
        _PLANS[code] = (g, user, tuple(roots.items()), *plan)
        if kept or specials:
            QUIET.discard(code)
        elif not user:
            QUIET.add(code)
    return plan


#: module name -> whether it is a compiled extension (`_compiled`).
_COMPILED_MODULES: dict[str, bool] = {}


def _compiled(name: str) -> bool:
    known = _COMPILED_MODULES.get(name)
    if known is None:
        file = getattr(sys.modules.get(name), "__file__", None)
        known = isinstance(file, str) and not file.endswith((".py", ".pyc", ".pyw"))
        _COMPILED_MODULES[name] = known
    return known


def _rooted(g: Mapping[str, Any], module: str, upto: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """*upto* in *g* (of *module*) as read from the last module it passes
    through: ``np.nan`` read in ten modules is ``numpy``'s ``nan``, checked
    once, and a library module a cell or an ``exec`` namespace holds is
    found again by its name."""
    v: Any = g
    at = (module, upto)
    for j, name in enumerate(upto[:-1]):
        v = v.get(name) if j == 0 else v.__dict__.get(name)
        if not isinstance(v, types.ModuleType):
            break
        mod_name = getattr(v, "__name__", None)
        if isinstance(mod_name, str) and sys.modules.get(mod_name) is v:
            at = (mod_name, upto[j + 1 :])
    return at


class _WithCells(dict):
    """A user's code's globals, and the library modules its closure holds
    under `DEREF` names: a chain from either is followed alike."""

    __slots__ = ("g",)

    def __init__(self, g: Mapping[str, Any], cells: Mapping[str, Any]) -> None:
        super().__init__({DEREF + k: v for k, v in cells.items()})
        self.g = g

    def get(self, name: str, default: Any = None) -> Any:
        if name.startswith(DEREF):
            return dict.get(self, name, default)
        return self.g.get(name, default)

    def __getitem__(self, name: str) -> Any:
        return dict.__getitem__(self, name) if name.startswith(DEREF) else self.g[name]


def build(
    codes: Mapping[types.CodeType, tuple[Mapping[str, Any], bool]],
    served: Iterable[tuple] = (),
    cells: Mapping[types.CodeType, Mapping[str, Any]] | None = None,
) -> tuple[list[tuple] | None, str | None]:
    """The items the code a body ran reads (*codes*: code -> ``(its
    globals, whether it is the user's)``), with those of the entries
    served inside it (*served*); and why the call cannot be stored, if so."""
    items: set[tuple] = set(served)
    specials: set[str] = set()
    stores: dict[int, set[str]] = {}
    plans = []
    for code, (g, user) in codes.items():
        if code.co_name == "<module>" or not isinstance(g.get("__name__"), str):
            continue
        bound = _chains(code)[1]
        if bound:
            stores.setdefault(id(g), set()).update(bound)
        held = cells.get(code) if cells else None
        if held:
            g = _WithCells(g, held)
            kept, found = _plan(code, g, user, memo=False)
        else:
            kept, found = _plan(code, g, user)
        specials |= found
        if kept:
            plans.append((g, user, kept))
    seen: set[tuple[int, tuple[str, ...]]] = set()
    places: set[tuple] = set()
    for g, user, kept in plans:
        module = g["__name__"]
        skip = stores.get(id(g), ())
        for chain in kept:
            if chain[0] in skip or (id(g), chain) in seen:
                continue
            seen.add((id(g), chain))
            try:
                hit = _follow(g, chain, user)
            except Exception:  # noqa: BLE001 - a chain that cannot be followed is not kept
                continue
            if hit is None:
                continue
            kind, upto, v = hit
            if len(upto) > 1 and "." + upto[-1] in skip:
                # An attribute the code that ran sets: a counter of its own
                # (numba's ``ArrayAnalysis.aa_count += 1``), not a setting.
                continue
            where, upto = _rooted(g, module, upto)
            if _compiled(where):
                # What a compiled module holds is set by its own code as it
                # loads: numba's table of C helper addresses, pyarrow's type
                # ids. The same in a process, other in the next, and no
                # setting a program changes.
                continue
            # A container bound to several names (pandas' _global_config)
            # is one object, checked once.
            same = ("obj", id(v)) if kind == "v" and type(v) in (dict, list, set) else (where, upto)
            if same in places:
                continue
            places.add(same)
            dotted = ".".join(upto)
            if kind == "c":
                special = _SPECIAL_VARS.get(v.name)
                if special:
                    specials.add(special)
                    continue
                value = v.get(_MISSING)
                fp = fingerprint("unset" if value is _MISSING else value)
                if fp is None:
                    return None, (
                        f"the context variable {v.name!r} ({where}.{dotted}) it read holds a "
                        f"{type(value).__qualname__}, which cash cannot compare between calls"
                    )
            else:
                fp = fingerprint(v)
            if fp is not None:
                items.add((kind, where, dotted, fp))
    for name in specials:
        items.add(("s", name, "", _SPECIALS[name]()))
    if len(items) > _MAX_ITEMS:
        return None, f"it read more than {_MAX_ITEMS} library settings, too many to check on every lookup"
    return sorted(items), None


def changed(items: Any) -> str | None:
    """The first item of *items* (`build`) whose value moved, as text for a
    miss reason; None when every one holds."""
    for check in _checks(items):
        try:
            same = check.holds()
        except Exception:  # noqa: BLE001 - a chain that no longer resolves moved
            same = False
        if not same:
            return check.text()
    return None


def note_served(items: Iterable[Any]) -> None:
    """An entry was served inside the bodies being recorded: what its run
    read, they read too."""
    from . import code_trace

    code_trace.note_library_items(list(map(tuple, items)))
