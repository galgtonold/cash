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
``("k", module, "a.b#[keys]", fingerprint)``
    one entry of a dict of settings: a chain only ever subscripted by
    constant strings (``config["display"]["precision"]``), or the dict a
    `KEYED_READERS` function reads by the key it is handed (pandas'
    ``get_option``). Only the entries read are checked: pandas' registry
    holds defaults a script and a notebook kernel set apart, and grows as
    modules register options. A dict read any other way is checked whole;
``("a", module, "Cls.name", fingerprint)``
    an attribute of a library's class a method that ran reads off ``self``
    or ``cls`` (`self_names`), by the class it ran on (`_class_items`): an
    atom, an enum member or a tuple of them, which a program sets on the
    class (``Model.scale = 5``) as a setting. Checked as a ``"v"`` item; a
    fast hit needs no check while the class is watched (`classes_watched`);
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
import json
import sys
import threading
import types
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any

from .. import _weak

__all__ = ["KEYED_NAMES", "build", "changed", "counts_file", "keyed_argument", "note_served"]

_MISSING = object()
#: The fingerprint of a settings entry that is not there.
ABSENT = "absent"
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
_CHAINS: _weak.WeakKeyDictionary[types.CodeType, tuple] = _weak.WeakKeyDictionary()


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

        refs[key] = _weak.ref(code, gone)

    def discard(self, code: types.CodeType) -> None:
        if code in self:
            del self._refs[id(code)]


#: A library's code that reads no setting: every chain it follows ends on
#: code or an object. The recorder stops arming it (`code_trace`), which
#: is most of what watching a library costs a miss.
QUIET = CodeSet()
#: Whether this process has said KEY-LIBRARY-STATE-UNSEEN: a call stored
#: without its body recorded (below 3.12, ``CASH_TRACE=0``, a coroutine)
#: has no setting checked, said once (`CallRunner._warn_settings_unseen`).
UNSEEN_SAID = False
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
_LOAD_CONST = _OP["LOAD_CONST"]
#: A subscript: its own op before 3.14, a ``BINARY_OP`` from it.
_BINARY_SUBSCR = _OP.get("BINARY_SUBSCR", -1)
_BINARY_OP = _OP.get("BINARY_OP", -1)
_NB_SUBSCR = next((i for i, (name, _sign) in enumerate(getattr(dis, "_nb_ops", ())) if name == "NB_SUBSCR"), -1)
#: A chain from a closure's variable, not a global, starts with this mark.
DEREF = "@"
#: LOAD_GLOBAL and LOAD_ATTR keep a flag in the low bit of their argument
#: (3.11 and 3.12 on): the name index is the rest.
_SHIFT_GLOBAL = 1 if sys.version_info >= (3, 11) else 0
_SHIFT_ATTR = 1 if sys.version_info >= (3, 12) else 0


def _chains(
    code: types.CodeType,
) -> tuple[tuple[tuple[str, ...], ...], frozenset[str], Mapping[tuple[str, ...], frozenset]]:
    """The attribute chains *code* follows from a global name, the globals
    it binds (and, as ``".name"``, the attributes it sets), and the chains
    it only ever subscripts by constant strings -> those key paths
    (``config["display"]["precision"]``): read off its bytecode, once."""
    held = _CHAINS.get(code)
    if held is not None:
        return held
    names = code.co_names
    consts = code.co_consts
    raw = code.co_code
    chains: set[tuple[str, ...]] = set()
    stores: set[str] = set()
    paths: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    whole: set[tuple[str, ...]] = set()
    cur: list[str] | None = None
    #: The constant keys the chain in *cur* was subscripted by so far, and
    #: a constant loaded after them not yet seen subscripting it.
    keys: list[str] | None = None
    pending: Any = _MISSING

    def end() -> None:
        if not cur:
            return
        chain = tuple(cur)
        chains.add(chain)
        if keys:
            # What the subscripts reached, its whole value (a constant
            # loaded after them is an operand of what follows).
            paths.setdefault(chain, set()).add(tuple(keys))
        else:
            whole.add(chain)

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
        if cur is not None:
            if op == _LOAD_ATTR and keys is None:
                cur.append(names[arg >> _SHIFT_ATTR])
                continue
            if op == _LOAD_CONST and pending is _MISSING and type(consts[arg]) is str:
                pending = consts[arg]
                continue
            if pending is not _MISSING and (op == _BINARY_SUBSCR or (op == _BINARY_OP and arg == _NB_SUBSCR)):
                keys = [*(keys or ()), pending]
                pending = _MISSING
                continue
            end()
            cur, keys, pending = None, None, _MISSING
        if op == _LOAD_GLOBAL or op == _LOAD_NAME:
            cur = [names[arg >> _SHIFT_GLOBAL if op == _LOAD_GLOBAL else arg]]
            continue
        if op == _LOAD_DEREF:
            try:
                name = code._varname_from_oparg(arg)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - a Python without it: closures are not followed
                continue
            if name in code.co_freevars:
                cur = [DEREF + name]
            continue
        if op in _STORES:
            stores.add(names[arg])
        elif op in _ATTR_STORES:
            stores.add("." + names[arg])
    end()
    keyed = {chain: frozenset(found) for chain, found in paths.items() if chain not in whole}
    held = (tuple(sorted(chains)), frozenset(stores), keyed)
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


#: code -> the attributes it loads off its first argument (`self_names`).
_SELF_NAMES: _weak.WeakKeyDictionary[types.CodeType, tuple[str, ...]] = _weak.WeakKeyDictionary()


def self_names(code: types.CodeType) -> tuple[str, ...]:
    """The attributes *code* loads off its first argument (``self.scale``,
    ``cls.scale``), dunders left out: what a method reads of its class's
    attributes. Empty for code with no argument. Read off its bytecode, once."""
    held = _SELF_NAMES.get(code)
    if held is not None:
        return held
    names: set[str] = set()
    if code.co_argcount or code.co_posonlyargcount:
        first = code.co_varnames[0]
        on_first = False
        try:
            for ins in dis.get_instructions(code):
                if on_first and ins.opname == "LOAD_ATTR" and isinstance(ins.argval, str):
                    if not ins.argval.startswith("__"):
                        names.add(ins.argval)
                if ins.opname.startswith("LOAD_FAST"):
                    val = ins.argval
                    on_first = (val[-1] if isinstance(val, tuple) and val else val) == first
                else:
                    on_first = False
        except Exception:  # noqa: BLE001 - unreadable bytecode: nothing read off it
            names = set()
    held = tuple(sorted(names))
    with _lock:
        _SELF_NAMES[code] = held
    return held


def _frozen_plain(v: Any) -> bool:
    """Is *v* plain data that cannot change in place: an atom, an enum
    member, or a tuple or frozenset of such? A class's list, dict or set is
    most often a cache its methods fill (a registry, a memo): another in the
    next process, never a setting a program assigns."""
    t = type(v)
    if t in _ATOMS or isinstance(v, enum.Enum):
        return True
    if t is tuple or t is frozenset:
        return all(_frozen_plain(x) for x in v)
    return False


def _class_name(cls: Any) -> tuple[str, str] | None:
    """``(module, qualname)`` under which *cls* is found again (`_resolve`),
    or None for a class no name finds."""
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or "<" in qualname:
        return None
    try:
        if _resolve(module, tuple(qualname.split("."))) is not cls:
            return None
    except Exception:  # noqa: BLE001 - not found by its name
        return None
    return module, qualname


def _class_items(receivers: Mapping[types.CodeType, type], stores: set[str]) -> dict[tuple[str, str], str]:
    """The attributes of a library's class the methods that ran read off
    their receiver (``self.scale`` with ``Model.scale = 1`` set on the
    class), plain data that cannot change in place: ``(module, "Cls.name")``
    -> fingerprint. Assigned on the class by the program (``Model.scale =
    5``), one is what a later call reads: a key holding the library by its
    version alone served the old result. The code that read one is watched
    on every traced miss from then on (left out of `QUIET`)."""
    found: dict[tuple[str, str], str] = {}
    for code, cls in receivers.items():
        names = [n for n in self_names(code) if "." + n not in stores]
        if not names or not isinstance(cls, type):
            continue
        mro = getattr(cls, "__mro__", ())
        holder = _MISSING
        read = False
        for name in names:
            for k in mro:
                held = k.__dict__.get(name, _MISSING)
                if held is _MISSING:
                    continue
                if not _frozen_plain(held) or not _library_owner(_owner(k)) or _compiled(k.__module__):
                    break
                if holder is _MISSING:
                    # The receiver's class by its name; a class only the
                    # program holds (a user's subclass) by the library class
                    # the attribute is on.
                    holder = _class_name(cls) if _library_owner(_owner(cls)) else None
                where = holder or _class_name(k)
                if where is None:
                    break
                try:
                    fp = fingerprint(getattr(_resolve(where[0], tuple(where[1].split("."))), name))
                except Exception:  # noqa: BLE001 - not read again by its name: not kept
                    fp = None
                if fp is not None:
                    found[(where[0], f"{where[1]}.{name}")] = fp
                    read = True
                break
        if read:
            with _lock:
                QUIET.discard(code)
    return found


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

    __slots__ = ("dotted", "fp", "kind", "last", "module", "names", "path", "snap")

    def __init__(self, item: Any) -> None:
        self.kind, self.module, self.dotted, self.fp = item
        chain, _sep, keys = self.dotted.partition(_KEYS)
        self.names = tuple(chain.split(".")) if chain else ()
        #: The entry of the dict *names* ends on that the item reads.
        self.path: tuple[str, ...] = tuple(json.loads(keys)) if keys else ()
        self.last: Any = _MISSING
        #: A copy of the container it ended on last, which matched *fp*.
        self.snap: Any = _MISSING

    def holds(self) -> bool:
        kind = self.kind
        if kind == "s":
            return _SPECIALS[self.module]() == self.fp
        v = _resolve(self.module, self.names)
        if self.path:
            v = _entry(v, self.path)
            if v is _MISSING:
                return self.fp == ABSENT
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
        chain = ".".join(self.names)
        return f"{self.module}.{chain}" + "".join(f"[{key!r}]" for key in self.path)


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
_PLANS: _weak.WeakKeyDictionary[types.CodeType, tuple] = _weak.WeakKeyDictionary()


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
    chains = _chains(code)[0]
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


#: Library functions that read one entry of a dict of settings by a key
#: their caller names: ``(module, qualname)`` -> ``(the global the dict is
#: bound to, the argument naming the entry as a dotted path)``, or None as
#: that argument for one that reads no entry's value. The recorder notes
#: the key of each start of them (`code_trace`), so a call is checked on
#: the options it read, not on every option a library registered: pandas'
#: option registry holds defaults a script and a notebook kernel set apart
#: (``display.max_columns``, ``display.encoding``) and grows as modules
#: register theirs.
KEYED_READERS: dict[tuple[str, str], tuple[str, str | None]] = {
    # ``get_option``, ``pd.options.a.b``, ``option_context``, ``set_option``.
    ("pandas._config.config", "_get_root"): ("_global_config", "key"),
    # Adds an option with its default; what it reads (is the name taken?)
    # is no value of a setting.
    ("pandas._config.config", "register_option"): ("_global_config", None),
}
#: The qualnames of `KEYED_READERS`, for the recorder's first look.
KEYED_NAMES = frozenset(qualname for _module, qualname in KEYED_READERS)
#: The separator between an item's chain and the keys (a JSON list) it
#: subscripts by: no name in a chain holds it.
_KEYS = "#"


def keyed_argument(code: types.CodeType, g: Mapping[str, Any]) -> str | None:
    """The argument of *code* (with globals *g*) naming the settings entry
    it reads, when it is one of `KEYED_READERS` that reads one."""
    spec = KEYED_READERS.get((g.get("__name__"), code.co_qualname))
    return spec[1] if spec else None


def _entry(v: Any, path: tuple[str, ...]) -> Any:
    """What ``v[path[0]][path[1]]...`` reads; `_MISSING` where an entry is
    not there."""
    for key in path:
        if type(v) is not dict:
            return _MISSING
        v = v.get(key, _MISSING)
        if v is _MISSING:
            return _MISSING
    return v


def _wants(
    codes: Mapping[types.CodeType, tuple[Mapping[str, Any], bool]],
    keys: Mapping[types.CodeType, set[str | None]],
    cells: Mapping[types.CodeType, Mapping[str, Any]] | None,
) -> tuple[list[tuple[Mapping[str, Any], bool, tuple[str, ...], set | None]], set[str], dict[int, set[str]]]:
    """Per globals and chain the code that ran reads something kept by:
    ``(globals, user, chain, key paths)``, the paths None where a code
    reads the whole of it; the specials reached; what each globals' code
    binds."""
    specials: set[str] = set()
    stores: dict[int, set[str]] = {}
    wants: dict[tuple[int, tuple[str, ...]], list] = {}
    for code, (g, user) in codes.items():
        if code.co_name == "<module>" or not isinstance(g.get("__name__"), str):
            continue
        cs = _chains(code)
        if cs[1]:
            stores.setdefault(id(g), set()).update(cs[1])
        spec = KEYED_READERS.get((g["__name__"], code.co_qualname))
        held = cells.get(code) if cells else None
        if held:
            g = _WithCells(g, held)
            kept, found = _plan(code, g, user, memo=False)
        else:
            kept, found = _plan(code, g, user)
        specials |= found
        for chain in kept:
            paths: set | None
            if spec is not None and chain == (spec[0],):
                if spec[1] is None:
                    continue
                read = keys.get(code)
                ok = bool(read) and all(isinstance(k, str) for k in read)
                paths = {tuple(k.split(".")) for k in read} if ok else None  # type: ignore[union-attr]
            else:
                found_paths = cs[2].get(chain)
                paths = set(found_paths) if found_paths else None
            want = wants.get((id(g), chain))
            if want is None:
                wants[(id(g), chain)] = [g, user, chain, paths]
            elif want[3] is not None:
                if paths is None:
                    want[3] = None
                else:
                    want[3] |= paths
    return [tuple(w) for w in wants.values()], specials, stores  # type: ignore[misc]


def _served_place(item: tuple) -> tuple | None:
    """Where an item an entry served inside the call reads, as `build`
    tells containers apart: the object it ends on, whatever it is called."""
    kind, module, dotted, _fp = item
    if kind not in ("v", "k"):
        return None
    chain, _sep, keys = dotted.partition(_KEYS)
    try:
        v = _resolve(module, tuple(chain.split(".")))
    except Exception:  # noqa: BLE001 - not there now: kept as named
        return None
    if type(v) not in (dict, list, set):
        return None
    return ("obj", id(v), keys) if kind == "k" else ("obj", id(v))


def build(
    codes: Mapping[types.CodeType, tuple[Mapping[str, Any], bool]],
    served: Iterable[tuple] = (),
    cells: Mapping[types.CodeType, Mapping[str, Any]] | None = None,
    keys: Mapping[types.CodeType, set[str | None]] | None = None,
    receivers: Mapping[types.CodeType, type] | None = None,
) -> tuple[list[tuple] | None, str | None]:
    """The items the code a body ran reads (*codes*: code -> ``(its
    globals, whether it is the user's)``), with those of the entries
    served inside it (*served*); and why the call cannot be stored, if so.
    *keys*: a `KEYED_READERS` code -> the keys its starts read.
    *receivers*: a method's code that ran -> the class of what it ran on."""
    items: set[tuple] = set()
    wants, specials, stores = _wants(codes, keys or {}, cells)
    #: Where each item reads -> its fingerprint.
    places: dict[tuple, str | None] = {}
    for g, user, chain, paths in wants:
        module = g["__name__"]
        skip = stores.get(id(g), ())
        if chain[0] in skip:
            continue
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
        dotted = ".".join(upto)
        if kind == "c":
            if (where, upto) in places:
                continue
            places[(where, upto)] = None
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
            items.add((kind, where, dotted, fp))
            continue
        if paths is not None and type(v) is dict and len(upto) == len(chain):
            # Read only by constant keys, or by the keys a reader of it was
            # handed: those entries are what the call read. A container
            # bound to several names (pandas' _global_config) is one
            # object, its entries checked once.
            for path in sorted(paths):
                text = json.dumps(list(path))
                same = ("obj", id(v), text)
                if same in places:
                    continue
                entry = _entry(v, path)
                fp = ABSENT if entry is _MISSING else fingerprint(entry)
                places[same] = fp
                if fp is not None:
                    items.add(("k", where, dotted + _KEYS + text, fp))
            continue
        same = ("obj", id(v)) if type(v) in (dict, list, set) else (where, upto)
        if same in places:
            continue
        fp = fingerprint(v)
        places[same] = fp
        if fp is not None:
            items.add((kind, where, dotted, fp))
    if receivers:
        attr_stores = {name for names in stores.values() for name in names if name.startswith(".")}
        for (where, dotted), fp in _class_items(receivers, attr_stores).items():
            if (where, tuple(dotted.split("."))) not in places:
                places[(where, tuple(dotted.split(".")))] = fp
                items.add(("a", where, dotted, fp))
    for item in served:
        item = tuple(item)
        same = _served_place(item)
        if same is not None:
            if places.get(same, _MISSING) == item[3]:
                # The same object under another name, read the same.
                continue
            places.setdefault(same, item[3])
        items.add(item)
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


def classes_watched(items: Any, watch: Any) -> bool:
    """Are *items* all attributes of classes (``"a"``), each class and its
    bases now watched by *watch* (``world_watch_type``: a change to one
    moves the world epoch, which ends every fast hit)? Then a fast hit
    needs no check of its own."""
    for item in items:
        if item[0] != "a":
            return False
        names = tuple(item[2].split("."))[:-1]
        try:
            cls = _resolve(item[1], names)
        except Exception:  # noqa: BLE001 - not found again: checked on each hit
            return False
        if not isinstance(cls, type):
            return False
        for k in cls.__mro__:
            if k.__flags__ & _HEAPTYPE and watch(k) is None:
                return False
    return True


_HEAPTYPE = 1 << 9


def note_served(items: Iterable[Any]) -> None:
    """An entry was served inside the bodies being recorded: what its run
    read, they read too."""
    from . import code_trace

    code_trace.note_library_items(list(map(tuple, items)))
