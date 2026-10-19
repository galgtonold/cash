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
one big plain value the watchers already prove unchanged) is served from
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
import datetime
import decimal
import functools
import operator
import os
import re
import sys
import textwrap
import threading
import time
import types
import weakref
from typing import TYPE_CHECKING, Any

from .. import _native
from .._clock import perf_counter as _perf_counter
from ..analysis.helper_code import own_code_is_user
from ..source_reading import own_source
from ..tracking.tracker_context import active_tracker
from .call_state import NESTED_CASH_SECONDS
from .user_code import is_cash_wrapper

if TYPE_CHECKING:
    from .cached_function import CachedFunction

__all__ = ["ENABLED", "VERIFY", "HitMemo", "HitMemoMismatch", "bump", "flush_all", "status"]

_cw = _native.cwatch
if _cw is not None and not hasattr(_cw, "fast_lookup"):
    _cw = None


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


#: Is the fast path on? Needs the extension; ``CASH_HITPATH=0`` turns it off.
ENABLED = _cw is not None and _flag("CASH_HITPATH", "1")
#: ``CASH_HITPATH_VERIFY=1``: every fast hit builds the key again and compares.
VERIFY = ENABLED and _flag("CASH_HITPATH_VERIFY", "0")

_fast_lookup = _cw.fast_lookup if ENABLED else None
_world_epoch = _cw.world_epoch if ENABLED else None
_cash_epoch = _cw.cash_epoch if ENABLED else None


class HitMemoMismatch(AssertionError):
    """``CASH_HITPATH_VERIFY=1``: the fast path served a key the ordinary
    key build no longer gives."""


def bump() -> None:
    """Something cash keeps changed in a way any key may read (a clear, a
    registration, a setting): no remembered hit is served until it is met
    again through the ordinary path."""
    if _cw is not None:
        _cw.bump_cash()


#: A function's state -> the memos whose key reads it (callers that reached
#: it as a cached callee): its miss drops their remembered hits too.
_DEPENDENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


#: A cached function's wrapper -> its state, for the scan of a caller.
_STATE_OF: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def own_state(wrapper: Any, state: list | None) -> None:
    """Note that *wrapper*'s remembered hits live in *state*."""
    if state is not None:
        _STATE_OF[wrapper] = state


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


def epochs(spec: CachedFunction) -> tuple | None:
    """``(world, cash, entries)`` now for *spec*, or None when the path is
    off: an ordinary hit is remembered only if none of them moved while its
    key was built."""
    if _world_epoch is None:
        return None
    state = spec.hit_state
    return _world_epoch(), _cash_epoch(), state[S_ENTRIES] if state is not None else None


#: Every memo with fast hits not yet handed to the net-loss accounting.
_MEMOS: weakref.WeakSet[HitMemo] = weakref.WeakSet()


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
    module = getattr(owner, "__module__", None)
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


def _signature_names(source: str) -> set[str]:
    """The names the annotations in *source*'s ``def`` line read (a key may
    resolve them); every identifier of *source* when it does not parse."""
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


class _Scan:
    """What one cached function's key can read, made watched (see module doc)."""

    def __init__(self, root_module: str | None) -> None:
        self.root_module = root_module
        self.seen: set = set()
        self.keep: list = []
        self.cells: list = []
        self.callees: list = []
        self.types: dict[int, tuple[type, int]] = {}
        self.names: set[str] = set()
        self.dicts: list[tuple[dict, bool]] = []
        self.count = 0

    def _first(self, key: Any, obj: Any) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        self.keep.append(obj)
        self.count += 1
        if self.count > _SCAN_LIMIT:
            raise Ineligible("too much to watch")
        return True

    def watch(self, d: dict, unkeyed: bool) -> None:
        if type(d) is not dict:
            raise Ineligible("a namespace that is not a plain dict")
        self.dicts.append((d, unkeyed))

    def value(self, v: Any, names: frozenset) -> None:
        t = type(v)
        if t in _ATOMS:
            return
        if t is types.ModuleType:
            name = getattr(v, "__name__", "")
            if name == "cash" or (isinstance(name, str) and name.startswith("cash.")):
                return
            if self._first(("module", id(v), id(names)), v):
                self.module(v, names)
            return
        if not self._first(id(v), v):
            return
        if _is_cash(v, t):
            return  # cash's own objects: what of them a key reads moves the cash epoch
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
            if owner is not None and not isinstance(owner, types.ModuleType):
                self.value(owner, names)
        elif t is types.MethodType:
            self.value(v.__func__, names)
            self.value(v.__self__, names)
        elif t is types.MethodWrapperType:
            self.value(v.__self__, names)
        elif t is staticmethod or t is classmethod:
            self.value(v.__func__, names)
        elif t is property:
            for part in (v.fget, v.fset, v.fdel):
                if part is not None:
                    self.value(part, names)
        elif t is functools.partial:
            if getattr(v, "__dict__", None):
                raise Ineligible("a partial with attributes")
            self.value(v.func, names)
            self.value(v.args, names)
            self.watch(v.keywords, True)
            for item in v.keywords.values():
                self.value(item, names)
        elif t in _FIXED_C or (t.__name__ == "ufunc" and t.__module__ == "numpy"):
            return
        else:
            raise Ineligible(f"a {t.__module__}.{t.__qualname__}")

    def module(self, mod: types.ModuleType, names: frozenset) -> None:
        md = getattr(mod, "__dict__", None)
        if type(md) is not dict:
            raise Ineligible("a module without a plain namespace")
        self.watch(md, False)
        for name in names:
            if name in md:
                self.value(md[name], names)

    def function(self, f: types.FunctionType) -> None:
        if is_cash_wrapper(f):
            # Another cached function, or this one called by name: its own
            # state is cash's (a registration, a setting) or its code.
            callee = _STATE_OF.get(getattr(f, "__func__", f))
            if callee is not None:
                self.callees.append(callee)
            inner = getattr(f, "__wrapped__", None)
            if inner is not None:
                self.value(inner, frozenset())
            return
        if not own_code_is_user(f, self.root_module):
            # Library code: keyed by its name, which the module watch covers.
            return
        code = f.__code__
        own: set[str] = set()
        _code_names(code, own)
        decorated: set[str] = set()
        try:
            source = own_source(f)
        except Exception:  # noqa: BLE001 - no source: the bytecode names it reads
            source = ""
        if source:
            for line in source.splitlines():
                if _DEF_LINE.match(line):
                    break
                decorated.update(_IDENT.findall(line))
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
                self.value(g[name], names)
            elif type(b) is dict and name in b:
                self.value(b[name], names)
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

    def klass(self, cls: type) -> None:
        if not (cls.__flags__ & _HEAPTYPE):
            return  # a builtin class: nothing in it changes
        if type(cls) not in _METACLASSES:
            raise Ineligible("a class with a metaclass of its own")
        for k in cls.__mro__:
            if not (k.__flags__ & _HEAPTYPE) or id(k) in self.types:
                continue
            tag = _cw.world_watch_type(k)
            if tag is None:
                raise Ineligible("a class without a version tag")
            self.types[id(k)] = (k, tag)
            for name, attr in list(vars(k).items()):
                if _is_dunder(name) or name.startswith("_abc_"):
                    # Machinery, never data the key reads; but code it runs.
                    if isinstance(attr, (types.FunctionType, staticmethod, classmethod, property)):
                        self.value(attr, frozenset())
                    continue
                self.value(attr, frozenset())

    def commit(self) -> None:
        """Watch what was found (after the whole scan passed)."""
        for d, unkeyed in self.dicts:
            _cw.world_watch_dict(d, unkeyed)
        _cw.world_add_names(self.names)


#: ``id(module dict)`` of every module namespace watched so far.
_WATCHED_MODULES: set[int] = set()
_WATCH_LOCK = threading.Lock()


def _watch_modules() -> None:
    """Watch every module's namespace (keyed) and ``sys.modules`` (every
    change): a helper cash finds by name can live in any of them."""
    with _WATCH_LOCK:
        modules = sys.modules
        _cw.world_watch_dict(modules, True)
        for mod in list(modules.values()):
            md = getattr(mod, "__dict__", None)
            if type(md) is dict and id(md) not in _WATCHED_MODULES:
                _cw.world_watch_dict(md, False)
                _WATCHED_MODULES.add(id(md))
        _cw.world_watch_dict(builtins.__dict__, False)


def _resolve(mod_name: str, chain: tuple) -> Any:
    obj: Any = sys.modules.get(mod_name)
    for attr in chain:
        if obj is None:
            return None
        obj = getattr(obj, attr, None)
    return obj


def scan(spec: CachedFunction, registry: Any) -> tuple[tuple, tuple, list]:
    """Watch the world of *spec*'s key. Returns the cells and the class
    tags a fast hit compares, and the states of the cached functions it
    reaches; raises `Ineligible`."""
    func, name = spec.func, spec.name
    s = _Scan(getattr(func, "__module__", None))
    s.value(func, frozenset())
    graph = registry.graph
    stack, visited = [name], {name}
    while stack:
        node = stack.pop()
        if node in registry.data_sources:
            raise Ineligible("a data source")
        report = registry.report_for(func, name) if node == name else registry.purity_reports.get(node)
        if report is not None:
            if report.unwalkable or report.unkeyable or report.environment_reads:
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
    _watch_modules()
    s.commit()
    cells = tuple(x for pair in s.cells for x in pair)
    tags = tuple(x for pair in s.types.values() for x in pair)
    return cells, tags, s.callees


def scan_trace(spec: CachedFunction, registry: Any, items: list) -> tuple[tuple, tuple, list]:
    """Watch what a stored trace (`code_trace`) names, as `scan` watches the
    key's world: while none of it moves, the trace still holds, and a fast
    hit needs no check of its own. Raises `Ineligible`."""
    s = _Scan(getattr(spec.func, "__module__", None))
    for item in items:
        kind = item[0]
        if kind in ("code", "class", "global", "held", "attr"):
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
        elif kind == "cached":
            cf = registry.cached.get(item[1])
            if cf is None or cf.hit_state is None:
                raise Ineligible("a cached function served inside without a fast path")
            s.callees.append(cf.hit_state)
        else:
            # A lookup by name or an import by name: checked on each hit.
            raise Ineligible(f"a traced {kind}")
    _watch_modules()
    s.commit()
    cells = tuple(x for pair in s.cells for x in pair)
    tags = tuple(x for pair in s.types.values() for x in pair)
    return cells, tags, s.callees


# -- the memo --------------------------------------------------------------

#: What `_cwatch.fast_call` hands back for "take the ordinary path".
MISS = object()

# The slots of a memo's state list, as `_cwatch.fast_call` reads them.
S_WORLD, S_ENTRIES, S_CELLS, S_TYPES, S_SERVE, S_ACC, S_SPEC, S_LOG, S_SPARE, S_CLEAR_AT, S_FLAGS = range(11)
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

    __slots__ = ("metadata", "deadline", "files", "ttl", "ram", "backend", "copy", "value", "md")


class HitMemo:
    """One cached function's fast hits (see the module doc)."""

    #: Remembered calls per function at most.
    MAX_ENTRIES = 4096

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
        self.served_since_scan = 0
        _MEMOS.add(self)

    # -- served ----------------------------------------------------------

    def serve(self, entry: tuple, args: tuple, kwargs: dict, t0: float) -> Any:
        """A fast hit `_cwatch.fast_call` found but leaves to Python: a ttl,
        files to check, a copy to make, the clear check due, verify mode, or
        accounting to hand over. Returns the result, or `MISS`."""
        state = self.state
        self.flush()
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

    def _verify(self, entry: tuple, args: tuple, kwargs: dict) -> None:
        from .._active import ACTIVE_CONFIG, EXPLAINING

        keys = self.runner._keys
        token = EXPLAINING.set(True)
        config_token = ACTIVE_CONFIG.set(self.runner._calls._config)
        try:
            built = keys.build(self.spec, args, kwargs)
        except Exception as e:
            raise HitMemoMismatch(f"fast hit of {self.spec.name}: the ordinary key build raised {e!r}") from e
        finally:
            ACTIVE_CONFIG.reset(config_token)
            EXPLAINING.reset(token)
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
        world_now = _world_epoch()
        state = self.state
        if world_now != before[0] or _cash_epoch() != before[1] or state[S_ENTRIES] is not before[2]:
            return  # something moved while the key was built
        if state[S_WORLD] != world_now:
            if self.failed_epoch == world_now:
                return
            self.slow_since_scan += 1
            if self.slow_since_scan < self.backoff:
                return
            self.backoff = 1 if self.served_since_scan else min(self.backoff * 2, 1024)
            self.slow_since_scan = self.served_since_scan = 0
            state[S_WORLD] = -1
            state[S_ENTRIES] = {}
            try:
                cells, tags, callees = scan(spec, runner._registry)
            except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: the ordinary path
                self.failed_epoch = world_now
                return
            for callee in callees:
                if callee is not state:
                    _DEPENDENTS.setdefault(callee, weakref.WeakSet()).add(self)
            # What changed between the key build and the watch is not
            # proven: this call is not remembered, the next one is.
            state[S_CELLS], state[S_TYPES] = cells, tags
            state[S_WORLD] = world_now
            self.traces = {}
            return
        trace = call.metadata.code_trace if call.metadata is not None else None
        if trace and not self._watch_trace(trace, world_now):
            return
        entry = self._entry(call, hit)
        if entry is None:
            return
        key = _cw.any_key(args, kwargs)
        if key is None:
            return
        entries = state[S_ENTRIES]
        if len(entries) >= self.MAX_ENTRIES:
            entries.clear()
        entries[key] = (before[1], *entry)

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
            cells, tags, callees = scan_trace(self.spec, self.runner._registry, trace)
        except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: the ordinary path
            self.traces[key] = -1
            return False
        for callee in callees:
            if callee is not state:
                _DEPENDENTS.setdefault(callee, weakref.WeakSet()).add(self)
        state[S_CELLS] = state[S_CELLS] + cells
        state[S_TYPES] = state[S_TYPES] + tags
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
        ):
            return None
        ledger = call.ledger or {}
        chain = ledger.get("@chain") or []
        # The stages after the globals: rng epoch, environment.
        if len(chain) < 7 or chain[5] != chain[4] or chain[6] != chain[5]:
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
        in_c = direct and h.deadline is None and not h.files
        saved = (metadata.saves_seconds if metadata.saves_seconds is not None else metadata.execution_time) or 0.0
        body = metadata.body_seconds
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
#: and their captures and defaults.
_LEDGER_KINDS = frozenset({"calls", "helper", "globals", "captures and defaults of cached function"})

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


def memo_for(spec: CachedFunction, runner: Any) -> HitMemo | None:
    """*spec*'s memo, made on first use; None when the path is off or the
    function cannot take one (async, ``key=``, declared files, ...)."""
    memo = spec.hit_memo
    if memo is not None:
        return memo
    if not ENABLED or spec.hit_state is None or not _plain_spec(spec):
        return None
    memo = HitMemo(spec, runner)
    spec.hit_memo = memo
    return memo


def _plain_spec(spec: CachedFunction) -> bool:
    import inspect

    return not (
        inspect.iscoroutinefunction(spec.func)
        or spec.arg_key is not None
        or spec.dynamic_depends_on
        or spec.declared_files
        or spec.seed_params
        or spec.rng_modules
    )
