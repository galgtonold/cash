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
* the whole text of each file those callables are defined in (its imports,
  its globals, the class bodies);
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

import atexit
import functools
import hashlib
import inspect
import sys
import types
from collections.abc import Callable
from typing import Any

from .. import _weak
from .._paths import is_pseudo_filename
from ..code_digest import callable_identity
from ..effects import EffectKind
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..install_paths import top_package
from ..loaded_code import code_objects_of
from ..purity import is_pure, is_stateful
from ..source_reading import own_source, read_code_file
from . import scan_store
from .helper_bindings import UNRESOLVED, held_ref, resolve_binding, resolve_local_import, unloaded_local_imports
from .helper_code import is_mock
from .purity_report import PurityIssue, PurityReport
from .unloaded_modules import source_digest

__all__ = ["STATS", "StoredReportMismatch", "flush", "load", "save", "status"]

#: Layout of a stored file; a change makes every older one a miss.
_FORMAT = 2
_KIND = "purity-report"
#: Reports kept per function: its recent texts, and the worlds they ran in.
_MAX_ENTRIES = 8
#: How many ``__wrapped__`` / ``__func__`` steps lead from where a module
#: holds a callable to the one the walk visited.
_MAX_STEPS = 8

STATS = {"loaded": 0, "stored": 0, "missed": 0, "stale": 0, "refused": 0, "verified": 0}


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
    read off its class), ``w`` for ``__wrapped__``; None when none lead."""
    steps = ""
    for _ in range(_MAX_STEPS):
        if live is target:
            return steps
        if type(live) is types.MethodType:
            live, steps = live.__func__, steps + "f"
            continue
        inner = _wrapped(live)
        if inner is None:
            return None
        live, steps = inner, steps + "w"
    return None


def _follow(live: Any, steps: str) -> Any:
    for step in steps:
        if step == "f":
            if type(live) is not types.MethodType:
                return UNRESOLVED
            live = live.__func__
        else:
            live = _wrapped(live)
            if live is None:
                return UNRESOLVED
    return live


def _home(obj: Any) -> tuple[str, tuple[str, ...]] | None:
    base = obj.__func__ if type(obj) is types.MethodType else obj
    module = getattr(base, "__module__", None) if isinstance(base, (type, types.FunctionType)) else None
    qualname = getattr(base, "__qualname__", None) if module else None
    if not isinstance(module, str) or not isinstance(qualname, str) or "<" in qualname:
        return None
    return module, tuple(qualname.split("."))


def _path_of(obj: Any, *also: tuple[str, tuple[str, ...]] | None) -> list:
    """``[module, chain, steps]`` that lead back to *obj* now: its home
    (module and qualified name), or one of *also* (where a caller names it)."""
    for path in (_home(obj), *also):
        if path is None:
            continue
        live = resolve_binding(*path)
        if live is UNRESOLVED:
            continue
        steps = _descend(live, obj)
        if steps is not None:
            return [path[0], list(path[1]), steps]
    raise _Unstorable(f"no name leads to {obj!r}")


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
    return ["v", _type_name(kind), _static_str(value, "__module__"), _static_str(value, "__qualname__")]


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


# --- the guards: what a report was worked out from ---


def _guards(
    objects: list[Any],
    bindings: list[Any],
    cached: list[Any],
    local_imports: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
    unbound: tuple[tuple[str, str], ...],
    unloaded: tuple[str, ...],
    root_module: str | None,
) -> dict:
    from ..decorator.library_versions import library_versions

    files: dict[str, str | None] = {}
    for obj in objects:
        filename = _file_of(obj)
        if filename is not None and filename not in files:
            files[filename] = _file_digest(filename)
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
        "objects": [_object_guard(o) for o in objects],
        "bindings": [None if b is UNRESOLVED else _token(b) for b in bindings],
        "cached": [None if c is UNRESOLVED else _token(c) for c in cached],
        "files": sorted(files.items()),
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


def _spell(report: PurityReport, objects: list[Any], paths: list[list]) -> dict:
    """*report* as JSON: each live object it holds as the index of its path
    in *paths* (*objects* are the visited ones, found at those paths)."""
    if report.unwalkable or report.unkeyable:
        raise _Unstorable("a walk that did not finish, or a mock")
    index = {id(o): i for i, o in enumerate(objects)}

    def at(obj: Any) -> int:
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
        held.append([qualname, _path_of(obj)])
    found = {q: _find(p) for q, p in held}
    for qualname, digest in report.helper_source_hashes.items():
        path = report.helper_resolution_paths.get(qualname)
        if path is not None:
            live = resolve_binding(*path)
        else:
            live = found.get(qualname, objects[0])  # the root has no path
        if live is UNRESOLVED or callable_identity(live) != digest:
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
        helper_objects={q: _weak.ref(_found(p)) for q, p in spelled["held"]},
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
    )


# --- the root, and where its reports are kept ---


def _root_path(func: Any) -> tuple[str, tuple[str, ...], str] | None:
    """Where a module holds *func*: ``(module, chain, steps)`` as `_path_of`
    spells it. Only a plain function a name leads back to has its report
    stored (a function an ``lru_cache`` or ``functools.wraps`` wrapper holds
    is found through the wrapper)."""
    if type(func) is not types.FunctionType:
        return None
    try:
        module, chain, steps = _path_of(func)
    except _Unstorable:
        return None
    return module, tuple(chain), steps


def _spelled_root(root: tuple[str, tuple[str, ...], str]) -> list:
    return [root[0], list(root[1]), root[2]]


def _file(root: tuple[str, tuple[str, ...], str]) -> Any:
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
    digest = _raw_digest(func)
    return None if digest is None else f"{digest}:{func.__code__.co_filename}:{func.__code__.co_firstlineno}"


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
        except (_Unstorable, ValueError, TypeError, LookupError, AttributeError):
            report = None
        if report is not None:
            STATS["loaded"] += 1
            return report
    STATS["stale"] += 1
    return None


def _check(entry: dict, func: Any) -> PurityReport | None:
    objects = []
    for path in entry["paths"]:
        obj = _find(path)
        if obj is UNRESOLVED:
            return None
        objects.append(obj)
    if not objects or objects[0] is not func:
        return None
    spelled = entry["report"]
    guard = entry["guards"]
    bindings = [resolve_binding(m, tuple(c)) for m, c in spelled["bindings"]]
    cached = [_find(p) for p in spelled["cached"]]
    imports = [(m, tuple(p), tuple(d)) for m, p, d, _token_, _found in guard["imports"]]
    now = _guards(
        objects,
        bindings,
        cached,
        imports,
        tuple(tuple(u) for u in spelled["unbound"]),
        tuple(spelled["unloaded"]),
        func.__module__,
    )
    if _as_json(now) != guard:
        return None
    # Each helper as the decorator finds it on every call: through the
    # name its caller uses, which may not be its home.
    paths = {q: (m, tuple(c)) for q, m, c in spelled["paths"]}
    held = {q: _find(p) for q, p in spelled["held"]}
    for qualname, digest in spelled["hashes"]:
        path = paths.get(qualname)
        live = held.get(qualname, func) if path is None else resolve_binding(*path)  # the root has no path
        if live is UNRESOLVED or callable_identity(live) != digest:
            return None
    return _read(spelled, objects)


def _as_json(value: Any) -> Any:
    """*value* as a JSON round trip gives it back: tuples as lists."""
    if isinstance(value, dict):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


#: Records waiting for the cache dir: a process that never makes one keeps
#: only the most recent.
_MAX_PENDING = 256
#: Records worked out in this process and not written yet: (root, entry).
_PENDING: list[tuple[tuple[str, tuple[str, ...], str], dict]] = []


def save(func: Callable[..., Any], report: PurityReport, walk: Any) -> None:
    """Keep *report*, just walked from *func* by *walk*, for later processes.
    Worked out now, written when the cache dir exists (`flush`)."""
    root = _root_path(func)
    if root is None or not scan_store.enabled():
        return
    text = _text(func)
    if text is None:
        return
    try:
        entry = _entry(func, report, walk, text)
    except (_Unstorable, ValueError, TypeError, LookupError, AttributeError, RecursionError):
        STATS["refused"] += 1
        return
    _PENDING.append((root, entry))
    del _PENDING[:-_MAX_PENDING]
    flush()


def _entry(func: Any, report: PurityReport, walk: Any, text: str) -> dict:
    objects: list[Any] = [func]
    paths: list[list] = [_path_of(func)]
    seen = {id(func)}
    names = walk.visited_names()
    for obj in walk.visited():
        base = obj.__func__ if type(obj) is types.MethodType else obj
        if id(base) in seen:
            continue
        if type(obj) is types.MethodType and not isinstance(obj.__self__, type):
            raise _Unstorable("a method of an object")
        seen.add(id(base))
        qualname = names.get(id(obj))
        objects.append(base)
        paths.append(_path_of(base, report.helper_resolution_paths.get(qualname) if qualname else None))
    spelled = _spell(report, objects, paths)
    bindings = [resolve_binding(m, tuple(c)) for m, c in spelled["bindings"]]
    cached = [_find(p) for p in spelled["cached"]]
    guards = _guards(
        objects,
        bindings,
        cached,
        walk.local_imports_read(),
        report.unbound_names,
        report.unloaded_modules,
        func.__module__,
    )
    return {"text": text, "paths": paths, "report": spelled, "guards": _as_json(guards)}


def flush() -> None:
    """Write the records worked out so far, once the cache dir exists."""
    if not _PENDING:
        return
    if scan_store.store_dir() is None:
        return
    pending = list(_PENDING)
    _PENDING.clear()
    by_root: dict[tuple, list[dict]] = {}
    for root, entry in pending:
        by_root.setdefault(root, []).append(entry)
    for root, new in by_root.items():
        file = _file(root)
        if file is None:
            return
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
        paths: list[list] = [_path_of(func)]
        want = _spell(walked, objects, paths)
        objects2: list[Any] = [func]
        paths2: list[list] = [_path_of(func)]
        got = _spell(loaded, objects2, paths2)
    except _Unstorable as e:
        raise StoredReportMismatch(f"stored report of {func.__qualname__}: {e}") from None
    if _as_json(want) != _as_json(got) or paths != paths2:
        raise StoredReportMismatch(f"stored report of {func.__qualname__} differs from walking it again")


atexit.register(flush)
