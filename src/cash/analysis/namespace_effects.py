"""The part of the statement analysis that reads the live namespace.

Resolving a path argument to its value, reading a user function's source with
``inspect``, recognising an estimator or a pyplot module: everything here needs
the objects the notebook holds, which the pure-AST modules
(:mod:`~cash.analysis.mutations`, :mod:`~cash.analysis.file_effects`,
:mod:`~cash.analysis.aliases`, :mod:`~cash.analysis.callee_effects`,
:mod:`~cash.analysis.object_protocol`) never touch.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
import types
from collections.abc import Mapping
from typing import Any

from ..effects import is_open_write_mode
from ..install_paths import installed_roots, normcase_path
from ..purity import is_pure
from .ast_util import resolve_callee
from .file_effects import (
    READ_TEXT_MARKERS,
    REPEATABILITY_REPLACING,
    WRITE_TEXT_MARKERS,
    call_repeatability,
    get_base_name,
    locally_opened_handles,
)

__all__ = [
    "statement_calls_user_writer",
    "user_callee_writing_files",
    "resolve_literal_path",
    "statement_written_paths",
    "statement_read_paths",
    "resolve_path_list",
    "statement_saves_current_pyplot_figure",
    "capturable_globals",
    "bare_call_argument_names",
    "bare_call_arguments",
    "is_estimator",
    "fits_its_receiver",
]


def statement_calls_user_writer(
    code: str,
    namespace: "Mapping[str, Any] | None",
    tree: "ast.Module | None" = None,
) -> str | None:
    """The user function that writes files when *code* runs, or None.

    :func:`statement_writes_files` reads the statement's own text, so
    ``save_png(kind, path)`` -- whose ``fig.savefig`` sits in the helper's
    body -- is not a writer to it. The same call judged by
    :func:`user_callee_writing_files` is.

    Both spellings of a call into user code are resolved: a bare
    ``save_png(...)``, and ``helpers.save_png(...)`` through a module. The
    second is how a function in the user's PROJECT is normally reached, and
    it used to be skipped -- only ``ast.Name`` callees were offered to the
    predicate, so four project-module exports were all cached and a
    deleted deliverable did not come back. The analysis was never the
    problem; it was simply never asked.

    Attribute chains are followed only through MODULES
    (``helpers.io.save(...)``), never through an arbitrary object. Resolving
    ``obj.method`` would mean ``getattr`` on a value, which runs a property's
    body if the attribute happens to be one -- executing user code to decide
    whether user code may be cached. A writer reached as a method on an
    instance is therefore still not seen; that is a narrower gap, and closing
    it needs a way to look up the attribute without evaluating it.
    """
    if not namespace or "(" not in code:
        return None
    if tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        found = user_callee_writing_files(resolve_callee(node.func, namespace))
        if found:
            return found
    return None


#: How many calls deep :func:`user_callee_writing_files` follows user code.
_MAX_CALLEE_DEPTH = 3

#: (co_filename, co_firstlineno, source, depth) -> name of the writing function
#: or None. Keyed on the depth the function was examined at: below the cap a
#: search sees fewer calls, and its "no" must not answer a question asked from
#: higher up.
_callee_write_cache: dict[tuple[str, int, str, int], str | None] = {}


def user_callee_writing_files(func: Any, _depth: int = 0) -> str | None:
    """The user function that writes files when *func* is called, or None.

    ``save(fig, "chart.png")``, where ``save`` calls ``fig.savefig``, writes a
    file exactly as the inline ``fig.savefig(...)`` does -- which runs every
    time. Served from the cache instead, the call skipped the write, and a
    Restart & Run All left the deck without its chart. So a
    call into USER code whose body writes a file, directly or through another
    user function it calls, is judged like the write itself.

    Only a write that REPLACES a file counts: a chart, an export, a model
    file -- something a later run expects to find. An append (a log line, a
    call counter) is the side effect a cache hit is understood to skip, like
    a ``print``, and counting it would stop every function that logs from
    caching. Code from an installed package is not looked into: its source
    says nothing about what this call does with the user's files. ``@pure``
    is the user's word that the function has no effect; it is taken.

    Calls are followed :data:`_MAX_CALLEE_DEPTH` deep, which also ends a
    call cycle, so the answer depends only on the function and the depth.
    """

    func = inspect.unwrap(func) if callable(func) else func
    if not isinstance(func, types.FunctionType) or is_pure(func) or _depth > _MAX_CALLEE_DEPTH:
        return None
    code_obj = func.__code__
    if normcase_path(os.path.abspath(code_obj.co_filename)).startswith(installed_roots()):
        return None
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return None
    key = (code_obj.co_filename, code_obj.co_firstlineno, source, _depth)
    if key in _callee_write_cache:
        return _callee_write_cache[key]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    handles = frozenset(locally_opened_handles(tree))
    replaces = any(
        isinstance(node, ast.Call) and call_repeatability(node, handles) == REPEATABILITY_REPLACING
        for node in ast.walk(tree)
    )
    found = func.__name__ if replaces else None
    if found is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = func.__globals__.get(node.func.id)
                if callee is not None and callee is not func:
                    found = user_callee_writing_files(callee, _depth + 1)
                    if found:
                        break
    if len(_callee_write_cache) > 500:
        _callee_write_cache.clear()
    _callee_write_cache[key] = found
    return found


# Write-call method forms whose FIRST positional argument (or a common path
# keyword) names the output FILE. Deliberately excludes ``to_sql`` / ``to_gbq``
# / ``to_clipboard`` (no filesystem path), the file-handle methods
# ``write`` / ``writelines`` (the path lives on the ``open()`` that made the
# handle), and ``json``/``pickle`` ``dump`` (path on the nested ``open()``);
# those are recovered from the ``open()`` call in the same statement instead.
_PATH_ARG0_WRITE_METHODS: frozenset[str] = frozenset(
    {
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_json",
        "to_feather",
        "to_excel",
        "to_hdf",
        "to_stata",
        "savefig",
    }
)


# ``os.<f>(PATH)`` / ``shutil.<f>(PATH)`` calls that make or remove a folder.
_FOLDER_FUNCTIONS: frozenset[str] = frozenset(
    {
        "mkdir",
        "makedirs",
        "rmdir",
        "removedirs",
        "rmtree",
    }
)


# Keyword names that carry the output path across the recognised write calls.
_PATH_KWARG_NAMES: frozenset[str] = frozenset(
    {
        "path",
        "path_or_buf",
        "fname",
        "excel_writer",
        "file",
    }
)


def resolve_literal_path(node: ast.AST, namespace: dict[str, Any] | None) -> str | None:
    """Resolve a call argument to an output-path string, or ``None``.

    Only a string literal, or a simple ``Name`` bound to a ``str`` /
    ``os.PathLike`` in *namespace*, is resolvable. Anything computed — an
    f-string, a ``str.format``, an ``os.path.join``, an attribute — returns
    ``None`` so the caller stays conservative and never skips a writer whose
    target it cannot pin down. *namespace* is an injected argument (kept out of
    the module's global reach), so this function stays a pure function of its
    inputs.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and namespace is not None:
        val = namespace.get(node.id)
        if isinstance(val, str):
            return val
        if isinstance(val, os.PathLike):
            try:
                return os.fspath(val)
            except TypeError:
                return None
    # The spellings notebooks actually use: ``OUT / 'chart.png'``,
    # ``Path(OUT, 'chart.png')``, ``os.path.join(OUT, name)``,
    # ``f'{OUT}/chart.png'``. Once only a literal or a bare name
    # resolved, so ``fig.savefig(OUT / 'chart.png')`` was "unresolvable" and the
    # scope gate never suppressed it: a chart nothing reads was re-drawn for
    # every downstream cell. Each part must itself resolve, so anything
    # genuinely computed still returns None.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = resolve_literal_path(node.left, namespace)
        right = resolve_literal_path(node.right, namespace)
        if left is not None and right is not None:
            return os.path.join(left, right)
        return None
    if (
        isinstance(node, ast.Call)
        and not node.keywords
        and node.args
        and (_is_path_constructor(node.func) or _is_os_path_join(node.func))
    ):
        parts = [resolve_literal_path(a, namespace) for a in node.args]
        if all(p is not None for p in parts):
            return os.path.join(*parts)
        return None
    if isinstance(node, ast.JoinedStr) and namespace is not None:
        pieces: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
                continue
            if (
                isinstance(value, ast.FormattedValue)
                and value.conversion == -1
                and value.format_spec is None
                and isinstance(value.value, ast.Name)
            ):
                val = namespace.get(value.value.id, _UNBOUND)
                if isinstance(val, (str, int, os.PathLike)) and not isinstance(val, bool):
                    pieces.append(os.fspath(val) if isinstance(val, os.PathLike) else str(val))
                    continue
            return None
        return "".join(pieces)
    return None


_UNBOUND = object()


def _is_os_path_join(func: ast.AST) -> bool:
    """``os.path.join`` / ``path.join`` (``from os import path``)."""
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "join"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
    )


def _is_path_constructor(func: ast.AST) -> bool:
    """True for a ``Path(...)`` / ``pathlib.Path(...)`` constructor call func."""
    if isinstance(func, ast.Name):
        return func.id == "Path"
    if isinstance(func, ast.Attribute):
        return func.attr == "Path"
    return False


def _call_path_argument(
    call: ast.Call,
    index: int,
    namespace: dict[str, Any] | None,
    kwarg_names: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve the path from *call*'s positional arg *index* or a path keyword."""
    if len(call.args) > index and not isinstance(call.args[index], ast.Starred):
        return resolve_literal_path(call.args[index], namespace)
    for kw in call.keywords:
        if kw.arg and kw.arg in kwarg_names:
            return resolve_literal_path(kw.value, namespace)
    return None


def _write_call_path(
    call: ast.Call,
    namespace: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Return ``(resolved_path_or_None, is_path_bearing)`` for one call node.

    ``is_path_bearing`` marks the recognised writer forms whose own arguments
    name the output file. When it is True but the path is ``None`` the path was
    present but not statically resolvable — the caller treats that as "cannot
    verify" and stays conservative.
    """
    func = call.func
    # open(PATH, 'w'|'a'|...) — only a write mode counts.
    if isinstance(func, ast.Name) and func.id == "open":
        if is_open_write_mode(call):
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
        return None, False
    if isinstance(func, ast.Attribute):
        method = func.attr
        # A folder made or removed -- ``OUT.mkdir()``, ``os.makedirs(OUT)``,
        # ``shutil.rmtree(OUT)``: its effect is the folder. Unresolved, the
        # ``rmtree`` + ``mkdir`` a report cell starts with could never be ruled
        # out as unread, and a cell below it re-ran the whole report after a
        # restart.
        if method in ("mkdir", "rmdir") and not call.args:
            return resolve_literal_path(func.value, namespace), True
        if get_base_name(func.value) in ("os", "shutil") and method in _FOLDER_FUNCTIONS:
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
        # Path(PATH).write_text(...) / Path(PATH).write_bytes(...)
        if method in ("write_text", "write_bytes"):
            recv = func.value
            if isinstance(recv, ast.Call) and _is_path_constructor(recv.func):
                return _call_path_argument(recv, 0, namespace), True
            return None, True  # receiver path not inline -> unresolvable
        # np.save(PATH, arr) — the path is arg0, but ONLY for a numpy receiver;
        # torch.save(obj, PATH) puts the path second and PIL ``img.save(PATH)``
        # is ambiguous, so a non-numpy ``save`` stays conservative.
        if method == "save":
            base = get_base_name(func.value)
            if base in ("np", "numpy"):
                return _call_path_argument(call, 0, namespace), True
            return None, True
        if method in _PATH_ARG0_WRITE_METHODS:
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    return None, False


def statement_written_paths(
    code: str,
    tree: "ast.Module | None" = None,
    namespace: dict[str, Any] | None = None,
) -> set[str] | None:
    """Resolvable output path(s) a file-writing statement writes, or ``None``.

    Extracts the literal / resolvable output path for the common write forms:
    ``df.to_csv(PATH)`` and its ``to_parquet`` / ``to_pickle`` / ``to_json`` /
    ``to_feather`` / ``to_excel`` / ``to_hdf`` siblings, ``savefig(PATH)``,
    ``np.save(PATH, ...)``, ``open(PATH, 'w'|'wb'|'a'|...)``,
    ``Path(PATH).write_text/write_bytes(...)``, and the nested-handle forms
    ``json.dump(obj, open(PATH, ...))`` / ``pickle.dump(obj, open(PATH, ...))``
    (the path comes from the ``open()``).

    Returns the set of resolved paths only when EVERY path-bearing write call in
    the statement resolves to a string literal (or a simple ``Name`` bound to a
    ``str`` / ``os.PathLike`` in *namespace*). Returns ``None`` the moment a path
    is not statically resolvable (f-string, computed expression, unknown name),
    or no path-bearing write call is recognised — so the caller falls through to
    its conservative re-fire behaviour rather than skip a writer whose effect it
    cannot verify. Failure-tolerant: any extraction ambiguity yields ``None``.
    """
    if not any(m in code for m in WRITE_TEXT_MARKERS):
        return None
    if tree is None:
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
    paths: set[str] = set()
    saw_path_bearing = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_path_bearing = _write_call_path(node, namespace)
        if not is_path_bearing:
            continue
        saw_path_bearing = True
        if resolved is None:
            return None  # a write target we could not pin down -> conservative
        paths.add(resolved)
    if not saw_path_bearing or not paths:
        return None
    return paths


def _read_call_path(
    call: ast.Call,
    namespace: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Return ``(resolved_path_or_None, is_path_bearing)`` for one READ call node.

    Mirror of :func:`_write_call_path` for the recognised file-READ forms:
    ``open(PATH)`` in a non-write mode, ``*.read_csv(PATH)`` / any ``.read_<x>``
    reader, ``np.load(PATH)`` / ``joblib.load(PATH)``, the nested-handle
    ``pickle.load(open(PATH))`` / ``json.load(open(PATH))``, and
    ``Path(PATH).read_text/read_bytes()``. ``is_path_bearing`` marks a recognised
    reader whose args name an input file; a ``None`` path there means the target
    was present but not statically resolvable (caller stays conservative).
    """
    func = call.func
    # open(PATH) / open(PATH, 'r'|'rb'|...) -- only a NON-write mode counts.
    if isinstance(func, ast.Name) and func.id == "open":
        if is_open_write_mode(call):
            return None, False
        return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    if isinstance(func, ast.Attribute):
        method = func.attr
        # Path(PATH).read_text(...) / Path(PATH).read_bytes(...)
        if method in ("read_text", "read_bytes"):
            recv = func.value
            if isinstance(recv, ast.Call) and _is_path_constructor(recv.func):
                return _call_path_argument(recv, 0, namespace), True
            return None, True
        # np.load / numpy.load / joblib.load(PATH); pickle/json.load(open(PATH)).
        if method == "load":
            base = get_base_name(func.value)
            if call.args and isinstance(call.args[0], ast.Call):
                inner = call.args[0]
                if isinstance(inner.func, ast.Name) and inner.func.id == "open":
                    return _call_path_argument(inner, 0, namespace, _PATH_KWARG_NAMES), True
            if base in ("np", "numpy", "joblib"):
                return _call_path_argument(call, 0, namespace), True
            return None, False
        # pandas / polars readers: any ``.read_<fmt>(PATH)`` takes the path arg0.
        if method.startswith("read_"):
            return _call_path_argument(call, 0, namespace, _PATH_KWARG_NAMES), True
    return None, False


def statement_read_paths(
    code: str,
    tree: "ast.Module | None" = None,
    namespace: dict[str, Any] | None = None,
) -> set[str] | None:
    """Resolvable input path(s) a statement READS, or ``None`` when uncertain.

    Companion to :func:`statement_written_paths`, used by the re-execution
    planner to scope file-writer re-firing to writers whose output a downstream
    consumer actually reads. Returns the set of statically
    resolved read paths, an **empty set** when the statement has no path-bearing
    read at all, or ``None`` the moment a recognised reader's path is NOT
    statically resolvable (f-string / computed) -- so the caller treats the read
    set as unknown and never suppresses a writer it cannot prove is unread.
    """
    if not any(m in code for m in READ_TEXT_MARKERS):
        return set()
    if tree is None:
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
    loop_paths = _loop_variable_paths(tree, namespace)
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_path_bearing = _read_call_path(node, namespace)
        if not is_path_bearing:
            continue
        if resolved is None:
            # ``pd.read_csv(f) for f in FILES``: the elements of FILES.
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.Name) and arg.id in loop_paths:
                paths.update(loop_paths[arg.id])
                continue
            return None  # a read target we could not pin down -> unknown
        paths.add(resolved)
    return paths


#: A list of more paths than this is not worth listing: the gate stays off.
_PATH_LIST_MAX = 4096


def resolve_path_list(node: ast.AST, namespace: dict[str, Any] | None) -> list[str] | None:
    """The paths in a list, tuple or set of paths, or ``None``.

    A literal display whose every element resolves, a name bound to such a
    list in *namespace*, or either wrapped in ``sorted``/``list``/``tuple``.
    ``pd.concat([pd.read_csv(f) for f in TF])`` used to read as
    unknown, so a cell doing it re-drew an unrelated stale chart above.
    """
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if len(node.elts) > _PATH_LIST_MAX:
            return None
        out = [resolve_literal_path(e, namespace) for e in node.elts]
        return None if any(p is None for p in out) else out
    if isinstance(node, ast.Name) and namespace is not None:
        val = namespace.get(node.id)
        if not isinstance(val, (list, tuple, set, frozenset)) or len(val) > _PATH_LIST_MAX:
            return None
        out = []
        for v in val:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, os.PathLike):
                try:
                    out.append(os.fspath(v))
                except TypeError:
                    return None
            else:
                return None
        return out
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("sorted", "list", "tuple")
        and len(node.args) == 1
    ):
        return resolve_path_list(node.args[0], namespace)
    return None


def _loop_variable_paths(tree: ast.AST, namespace: dict[str, Any] | None) -> dict[str, list[str]]:
    """``{loop variable: the paths it takes}`` for ``for f in FILES`` loops and
    comprehensions whose iterable is a resolvable list of paths."""
    found: dict[str, list[str]] = {}
    unknown: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            if not isinstance(node.target, ast.Name):
                unknown.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
                continue
            values = resolve_path_list(node.iter, namespace)
            if values is None:
                unknown.add(node.target.id)
            else:
                found.setdefault(node.target.id, []).extend(values)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            # ``f = other`` inside the loop: f is no longer only the list's elements.
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                unknown.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
    return {k: v for k, v in found.items() if k not in unknown}


def _receiver_is_pyplot_module(recv: ast.AST, namespace: dict[str, Any] | None) -> bool:
    """True when *recv* is the ``matplotlib.pyplot`` MODULE, not a Figure/Axes.

    ``plt.savefig(...)`` (receiver is the module) saves pyplot's process-global
    current figure; ``fig.savefig(...)`` (receiver is a Figure) is receiver-bound
    and defended elsewhere. The only reliable separator is what the receiver name
    resolves to at runtime, so *namespace* is consulted when available.
    """
    # ``matplotlib.pyplot.savefig(...)`` -- an attribute chain ending in .pyplot.
    if isinstance(recv, ast.Attribute):
        return recv.attr == "pyplot"
    if isinstance(recv, ast.Name):
        if namespace is not None and recv.id in namespace:
            mod = namespace[recv.id]
            # A Figure/Axes has no ``__name__``; the pyplot module's is exact.
            return getattr(mod, "__name__", "") == "matplotlib.pyplot"
        # Namespace unavailable / name not bound: accept the conventional alias
        # as a conservative fallback (everyone imports pyplot as ``plt``).
        return recv.id in ("plt", "pyplot")
    return False


def statement_saves_current_pyplot_figure(
    code: str,
    namespace: dict[str, Any] | None = None,
) -> bool:
    """True when *code* saves pyplot's CURRENT figure via a module-level call.

    ``plt.savefig(path)`` writes whatever figure pyplot's process-global ``Gcf``
    registry holds. Its only variable input is the MODULE ``plt`` -- there is no
    value-level edge from the statement to the figure it saves. So when the
    re-execution planner schedules such a write while the ``plt.subplots()`` /
    ``plt.figure()`` that registered the current figure is NOT scheduled,
    re-running the write makes ``plt.gcf()`` invent a blank default figure and
    flush it over the user's chart -- a silent on-disk wrong answer.

    This isolates that undefended module-level form so the planner can refuse it.
    The receiver-bound ``fig.savefig(path)`` is NOT flagged: it is defended by the
    carrier-history pass (its input ``fig`` is a tracked carrier). Detection is
    namespace-aware where possible and falls back to the conventional ``plt`` /
    ``pyplot`` alias. Failure-tolerant: any parse/analysis error returns False.
    """
    if "savefig" not in code:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "savefig":
            if _receiver_is_pyplot_module(func.value, namespace):
                return True
    return False


def capturable_globals(names, namespace: Mapping[str, Any]) -> frozenset[str]:
    """The *names* that are real notebook variables in *namespace*.

    A name a callee's source mutates is only a variable of the caller's when it
    is bound there: otherwise it is the callee's own module global, or a
    closure cell that never appears in any namespace, and declaring it would
    invent a variable (reconstruction then tries to produce it and re-runs the
    statement). A module is never a value to capture either.
    """
    return frozenset(n for n in names if n in namespace and not isinstance(namespace[n], types.ModuleType))


#: Types a call cannot change in place.
_IMMUTABLE_ARGUMENT_TYPES = (int, float, complex, str, bytes, bool, type(None), frozenset, tuple, range)


def bare_call_argument_names(tree: ast.Module | None) -> frozenset[str]:
    """Every plain name a bare expression statement hands straight to its call,
    live or not: `bare_call_arguments` without the namespace filter."""
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            names.update(
                arg.id for arg in [*call.args, *(kw.value for kw in call.keywords)] if isinstance(arg, ast.Name)
            )
    return frozenset(names)


def bare_call_arguments(tree: ast.Module | None, user_ns: dict) -> frozenset[str]:
    """Names a bare expression statement hands straight to its call, which the
    call could change in place: ``im.add_qc(df)``, ``sc.tl.leiden(hv)``.

    One definition for the runtime (which observes them) and the simulation
    (which reproduces the runtime's verdict), so the two cannot disagree about
    which names are candidates. Modules, classes, functions and immutable
    values are never candidates.
    """
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            if isinstance(arg, ast.Name):
                names.add(arg.id)
    out: set[str] = set()
    for name in names:
        if name not in user_ns:
            continue
        value = user_ns[name]
        if isinstance(value, (types.ModuleType, type, _IMMUTABLE_ARGUMENT_TYPES)):
            continue
        if inspect.isroutine(value):
            continue
        out.add(name)
    return frozenset(out)


def is_estimator(value: object) -> bool:
    """Whether *value* duck-types as an sklearn estimator: a callable ``fit``
    and a callable ``get_params``, and not a module.

    ``get_params`` is what keeps out ``list.append``-style mutators and any
    object that merely has a ``fit`` method, so rules keyed on estimators
    never loosen general mutation handling. One predicate for the runtime and
    the simulation.
    """
    if isinstance(value, types.ModuleType):
        return False
    return callable(getattr(value, "fit", None)) and callable(getattr(value, "get_params", None))


#: Methods that fit their receiver in place, whatever they return.
FITTING_METHODS = frozenset({"fit", "partial_fit", "fit_transform", "fit_predict", "fit_resample"})


def fits_its_receiver(method: str, receiver: object) -> bool:
    """``vec.fit_transform(texts)`` / ``km.fit_predict(Z)`` on an estimator.

    Such a call returns a value AND fits the estimator it is called on. Cached
    as ``X = vec.fit_transform(texts)`` with ``X`` as the only output, a hit
    restored ``X`` and left ``vec`` unfitted -- silent in the same kernel,
    ``NotFittedError`` after a restart. Shared by the runtime and the
    simulation.
    """
    return method in FITTING_METHODS and is_estimator(receiver)
