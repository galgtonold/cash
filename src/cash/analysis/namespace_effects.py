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
import builtins
import functools
import inspect
import os
import pathlib
import sys
import textwrap
import types
from collections.abc import Callable, Mapping
from typing import Any

from .._memo import STATEMENTS, USER_CALLEES, LruMemo
from ..diagnostics import warn_diagnostic
from ..effects import is_open_write_mode
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..install_paths import is_user_code_file
from ..purity import is_pure
from ..source_reading import own_source, wrapper_layers
from ..tracking.randomness.state import rng_carrier_kind
from .annotations import assume_safe_block_lines, audited_lines
from .ast_util import called_names, resolve_callee
from .callee_effects import (
    all_param_names,
    argument_roots,
    iterated_sources,
    positional_param_names,
    resolve_function_def,
    scope_locals,
    source_called_names,
    source_global_rebinds,
    statement_calls,
)
from .file_effects import (
    READ_TEXT_MARKERS,
    REPEATABILITY_ACCUMULATING,
    REPEATABILITY_REPLACING,
    WRITE_TEXT_MARKERS,
    call_changes_files,
    call_repeatability,
    get_base_name,
    locally_opened_handles,
)
from .handed_callables import handed_callable_values, iteration_bindings
from .mutations import MUTATING_METHODS, RECEIVER_READONLY_WRITE_METHODS, chain_is_pure
from .read_only_callees import NOT_A_LITERAL, reads_its_arguments_only, turns_on_a_write

__all__ = [
    "statement_calls_user_writer",
    "statement_user_writer_call",
    "code_waived_lines",
    "statement_starting_workers",
    "user_callee_starting_workers",
    "user_callee_writing_files",
    "resolve_literal_path",
    "statement_written_paths",
    "statement_removed_paths",
    "statement_read_paths",
    "LISTING_TEXT_MARKERS",
    "resolve_path_list",
    "statement_saves_current_pyplot_figure",
    "statement_user_pyplot_call",
    "user_callee_using_pyplot",
    "capturable_globals",
    "notebook_global_rebinds",
    "bare_call_argument_names",
    "bare_call_arguments",
    "call_argument_names",
    "call_arguments",
    "is_estimator",
    "fits_its_receiver",
    "call_only_reads",
    "handed_through_items",
]


def statement_calls_user_writer(
    code: str,
    namespace: "Mapping[str, Any] | None",
    tree: "ast.Module | None" = None,
    *,
    appends: bool = True,
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

    Without *appends*, a function that only appends to files is not a
    writer (see :func:`user_callee_writing_files`).
    """
    found = statement_user_writer_call(code, namespace, tree, appends=appends)
    return found[1] if found else None


def statement_user_writer_call(
    code: str,
    namespace: "Mapping[str, Any] | None",
    tree: "ast.Module | None" = None,
    *,
    appends: bool = True,
) -> tuple[str, str] | None:
    """``(callee as spelled, writing function)`` for the first call in *code*
    into user code that writes files, or None.

    The spelling is what the statement says (``tl.export``), for a reason the
    user can find in their cell; the writing function is where the write is,
    which may be a helper that one calls. :func:`statement_calls_user_writer`
    is the same question without the spelling.
    """
    if not namespace or "(" not in code:
        return None
    calls = _calls(code) if tree is None else tuple(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    bound = _bindings(code) if tree is None else iteration_bindings(tree)
    for node in calls:
        found = user_callee_writing_files(resolve_callee(node.func, namespace), appends=appends)
        if found:
            return ast.unparse(node.func), found
        if node.args or node.keywords or not isinstance(node.func, ast.Name) or node.func.id in bound:
            # ``map(save, xs)``, ``s.apply(save)``, ``s.apply(lambda v:
            # save(v))``: the callee calls what it is handed, as a call by
            # name does. ``[f() for f in fns]``, ``ops[key]()``,
            # ``bus.emit()``: what is called is read out of what it holds.
            for value, function in _handed_functions(node, namespace, bound):
                found = user_callee_writing_files(function, appends=appends)
                if found:
                    return getattr(value, "__name__", None) or ast.unparse(node.func), found
    return None


#: A user function's file and code object -> the lines of its body waived by
#: ``# @cash:assume-safe`` or ``with cash.assume_safe():``, numbered as its
#: frames number them, and whether the waiver is on its ``def`` line.
_waived_lines_cache: LruMemo[tuple[str, types.CodeType], tuple[frozenset[int], bool]] = LruMemo(USER_CALLEES)


def code_waived_lines(code_obj: types.CodeType) -> tuple[frozenset[int], bool]:
    """``(waived lines, whole body)`` of the user function running *code_obj*:
    the lines a waiver covers, in the file's numbering (``frame.f_lineno``),
    and whether the waiver is on its ``def`` line. Empty for code cash does
    not read (no source, not the user's)."""
    memo_key = (code_obj.co_filename, code_obj)
    known = _waived_lines_cache.get(memo_key)
    if known is not None:
        return known
    verdict: tuple[frozenset[int], bool] = (frozenset(), False)
    if is_user_code_file(code_obj.co_filename):
        try:
            source = textwrap.dedent(own_source(code_obj))
            waived, whole = audited_lines(source) if "@cash:" in source else (frozenset(), False)
            if "with" in source:
                waived = waived | assume_safe_block_lines(ast.parse(source))
            first = code_obj.co_firstlineno - 1
            verdict = (frozenset(first + line for line in waived), whole)
        except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError, ValueError):
            pass
    _waived_lines_cache[memo_key] = verdict
    return verdict


def statement_handed_callables(code: str, namespace: "Mapping[str, Any] | None") -> list[Any]:
    """The callables of the user's the calls in *code* run without naming
    them (`handed_callable_values`): ``s.apply(save)``, ``[f() for f in
    fns]``, ``bus.emit()`` running what ``bus`` holds."""
    if not namespace or "(" not in code:
        return []
    bound = _bindings(code)
    out: dict[int, Any] = {}
    for node in _calls(code):
        if node.args or node.keywords or not isinstance(node.func, ast.Name) or node.func.id in bound:
            for value in handed_callable_values(node, namespace, bound):
                out.setdefault(id(value), value)
    return list(out.values())


def _handed_functions(
    node: ast.Call, namespace: "Mapping[str, Any]", bound: "Mapping[str, tuple[ast.expr, ...]] | None"
) -> list[tuple[Any, Any]]:
    """``(callable, the function calling it runs first)`` for what *node* is
    handed (`handed_callable_values`), once per function: the same method
    of each of many objects (``r.value() for r in rows``) is read once."""
    out: dict[int, tuple[Any, Any]] = {}
    for value in handed_callable_values(node, namespace, bound):
        function = _function_called(value)
        out.setdefault(id(function), (value, function))
    return list(out.values())


def _function_called(value: Any) -> Any:
    """The function calling *value* runs first: a method's function, a
    class's ``__init__``, a callable object's ``__call__``; *value* itself
    for anything else."""
    if isinstance(value, types.MethodType):
        return value.__func__
    if isinstance(value, types.FunctionType) or not callable(value):
        return value
    owner = value if isinstance(value, type) else type(value)
    try:
        return inspect.getattr_static(owner, "__init__" if isinstance(value, type) else "__call__")
    except AttributeError:
        return value


@functools.lru_cache(maxsize=STATEMENTS)
def _bindings(code: str) -> dict[str, tuple[ast.expr, ...]]:
    """`iteration_bindings` of *code*; read them, never change them.
    Memoised by the text, as `_calls`."""
    try:
        return iteration_bindings(ast.parse(code))
    except SyntaxError:
        return {}


@functools.lru_cache(maxsize=STATEMENTS)
def _calls(code: str) -> tuple[ast.Call, ...]:
    """The calls in *code*, in walk order; read them, never change them.

    Memoised by the text: the upstream check asks about every statement
    above the cell on every run. The callees are still resolved on every
    question, since a name can be rebound.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    return tuple(n for n in ast.walk(tree) if isinstance(n, ast.Call))


#: How many user functions one question may follow before cash gives up on
#: telling. Not a depth: every function a call reaches is followed, and a
#: seen set ends call cycles. This only stops a walk that is running away;
#: past it, the call is judged a writer, so its statement runs every time.
_CALLEE_LIMIT = 5_000

#: What a statement's writer is called when cash could not follow all of it.
_TOO_MANY_CALLEES = f"<more than {_CALLEE_LIMIT} functions to follow>"

#: A function's file and code object -> whether its body itself changes a
#: file other than by appending (replaces, deletes, moves, creates one),
#: whether it appends to one, whether it waives any of its lines
#: (``# @cash:assume-safe``, ``with cash.assume_safe():``) and the callee
#: expressions of its calls; None when cash does not look
#: into it (an installed package's code, no source, a body that does not
#: parse). Keyed by the code that RUNS, so a redefined function is read again,
#: and by its file, because code objects compare equal across files: the same
#: function in an installed package and in the user's project must not share
#: a verdict. Only what the
#: body says: the callees are resolved again on every question, because a
#: name the body calls can be rebound, and a verdict memoised with the
#: callee it had then answered for one that was no longer there.
_body_cache: LruMemo[tuple[str, types.CodeType], "tuple[bool, bool, bool, tuple[ast.expr, ...]] | None"] = LruMemo(
    USER_CALLEES
)
#: A miss in `_body_cache`, whose entries may be None.
_NOT_SEEN = object()


def _read_body(func: types.FunctionType) -> "tuple[bool, bool, bool, tuple[ast.expr, ...]] | None":
    """What *func*'s own source says: see :data:`_body_cache`."""
    code_obj = func.__code__
    if not is_user_code_file(code_obj.co_filename):
        return None
    try:
        source = textwrap.dedent(own_source(func))
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    handles = frozenset(locally_opened_handles(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    # ``# @cash:assume-safe`` on a write, or ``with cash.assume_safe():``
    # around it, is the user's word that skipping it on a hit is fine; on the
    # ``def`` line it covers the whole body. In a helper it covers every
    # caller, as it does for the analyzer's other findings.
    waived, whole_body = audited_lines(source) if "@cash:" in source else (frozenset(), False)
    if "with" in source:
        waived = waived | assume_safe_block_lines(tree, func)
    if whole_body:
        calls = []
    elif waived:
        # A waived call to another helper waives what that helper writes.
        calls = [node for node in calls if node.lineno not in waived]
    verdicts = {call_repeatability(node, handles) for node in calls}
    appends = REPEATABILITY_ACCUMULATING in verdicts
    # Not only a write that provably replaces its file: ``p.touch()``,
    # ``p.unlink()``, ``os.remove(p)``, ``shutil.move(a, b)`` change the
    # folder as much, and a hit that skipped them left a marker file missing
    # or a stale one in place. Only a provable append stays apart.
    replaces = REPEATABILITY_REPLACING in verdicts or any(
        call_changes_files(node) and call_repeatability(node, handles) != REPEATABILITY_ACCUMULATING for node in calls
    )
    # One of each spelling, and only the ones a name lookup can resolve
    # (``helper``, ``module.helper``): ``df.assign(...)`` ten times over is
    # one lookup, and ``make()(...)`` resolves to nothing at all.
    callees: dict[tuple[str, ...], ast.expr] = {}
    for node in calls:
        parts: list[str] = []
        expr = node.func
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            callees.setdefault((expr.id, *reversed(parts)), node.func)
    return replaces, appends, bool(waived) or whole_body, tuple(callees.values())


def _examined_body(func: Any) -> "tuple[types.FunctionType, bool, bool, bool, tuple[ast.expr, ...]] | None":
    """``(function, changes a file itself, appends to one itself, waives
    some of its lines, callee expressions)`` for a user function cash can
    read, or None for anything
    else: not a Python function, ``@pure``, from an installed package, or
    without source."""
    if not isinstance(func, types.FunctionType):
        func = inspect.unwrap(func) if callable(func) else func
        if not isinstance(func, types.FunctionType):
            return None
    if is_pure(func):
        return None
    code_obj = func.__code__
    memo_key = (code_obj.co_filename, code_obj)
    known = _body_cache.get(memo_key, _NOT_SEEN)
    if known is _NOT_SEEN:
        known = _read_body(func)
        _body_cache[memo_key] = known
    if known is None:
        return None
    return func, *known


def user_callee_writing_files(func: Any, *, appends: bool = True) -> str | None:
    """The user function that writes files when *func* is called, or None.

    ``save(fig, "chart.png")``, where ``save`` calls ``fig.savefig``, writes a
    file exactly as the inline ``fig.savefig(...)`` does -- which runs every
    time. Served from the cache instead, the call skipped the write, and a
    Restart & Run All left the deck without its chart. So a
    call into USER code whose body writes a file, directly or through another
    user function it calls, is judged like the write itself.

    A write that replaces a file counts (a chart, an export, a model file),
    and so does an append (``open(p, "a")``, ``to_csv(mode="a")``): an
    experiment log the notebook starts afresh and reads back below loses the
    rows a cache hit skips, in this kernel and in the next process that reads
    it. ``# @cash: assume-safe`` on the statement is the user's word that the
    write may be skipped. Without *appends* only a replacing write counts:
    the upstream replay asks that way, as re-firing an append would write
    its rows twice. Code from an installed package is not looked into: its
    source says nothing about what this call does with the user's files.
    ``@pure`` is the user's word that the function has no effect; it is
    taken.

    Every user function the call reaches is followed, however deep; a seen
    set ends call cycles. Calls were once followed three deep, and a write
    four calls down was cached and skipped on a hit. A walk past
    :data:`_CALLEE_LIMIT` functions is taken for a writer, with a warning:
    the statement runs every time rather than be served without a write.
    """
    return _first_user_callee(
        func,
        lambda current, replaces, appends_itself, waives, callees: replaces or (appends and appends_itself),
        "writes files",
    )


def user_callee_using_pyplot(func: Any) -> str | None:
    """The user function that calls a function of the pyplot module
    (``plt.gca()``, ``plt.gcf()``, ``plt.get_fignums()``, ...) when *func* is
    called, or None.

    ``plt.gca()`` written in a statement is display: it reads pyplot's
    current figure, which no key holds, and runs every time. A helper that
    does the same (``len(plt.gca().lines)``) changes no figure, so the check
    of pyplot's state before and after it sees nothing, and its result was
    served for the drawing of an earlier run. So a call into user code whose
    body calls pyplot, directly or through another user function, is judged
    like the call written in the statement. Followed as
    :func:`user_callee_writing_files` follows a call; ``@pure`` is taken.
    """
    if "matplotlib.pyplot" not in sys.modules:
        # No figure exists for a helper to read before pyplot is imported;
        # one it makes itself is seen by the before-and-after check.
        return None
    return _first_user_callee(func, _calls_pyplot, "uses pyplot")


def _calls_pyplot(
    func: types.FunctionType, replaces: bool, appends_itself: bool, waives: bool, callees: tuple[ast.expr, ...]
) -> bool:
    """Whether *func*'s body calls a function of the pyplot module: through
    the module (``plt.gca``) or a name bound to one of its functions, one it
    imports itself (``import matplotlib.pyplot as mp``), or one a default
    of its holds (``def f(get=plt.gca)``)."""
    names = func.__code__.co_names
    if "matplotlib.pyplot" in names or ("matplotlib" in names and "pyplot" in names):
        return True
    defaults = (*(func.__defaults__ or ()), *(func.__kwdefaults__ or {}).values())
    if any(_is_pyplot_object(value) for value in defaults):
        return True
    namespace = func.__globals__
    for expr in callees:
        if isinstance(expr, ast.Attribute):
            if _receiver_is_pyplot_module(expr.value, namespace):
                return True
        elif isinstance(expr, ast.Name):
            target = namespace.get(expr.id)
            if callable(target) and getattr(target, "__module__", None) == "matplotlib.pyplot":
                return True
    return False


def statement_user_pyplot_call(
    code: str,
    namespace: "Mapping[str, Any] | None",
    tree: "ast.Module | None" = None,
) -> tuple[str, str] | None:
    """``(callee as spelled, function calling pyplot)`` for the first call in
    *code* into user code that calls pyplot (:func:`user_callee_using_pyplot`),
    or None."""
    if not namespace or "(" not in code or "matplotlib.pyplot" not in sys.modules:
        return None
    calls = _calls(code) if tree is None else tuple(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    bound = _bindings(code) if tree is None else iteration_bindings(tree)
    for node in calls:
        # A method of the notebook's own class (``insp.nlines()``) as well.
        callee = resolve_callee(node.func, namespace, modules_only=False)
        if isinstance(callee, types.MethodType):
            callee = callee.__func__
        found = user_callee_using_pyplot(callee)
        if found:
            return ast.unparse(node.func), found
        # ``nlines(plt)``: user code handed pyplot reads it through a name
        # its body cannot resolve.
        if _hands_pyplot(node, namespace):
            body = _examined_body(callee)
            if body is not None:
                return ast.unparse(node.func), body[0].__name__
        if node.args or node.keywords or not isinstance(node.func, ast.Name) or node.func.id in bound:
            # ``map(nlines, xs)``, ``s.apply(nlines)``: run as a call by name is.
            for value, function in _handed_functions(node, namespace, bound):
                found = user_callee_using_pyplot(function)
                if found:
                    return getattr(value, "__name__", None) or ast.unparse(node.func), found
    return None


def _is_pyplot_object(value: Any) -> bool:
    """Whether *value* is the pyplot module or one of its functions."""
    if isinstance(value, types.ModuleType):
        return value.__name__ == "matplotlib.pyplot"
    return callable(value) and getattr(value, "__module__", None) == "matplotlib.pyplot"


def _hands_pyplot(call: ast.Call, namespace: "Mapping[str, Any]") -> bool:
    """Whether *call* passes the pyplot module or one of its functions."""
    args = [*call.args, *(keyword.value for keyword in call.keywords)]
    for arg in args:
        if isinstance(arg, ast.Starred):
            arg = arg.value
        if isinstance(arg, (ast.Name, ast.Attribute)) and _is_pyplot_object(resolve_callee(arg, namespace)):
            return True
    return False


#: Calls that start worker processes, or threads that do not run in the
#: caller's context (``ThreadPoolExecutor`` does: cash copies it). What the
#: work sent there writes, the write observer of the statement does not see.
_WORKER_STARTERS: frozenset[str] = frozenset(
    {"ProcessPoolExecutor", "Pool", "Process", "Parallel", "ThreadPool", "Thread", "get_reusable_executor"}
)


def _worker_starter(expr: ast.expr) -> str | None:
    """The name *expr* calls when it starts workers (`_WORKER_STARTERS`)."""
    name = expr.attr if isinstance(expr, ast.Attribute) else expr.id if isinstance(expr, ast.Name) else None
    return name if name in _WORKER_STARTERS else None


def _starts_workers(
    func: types.FunctionType, replaces: bool, appends_itself: bool, waives: bool, callees: tuple[ast.expr, ...]
) -> bool:
    return any(_worker_starter(expr) for expr in callees)


def user_callee_starting_workers(func: Any) -> str | None:
    """The first user function calling *func* reaches that starts worker
    processes or threads (``ProcessPoolExecutor``, ``multiprocessing.Pool``,
    ``joblib.Parallel``, ``ThreadPool``, ``threading.Thread``), or None. A
    line waived with ``# @cash:assume-safe`` does not count."""
    return _first_user_callee(func, _starts_workers, "starts worker processes")


def statement_starting_workers(
    code: str,
    namespace: "Mapping[str, Any] | None",
    tree: "ast.Module | None" = None,
) -> str | None:
    """What in *code* starts worker processes or threads, itself or in a
    function of the user's it calls (:func:`user_callee_starting_workers`),
    or None. Whatever the work run there writes is invisible to the
    statement's write observer, so a hit could skip it."""
    if "(" not in code:
        return None
    calls = _calls(code) if tree is None else tuple(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    for node in calls:
        started = _worker_starter(node.func)
        if started:
            return started
    if not namespace:
        return None
    for node in calls:
        found = user_callee_starting_workers(resolve_callee(node.func, namespace))
        if found:
            return found
        if node.args or node.keywords:
            for value in handed_callable_values(node, namespace):
                found = user_callee_starting_workers(_function_called(value))
                if found:
                    return found
    return None


def _first_user_callee(
    func: Any,
    test: "Callable[[types.FunctionType, bool, bool, bool, tuple[ast.expr, ...]], bool]",
    does: str,
) -> str | None:
    """The name of the first user function *func* reaches, itself included,
    for which *test* holds, depth first; `_TOO_MANY_CALLEES` past
    :data:`_CALLEE_LIMIT` functions, with a warning saying cash cannot tell
    whether it *does* what is asked."""
    # A ``functools.wraps`` wrapper's own body runs as well as the wrapped
    # function's: each layer is read by its own source.
    roots = [body for layer in wrapper_layers(func) if (body := _examined_body(layer)) is not None]
    if not roots:
        return None
    root = roots[0]
    seen: set[int] = {id(body[0]) for body in roots}
    held = [body[0] for body in roots]  # keeps every examined function alive, so no id is reused
    stack = list(reversed(roots))
    while stack:
        current, replaces, appends_itself, waives, callees = stack.pop()
        if test(current, replaces, appends_itself, waives, callees):
            return current.__name__
        found = []
        for expr in callees:
            # ``helper(...)`` and ``other_module.helper(...)`` alike: a
            # project module's export often hands the write to a sibling
            # module. Attributes are followed through modules only.
            body = _examined_body(resolve_callee(expr, current.__globals__))
            if body is None or id(body[0]) in seen:
                continue
            seen.add(id(body[0]))
            held.append(body[0])
            found.append(body)
        if len(seen) > _CALLEE_LIMIT:
            name = getattr(root[0], "__qualname__", root[0].__name__)
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "KEY-HELPERS-UNWALKABLE",
                f"cash cannot tell whether {name}() {does}: it reaches more than "
                f"{_CALLEE_LIMIT} functions. A statement calling it runs every time.",
                "If your code really calls this many functions, report it: cash should follow them all.",
            )
            return _TOO_MANY_CALLEES
        # Depth first, in the order the body makes its calls.
        stack.extend(reversed(found))
    return None


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
    # ``str(OUT / '*.png')``, ``os.fspath(p)``: the same path as text.
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and not isinstance(node.args[0], ast.Starred)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "str")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "fspath"
                and get_base_name(node.func.value) == "os"
            )
        )
    ):
        return resolve_literal_path(node.args[0], namespace)
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
    (the path comes from the ``open()``), the folder forms (``OUT.mkdir()``,
    ``shutil.rmtree(OUT)``) and file deletions (``PATH.unlink()``,
    ``os.remove(PATH)``; a loop variable over a folder's listing stands for
    that folder).

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
    listed = _listed_entry_folders(tree, namespace)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_path_bearing = _removal_call_path(node, namespace, listed)
        if not is_path_bearing:
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


def statement_removed_paths(
    code: str,
    tree: "ast.Module | None" = None,
    namespace: dict[str, Any] | None = None,
) -> set[str] | None:
    """The paths *code* deletes: a file (``os.remove(p)``, ``p.unlink()``),
    the folder a loop over its listing deletes entries of (``for old in
    glob(OUT + '/*.csv'): os.remove(old)``), a folder ``shutil.rmtree`` or
    ``rmdir`` removes. Empty when it deletes nothing, None when a deletion
    does not resolve."""
    if not any(m in code for m in ("remove", "unlink", "rmtree", "rmdir")):
        return set()
    if tree is None:
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
    removed: set[str] = set()
    listed = _listed_entry_folders(tree, namespace)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved, is_removal = _removal_call_path(node, namespace, listed)
        func = node.func
        if not is_removal and isinstance(func, ast.Attribute) and func.attr in ("rmtree", "rmdir"):
            if get_base_name(func.value) in ("os", "shutil"):
                resolved, is_removal = _call_path_argument(node, 0, namespace, _PATH_KWARG_NAMES), True
            elif not node.args:  # ``Path(OUT).rmdir()``
                resolved, is_removal = resolve_literal_path(func.value, namespace), True
        if not is_removal:
            continue
        if resolved is None:
            return None
        removed.add(resolved)
    return removed


#: ``os.<f>(PATH)`` calls that delete one file.
_FILE_REMOVAL_FUNCTIONS: frozenset[str] = frozenset({"remove", "unlink"})

#: Characters that make a ``glob`` pattern component a wildcard.
_GLOB_MAGIC = ("*", "?", "[")


def _removal_call_path(
    call: ast.Call,
    namespace: dict[str, Any] | None,
    listed: "tuple[dict[str, str], dict[str, str]]",
) -> tuple[str | None, bool]:
    """``(path, is_path_bearing)`` for a call that deletes a file:
    ``PATH.unlink()``, ``os.remove(PATH)`` / ``os.unlink(PATH)``.

    A *PATH* that is an entry of a loop over a folder's listing (*listed*,
    see :func:`_listed_entry_folders`) resolves to that folder: whatever it
    deletes is inside it. ``for old in OUT.glob('*.png'): old.unlink()`` is
    how a chart cell clears its folder before drawing, and unresolved it
    could never be ruled out as unread.
    """
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None, False
    if func.attr in _FILE_REMOVAL_FUNCTIONS and get_base_name(func.value) == "os":
        if not call.args or isinstance(call.args[0], ast.Starred):
            return None, True
        target = call.args[0]
    elif func.attr == "unlink" and not call.args:
        target = func.value
    else:
        return None, False
    folder = _listed_entry(target, namespace, *listed)
    if folder is not None:
        return folder, True
    return resolve_literal_path(target, namespace), True


def _listed_entry(
    node: ast.AST, namespace: dict[str, Any] | None, entries: dict[str, str], names: dict[str, str]
) -> str | None:
    """The folder *node* is an entry of, or ``None``.

    An entry is a loop variable over a listing that yields paths (``p`` in
    ``for p in OUT.glob(...)``; ``e`` or ``e.path`` over ``os.scandir``), or
    a bare name from ``os.listdir(OUT)`` joined back onto that same folder
    (``os.path.join(OUT, name)``, ``Path(OUT, name)``, ``OUT / name``).
    """
    if isinstance(node, ast.Name):
        return entries.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name):
        return entries.get(node.value.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts = [node.left, node.right]
    elif (
        isinstance(node, ast.Call)
        and not node.keywords
        and (_is_path_constructor(node.func) or _is_os_path_join(node.func))
    ):
        parts = list(node.args)
    else:
        return None
    if len(parts) != 2 or not isinstance(parts[1], ast.Name) or parts[1].id not in names:
        return None
    folder = names[parts[1].id]
    base = resolve_literal_path(parts[0], namespace)
    if base is None or os.path.normpath(base) != os.path.normpath(folder):
        return None
    return folder


def _listing_folder(node: ast.AST, namespace: dict[str, Any] | None) -> str | None:
    """The folder whose entries iterable *node* lists, or ``None``.

    ``FOLDER.glob(pattern)`` / ``.rglob(pattern)`` / ``.iterdir()`` with a
    resolvable *FOLDER*, and ``glob.glob(pattern)`` / ``glob.iglob(pattern)``
    with a resolvable pattern (its folder is the part before the first
    wildcard), each optionally inside ``sorted``/``list``/``tuple``. Every
    entry they yield lies inside that folder.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("sorted", "list", "tuple")
        and len(node.args) == 1
        and not node.keywords
    ):
        return _listing_folder(node.args[0], namespace)
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    method = node.func.attr
    if get_base_name(node.func.value) == "glob":
        if method not in ("glob", "iglob") or not node.args or isinstance(node.args[0], ast.Starred):
            return None
        pattern = resolve_literal_path(node.args[0], namespace)
        if pattern is None:
            return None
        if node.keywords and any(kw.arg in ("root_dir", "dir_fd") for kw in node.keywords):
            return None  # the pattern is relative to another folder
        parts = pattern.replace(os.sep, "/").split("/")
        fixed = []
        for part in parts:
            if any(c in part for c in _GLOB_MAGIC):
                break
            fixed.append(part)
        else:
            fixed = fixed[:-1]  # no wildcard: the pattern names one entry
        folder = "/".join(fixed)
        if not folder:
            return "/" if pattern.startswith("/") else os.curdir
        return folder
    if method in ("glob", "rglob", "iterdir", "walk"):
        return resolve_literal_path(node.func.value, namespace)
    return None


#: Methods that list a folder: ``os.listdir`` / ``os.scandir``,
#: ``glob.glob`` / ``glob.iglob``, ``Path.glob`` / ``rglob`` / ``iterdir``,
#: and ``os.walk`` / ``Path.walk`` (the folder and everything below it).
_LISTING_METHODS: frozenset[str] = frozenset({"listdir", "scandir", "glob", "iglob", "rglob", "iterdir", "walk"})

#: Text a statement that lists a folder contains (see ``_LISTING_METHODS``).
LISTING_TEXT_MARKERS: tuple[str, ...] = ("listdir", "scandir", "glob", "iterdir", "walk")


def _listing_read_folder(call: ast.Call, namespace: dict[str, Any] | None) -> str | None:
    """The folder a listing call (``_LISTING_METHODS``) reads, or ``None``
    when it does not resolve."""
    func = call.func
    if isinstance(func, ast.Attribute) and get_base_name(func.value) == "os":
        if not call.args and not call.keywords:
            return os.curdir
        return _call_path_argument(call, 0, namespace, frozenset({"path", "top"}))
    return _listing_folder(call, namespace)


def _listing_of(node: ast.AST, namespace: dict[str, Any] | None) -> tuple[str, bool] | None:
    """``(folder, yields bare names)`` for a loop's iterable that lists a
    folder, or ``None``: :func:`_listing_folder`, ``os.scandir(FOLDER)``
    (entries) and ``os.listdir(FOLDER)`` (names, relative to FOLDER)."""
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("sorted", "list", "tuple")
        and len(node.args) == 1
        and not node.keywords
    ):
        node = node.args[0]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("listdir", "scandir")
        and get_base_name(node.func.value) == "os"
    ):
        if not node.args and not node.keywords:
            return os.curdir, False  # names relative to the current folder are paths in it
        folder = _call_path_argument(node, 0, namespace, frozenset({"path"}))
        return None if folder is None else (folder, node.func.attr == "listdir")
    folder = _listing_folder(node, namespace)
    return None if folder is None else (folder, False)


def _listed_entry_folders(tree: ast.AST, namespace: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, str]]:
    """``({loop variable: folder}, {loop variable: folder})`` for loops and
    comprehensions over a folder's listing (:func:`_listing_of`): the first
    for listings that yield paths inside the folder, the second for
    ``os.listdir``, which yields names relative to it.

    A name bound anywhere else in the statement -- assigned, a second loop,
    ``with ... as``, a parameter -- is left out: it need not be an entry.
    """
    found: dict[str, tuple[str, bool]] = {}
    loop_targets: set[int] = set()
    unknown: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) or not isinstance(node.target, ast.Name):
            continue
        listing = _listing_of(node.iter, namespace)
        name = node.target.id
        if listing is None or found.get(name, listing) != listing:
            unknown.add(name)
            continue
        found[name] = listing
        loop_targets.add(id(node.target))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load) and id(node) not in loop_targets:
            unknown.add(node.id)
        elif isinstance(node, ast.arg):
            unknown.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            unknown.add(node.name)
        elif isinstance(node, ast.alias):
            unknown.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            unknown.add(node.name)
    known = {name: listing for name, listing in found.items() if name not in unknown}
    return (
        {name: folder for name, (folder, bare) in known.items() if not bare},
        {name: folder for name, (folder, bare) in known.items() if bare},
    )


def _read_call_path(
    call: ast.Call,
    namespace: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Return ``(resolved_path_or_None, is_path_bearing)`` for one READ call node.

    Mirror of :func:`_write_call_path` for the recognised file-READ forms:
    ``open(PATH)`` in a non-write mode, ``*.read_csv(PATH)`` / any ``.read_<x>``
    reader, ``np.load(PATH)`` / ``joblib.load(PATH)``, the nested-handle
    ``pickle.load(open(PATH))`` / ``json.load(open(PATH))``, and
    ``Path(PATH).read_text/read_bytes()``, and a folder's listing
    (``os.listdir(DIR)``, ``DIR.glob(...)``: the folder). ``is_path_bearing`` marks a recognised
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
        # A folder's listing reads the folder: what is written into it
        # changes what the next listing finds.
        if method in _LISTING_METHODS:
            folder = _listing_read_folder(call, namespace)
            if folder is None and method == "walk" and get_base_name(func.value) != "os":
                # `ast.walk(tree)`, `nx.walk(...)`: a walk of no folder.
                return None, False
            return folder, True
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
    if not any(m in code for m in READ_TEXT_MARKERS + LISTING_TEXT_MARKERS):
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


def notebook_global_rebinds(tree: ast.AST | None, resolve_source, namespace: Mapping[str, Any]) -> frozenset[str]:
    """Names a function defined in the notebook binds under ``global`` when
    *tree* calls it, directly or through other notebook functions.

    Such a name is the notebook's variable whether or not it is bound yet:
    the function's globals are the namespace (*namespace* itself). The first
    run of ``def setw(): global w; w = ...`` + ``setw()`` is the one that
    creates ``w``, which :func:`capturable_globals` cannot see.
    """
    out: set[str] = set()
    seen: set[str] = set()
    pending = list(called_names(tree))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        func = namespace.get(name)
        if getattr(func, "__globals__", None) is not namespace:
            continue
        try:
            source = resolve_source(name)
        except SOURCE_RETRIEVAL_ERRORS:
            continue
        if source:
            out |= source_global_rebinds(source)
            pending.extend(source_called_names(source))
    return frozenset(out)


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


def call_argument_names(tree: ast.Module | None) -> frozenset[str]:
    """Every variable a call in *tree* is handed, live or not:
    `call_arguments` without the namespace filter."""
    if tree is None:
        return frozenset()
    iterated = iterated_sources(tree)
    names: set[str] = set(bare_call_argument_names(tree))
    for call in statement_calls(tree, control_bodies=False):
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            names |= argument_roots(arg, iterated)
    return frozenset(names | handed_through_items(tree, iterated))


#: Builtins that read what they are given and change none of it.
_READING_BUILTINS = frozenset({"len", "print", "repr", "str", "type", "id", "isinstance"})

#: Libraries whose objects are read by those builtins without side effects.
#: A class of a notebook's own can do anything in ``__len__`` or ``__repr__``.
_READ_SAFE_LIBRARIES = frozenset({"builtins", "pandas", "numpy"})


def _literal(node: ast.expr):
    """A constant's value, else `NOT_A_LITERAL` (anything at run time)."""
    return node.value if isinstance(node, ast.Constant) else NOT_A_LITERAL


def _reads_only_as_called(call: ast.Call, fn) -> bool:
    """`reads_its_arguments_only` for *fn* called as *call* is written:
    ``np.median(a, overwrite_input=True)`` asks for a write, and so may a
    ``*args`` or ``**kwargs`` whose content the text does not show."""
    open_ended = any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords)
    return reads_its_arguments_only(
        fn,
        [_literal(arg) for arg in call.args if not isinstance(arg, ast.Starred)],
        {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg is not None},
        open_ended=open_ended,
    )


def _only_reads(call: ast.Call, arg_names: list[str], user_ns: dict, *, bound_by_now: bool = True) -> bool:
    """``len(df)``, ``print(arr)``, ``np.quantile(deltas, 0.95)``: the
    unshadowed builtin or a library function that only reads its arguments
    (`reads_its_arguments_only`), on pandas, numpy or builtin values only.

    Such a statement is no candidate for a change in place, so its arguments
    are not fingerprinted before and after: for a frame that is a hash of
    every value, twice, 14 s for ``len(df)`` over 1.7 million rows; for a
    list of 1.7 million values, 38.7 s for an ``np.quantile`` plain Python
    ran in 0.4 s.

    *bound_by_now* says *user_ns* holds every name the call can see when it
    runs. Read before a loop it does not: ``ax`` in ``for ax, col in
    zip(axes, cols): ax.plot(emb[col])`` is bound by the loop itself.
    """
    func = call.func
    root = func
    while isinstance(root, ast.Attribute):
        root = root.value
    if bound_by_now and isinstance(root, ast.Name) and root.id not in user_ns and not hasattr(builtins, root.id):
        # ``seaborn.countplot(data=df)`` with ``seaborn`` never imported:
        # the name fails before any argument is evaluated, let alone changed.
        return True
    if isinstance(func, ast.Name) and func.id in _READING_BUILTINS:
        if func.id in user_ns and user_ns[func.id] is not getattr(builtins, func.id):
            return False
    elif not _reads_only_as_called(call, resolve_callee(func, user_ns)):
        return False
    return all(
        type(user_ns[name]).__module__.partition(".")[0] in _READ_SAFE_LIBRARIES
        for name in arg_names
        if name in user_ns
    )


def bare_call_arguments(tree: ast.Module | None, user_ns: dict) -> frozenset[str]:
    """Names a bare expression statement hands straight to its call, which the
    call could change in place: ``im.add_qc(df)``, ``sc.tl.leiden(hv)``.

    Modules, classes, functions and immutable values are never candidates
    (:func:`call_arguments` is the engines' entry point).
    """
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        arg_names = [arg.id for arg in [*call.args, *(kw.value for kw in call.keywords)] if isinstance(arg, ast.Name)]
        if _only_reads(call, arg_names, user_ns):
            continue
        names.update(arg_names)
    return _changeable(names, user_ns)


def call_arguments(
    tree: ast.Module | None,
    user_ns: dict,
    resolve_source: Callable[[str], str | None] | None = None,
) -> frozenset[str]:
    """Names a statement's calls could change in place, to watch around it.

    The arguments of a bare call (:func:`bare_call_arguments`), and the
    variables handed to a function of the user's anywhere else in the
    statement: ``summary = add_features(df)``, ``r = work(s.d)``, ``rs =
    [work(d) for d in ds]`` (a change to ``d`` is one to what ``ds``
    holds). A helper that changes its argument and returns a summary
    changes it as much as one that returns nothing, and a hit that
    restored only ``summary`` left ``df`` without its new column.

    Only a function of the user's (:func:`users_callee`) is followed past
    a bare call: a library function whose result is kept (``m =
    np.mean(arr)``, ``X = scaler.transform(df)``) returns its answer
    instead of changing its arguments, and watching it would hash each
    frame it reads twice. Not inside a loop's or a branch's body, which
    the control structure owns.

    The globals a function of the user's it calls hands to such code
    (:func:`callee_handed_globals`: ``def push(x): heapq.heappush(H, x)``)
    are watched too; *resolve_source* reads a function's source (default:
    the function the kernel holds under the name).

    One definition for the runtime (which observes them) and the simulation
    (which reproduces the runtime's verdict), so the two cannot disagree about
    which names are candidates. Modules, classes, functions and immutable
    values are never candidates.
    """
    if tree is None:
        return frozenset()
    names: set[str] = set(bare_call_arguments(tree, user_ns))
    iterated = iterated_sources(tree)
    named_users_calls: list[ast.Call] = []
    for call in statement_calls(tree, control_bodies=False):
        if users_callee(call.func, user_ns):
            if isinstance(call.func, ast.Name):
                named_users_calls.append(call)
        elif not _asks_for_a_write(call, user_ns):
            continue
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            names |= argument_roots(arg, iterated)
    if named_users_calls:
        names |= _handed_globals(named_users_calls, user_ns, resolve_source or _live_user_function_source(user_ns))
    return _changeable(names | handed_through_items(tree, iterated), user_ns)


def _live_user_function_source(user_ns: Mapping[str, Any]) -> Callable[[str], str | None]:
    """``name -> source`` for a function of the user's the kernel holds
    under *name* (`_users_code`), else None: no scan of other modules."""

    def resolve(name: str) -> str | None:
        fn = user_ns.get(name)
        if not isinstance(fn, types.FunctionType):
            return None
        code = fn.__code__
        if code in _USER_SOURCES:
            return _USER_SOURCES[code]
        source = None
        if _users_code(fn):
            try:
                source = own_source(fn)
            except SOURCE_RETRIEVAL_ERRORS:
                source = None
        if len(_USER_SOURCES) >= 512:
            _USER_SOURCES.clear()
        _USER_SOURCES[code] = source
        return source

    return resolve


#: The source of a function of the user's, by its code object: one
#: ``def`` run once has one text, and a statement calling it asks for it on
#: every run.
_USER_SOURCES: dict[types.CodeType, str | None] = {}


#: Builtins that read what they are handed and change none of it, unshadowed.
ARGUMENT_READING_BUILTINS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "format", "frozenset", "hash",
        "int", "isinstance", "issubclass", "len", "list", "max", "min", "print", "range", "repr", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "type", "zip", "id", "callable", "chr", "ord",
    }
)  # fmt: skip


def function_handoffs(
    fdef: ast.FunctionDef | ast.AsyncFunctionDef,
    user_ns: Mapping[str, Any],
    resolve_source: Callable[[str], str | None] | None,
    seen: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str]]:
    """``(params, globals)``: the parameters and the free names the body of
    *fdef* hands to code whose effect on them its text cannot decide.

    ``def push(h, x): heapq.heappush(h, x)`` hands ``h``,
    ``def mix(): random.Random(s).shuffle(DECK)`` hands ``DECK``: nothing
    in the text tells a library function that changes what it is handed
    from one that reads it, so a caller watches what it hands such a
    function (a fingerprint before and after), as it watches what it hands
    the library function itself. Handed means: to a library function or a
    method no rule knows to only read (``call_only_reads``, the unshadowed
    reading builtins), as the receiver of a method not known to be pure,
    or to a further function of the user's that hands that parameter on
    (*resolve_source*, followed once per name). What a ``def`` or a
    ``lambda`` in the body would do runs later, if ever, and is left out;
    a lambda handed to a call is followed (``map(lambda v: push(h, v), xs)``).
    """
    params, local, iterated, calls, reads, called = _function_shape(fdef)
    # The answer depends on the text and on what the names it reads are
    # bound to (which callee a call reaches, what a global holds), the
    # source of the functions it calls by name included.
    key = (
        fdef,
        seen,
        tuple((name, id(value), type(value)) for name in reads for value in (user_ns.get(name, _UNBOUND),)),
        tuple(_source_or_none(resolve_source, name) for name in called if name not in seen),
    )
    found = _HANDOFFS.get(key)
    if found is not None:
        return found
    roots: set[str] = set()
    free: set[str] = set()
    for call in calls:
        _call_handoffs(call, local, iterated, user_ns, resolve_source, seen, roots, free)
    free |= {name for name in roots if name not in local}
    found = (frozenset(roots & params), frozenset(free))
    if len(_HANDOFFS) >= 1024:
        _HANDOFFS.clear()
    _HANDOFFS[key] = found
    return found


#: `function_handoffs` answers, by text and bindings.
_HANDOFFS: dict[tuple, tuple[frozenset[str], frozenset[str]]] = {}
_UNBOUND = object()


def _source_or_none(resolve_source: Callable[[str], str | None] | None, name: str) -> str | None:
    if resolve_source is None:
        return None
    try:
        return resolve_source(name)
    except Exception:  # noqa: BLE001 - an unreadable callee is not one that is read
        return None


@functools.lru_cache(maxsize=512)
def _function_shape(
    fdef: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[
    frozenset[str], frozenset[str], dict[str, frozenset[str]], tuple[ast.Call, ...], tuple[str, ...], tuple[str, ...]
]:
    """What :func:`function_handoffs` reads from the text of *fdef* alone:
    its parameters, its locals, what its loops iterate, its calls, the
    free names it reads and those it calls. Memoised per tree
    (`resolve_function_def` hands out one per source), which is only read."""
    local = scope_locals(fdef)
    calls = tuple(calls_with_lambdas(fdef))
    reads = sorted({n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and n.id not in local})
    called = sorted({c.func.id for c in calls if isinstance(c.func, ast.Name) and c.func.id not in local})
    return frozenset(all_param_names(fdef)), local, iterated_sources(fdef), calls, tuple(reads), tuple(called)


def calls_with_lambdas(tree: ast.AST, *, control_bodies: bool = True) -> list[ast.Call]:
    """The calls *tree* makes as it runs (`statement_calls`), and those in
    the body of a ``lambda`` it hands to one of them (``map(lambda v:
    push(h, v), xs)``)."""
    calls = statement_calls(tree, control_bodies=control_bodies)
    pending = list(calls)
    while pending:
        call = pending.pop()
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            if isinstance(arg, ast.Lambda):
                inner = statement_calls(arg.body)
                calls.extend(inner)
                pending.extend(inner)
    return calls


def _call_handoffs(
    call: ast.Call,
    local: frozenset[str] | set[str],
    iterated: dict[str, frozenset[str]],
    user_ns: Mapping[str, Any],
    resolve_source: Callable[[str], str | None] | None,
    seen: frozenset[str],
    roots: set[str],
    free: set[str],
) -> None:
    """Add to *roots* the names the one *call* hands to code whose effect is
    unknown, and to *free* the free names a function of the user's it calls
    hands on (see :func:`function_handoffs`)."""
    func = call.func
    if isinstance(func, ast.Name) and func.id not in local:
        if func.id in seen:
            return
        callee = _readable_function(func.id, resolve_source)
        if callee is not None:
            callee_params, callee_free = function_handoffs(callee, user_ns, resolve_source, seen | {func.id})
            roots.update(handed_to_params(call, callee, callee_params, iterated))
            free.update(callee_free)
            return
    args = [*call.args, *(kw.value for kw in call.keywords)]
    if isinstance(func, ast.Attribute):
        base = func.value
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        while isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute):
            base = base.func.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
        if (
            isinstance(base, ast.Name)
            and not (base.id not in local and _followed_elsewhere(user_ns.get(base.id)))
            and func.attr not in RECEIVER_READONLY_WRITE_METHODS
            and not chain_is_pure(func.attr, frozenset())
        ):
            roots.update(iterated.get(base.id, frozenset({base.id})))
        if func.attr in MUTATING_METHODS:
            return  # it keeps what it is handed, and changes none of it
    if (
        isinstance(func, ast.Name)
        and func.id in ARGUMENT_READING_BUILTINS
        and func.id not in local
        and user_ns.get(func.id, getattr(builtins, func.id)) is getattr(builtins, func.id)
    ):
        return
    arg_names = [arg.id for arg in args if isinstance(arg, ast.Name)]
    if _only_reads(call, arg_names, user_ns if isinstance(user_ns, dict) else dict(user_ns)):
        return
    for arg in args:
        roots.update(
            name
            for name in argument_roots(arg, iterated)
            if name in local or not _followed_elsewhere(user_ns.get(name))
        )


def _followed_elsewhere(value: Any) -> bool:
    """A global the handoff rules leave alone: one no call can change in
    place (a module, a class, a function, an immutable value, a path, whose
    methods act on the file system and not on the object), or a random
    generator, whose draws are followed as a carrier's are."""
    if isinstance(value, (types.ModuleType, type, pathlib.PurePath, _IMMUTABLE_ARGUMENT_TYPES)):
        return True
    return inspect.isroutine(value) or rng_carrier_kind(value) is not None


def _readable_function(name: str, resolve_source: Callable[[str], str | None] | None):
    """The ``def`` of the function of the user's *name* calls, or None."""
    if resolve_source is None:
        return None
    try:
        return resolve_function_def(name, resolve_source)
    except Exception:  # noqa: BLE001 - an unreadable callee is not one that is read
        return None


def handed_to_params(
    call: ast.Call,
    fdef: ast.FunctionDef | ast.AsyncFunctionDef,
    params: frozenset[str],
    iterated: dict[str, frozenset[str]],
) -> set[str]:
    """The caller's variables *call* hands to the parameters *params* of
    *fdef*: by position, by keyword, and everything after a ``*args`` or
    into a ``**kwargs`` (whose slot cannot be told)."""
    if not params:
        return set()
    out: set[str] = set()
    positional = positional_param_names(fdef)
    vararg = fdef.args.vararg.arg if fdef.args.vararg else None
    kwarg = fdef.args.kwarg.arg if fdef.args.kwarg else None
    unknown_slots = False
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Starred):
            unknown_slots = True
        if (
            unknown_slots
            or (i >= len(positional) and vararg in params)
            or (i < len(positional) and positional[i] in params)
        ):
            out |= argument_roots(arg, iterated)
    for kw in call.keywords:
        if kw.arg is None or kw.arg in params or (kwarg in params and kw.arg not in all_param_names(fdef)):
            out |= argument_roots(kw.value, iterated)
    return out


def rewatch_exempt(
    tree: ast.Module | None,
    user_ns: Mapping[str, Any],
    resolve_source: Callable[[str], str | None] | None,
) -> frozenset[str]:
    """The names *tree* only hands, each time as a plain argument, to a
    function of the user's whose source is read (``bt =
    run_backtest(series, 3)``), or as data to a draw on a Figure/Axes
    (``ax.bar(names, totals)``).

    A verdict learned from earlier runs is learned again on new inputs
    (``MutationClassifier.start_rewatch``, ``LoopWatch``) for what a
    statement or a loop hands straight to a library call: an empty
    ``[heapq.heappush(h, x) for x in pending]`` changed nothing, and that
    says nothing about the next items. What a helper of the user's is
    handed keeps the verdict of the run that first watched it; counting it
    as changed whenever the inputs move would make every reader of
    ``series`` wait for the backtest that was handed it. Shared by both
    engines, which must not disagree on what such a run may change."""
    if tree is None:
        return frozenset()
    from .cacheability_decision import receiver_is_identity_coupled  # it imports this module

    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    safe_nodes: set[int] = set()
    for call in calls_with_lambdas(tree):
        func = call.func
        if isinstance(func, ast.Attribute):
            # A draw on a Figure/Axes (`ax.bar(names, totals)`) reads the
            # data it is handed; a buffer or a file it writes into is no data.
            root = func.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and receiver_is_identity_coupled(user_ns.get(root.id)):
                for arg in [*call.args, *(kw.value for kw in call.keywords if kw.arg is not None)]:
                    # `ax.plot(emb[col])`, `ax.bar(df.x, df.y)`: a read of the data.
                    while isinstance(arg, (ast.Subscript, ast.Attribute)) and isinstance(arg.ctx, ast.Load):
                        arg = arg.value
                    if isinstance(arg, ast.Name) and _plain_data(user_ns.get(arg.id, _UNBOUND)):
                        safe_nodes.add(id(arg))
            continue
        if resolve_source is None or not isinstance(func, ast.Name) or not users_callee(func, user_ns):
            continue
        if _readable_function(func.id, resolve_source) is None or not _result_kept(call, parents, user_ns):
            continue
        for arg in [*call.args, *(kw.value for kw in call.keywords if kw.arg is not None)]:
            if isinstance(arg, ast.Name):
                safe_nodes.add(id(arg))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and _array_like(user_ns.get(node.id, _UNBOUND))
            and _read_through_a_view(node, parents)
        ):
            safe_nodes.add(id(node))
    if not safe_nodes:
        return frozenset()
    safe: set[str] = set()
    other: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (safe if id(node) in safe_nodes else other).add(node.id)
    return frozenset(safe - other)


def _array_like(value: Any) -> bool:
    module = getattr(type(value), "__module__", "") or ""
    return module.split(".", 1)[0] in ("numpy", "pandas")


def _read_through_a_view(node: ast.Name, parents: dict[int, ast.AST]) -> bool:
    """*node* (a frame or an array) is read through an index or an
    attribute that is handed to a call (``pd.Categorical(obs[r])``,
    ``ax.scatter(emb[:, 0], emb[:, 1])``) or plotted
    (``df.loc[cc].plot(ax=ax)``), or whose method is known not to change
    it. Not a plain ``f(df)``, which hands the object itself."""
    top: ast.AST = node
    attrs: list[str] = []
    while True:
        parent = parents.get(id(top))
        if isinstance(parent, (ast.Subscript, ast.Attribute)) and parent.value is top:
            if not isinstance(parent.ctx, ast.Load):
                return False
            if isinstance(parent, ast.Attribute):
                attrs.append(parent.attr)
            top = parent
            continue
        break
    if top is node:
        return False
    if isinstance(parent, ast.keyword):
        return parent.arg is not None
    if isinstance(parent, ast.Call):
        if parent.func is not top:
            return top in parent.args
        method = attrs[-1] if attrs else ""
        return "plot" in attrs or chain_is_pure(method, frozenset())
    return False


#: Values a draw only reads: the builtin data types, and numpy's and pandas'.
_PLAIN_DATA_TYPES = (list, dict, set, frozenset, tuple, str, bytes, int, float, complex, bool, range)


def _plain_data(value: Any) -> bool:
    if isinstance(value, _PLAIN_DATA_TYPES):
        return True
    module = getattr(type(value), "__module__", "") or ""
    return module.split(".", 1)[0] in ("numpy", "pandas")


def _result_kept(node: ast.expr, parents: dict[int, ast.AST], user_ns: Mapping[str, Any]) -> bool:
    """Nothing in the statement changes what *node* answers: it is bound to
    a name, dropped, read by an operator, handed to a builtin that reads or
    copies it, or kept by a method that only stores it (``rows.append``),
    directly or as an item of a display. Not ``f(data)[0].append(1)`` nor
    ``np.copyto(f(data), x)``, which change an object *f* may hand back
    from what it was given."""
    while True:
        parent = parents.get(id(node))
        if isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict)) or (
            isinstance(parent, ast.keyword) and parent.arg is not None
        ):
            node = parent
            continue
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            return all(isinstance(t, ast.Name) for t in targets)
        if isinstance(parent, (ast.Expr, ast.Compare, ast.BinOp, ast.UnaryOp)):
            return True
        if isinstance(parent, ast.Call) and parent.func is not node:
            func = parent.func
            if isinstance(func, ast.Name) and func.id in ARGUMENT_READING_BUILTINS:
                builtin = getattr(builtins, func.id)
                return user_ns.get(func.id, builtin) is builtin
            return isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS
        return False


def callee_handed_globals(
    tree: ast.Module | None,
    user_ns: Mapping[str, Any],
    resolve_source: Callable[[str], str | None] | None,
) -> frozenset[str]:
    """The globals the functions of the user's that *tree* calls by name
    (outside a loop's or a branch's body, which the structure owns) hand to
    code whose effect on them is unknown (:func:`function_handoffs`), live
    or not."""
    if tree is None or resolve_source is None:
        return frozenset()
    return _handed_globals(statement_calls(tree, control_bodies=False), user_ns, resolve_source)


def _handed_globals(
    calls: list[ast.Call],
    user_ns: Mapping[str, Any],
    resolve_source: Callable[[str], str | None] | None,
) -> frozenset[str]:
    out: set[str] = set()
    for call in calls:
        func = call.func
        if not isinstance(func, ast.Name):
            continue
        fdef = _readable_function(func.id, resolve_source)
        if fdef is None:
            continue
        out |= function_handoffs(fdef, user_ns, resolve_source, frozenset({func.id}))[1]
    return frozenset(out)


def handed_through_items(tree: ast.Module | None, iterated: dict[str, frozenset[str]]) -> frozenset[str]:
    """The variables a statement hands, item by item, to code that may
    change the items: a method no rule knows to be pure called on a
    comprehension's or a loop's target (``[r.update(tax=1) for r in
    records]``), and what a call is handed beside a ``lambda``
    (``list(map(lambda r: r.update(tax=1), records))``). Either changes
    what ``records`` holds, as handing ``r`` to a function of the user's
    would, so they are watched the same way."""
    if tree is None:
        return frozenset()
    names: set[str] = set()
    for call in statement_calls(tree, control_bodies=False):
        func = call.func
        if isinstance(func, ast.Attribute):
            base = func.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if (
                isinstance(base, ast.Name)
                and base.id in iterated
                and func.attr not in RECEIVER_READONLY_WRITE_METHODS
                and not chain_is_pure(func.attr, frozenset())
            ):
                names |= iterated[base.id]
        args = [*call.args, *(kw.value for kw in call.keywords)]
        if any(isinstance(arg, ast.Lambda) for arg in args):
            for arg in args:
                if not isinstance(arg, ast.Lambda):
                    names |= argument_roots(arg, iterated)
    return frozenset(names)


def _asks_for_a_write(call: ast.Call, user_ns: Mapping[str, Any]) -> bool:
    """A library call whose result is kept but which is asked to write into
    what it is handed: ``m = np.median(a, overwrite_input=True)`` sorts
    ``a`` in place, ``s = np.mean(a, 0, None, b)`` fills ``b``.

    The same rule `reads_its_arguments_only` applies to a bare call: an
    ``out=``, ``inplace=`` or ``overwrite_*=`` that is not the literal
    ``False`` or ``None``, by keyword for any callee, and by position for a
    function whose signature names one there.
    """
    if any(kw.arg is not None and turns_on_a_write(kw.arg, _literal(kw.value)) for kw in call.keywords):
        return True
    open_ended = any(kw.arg is None for kw in call.keywords) or any(isinstance(a, ast.Starred) for a in call.args)
    if len(call.args) < 2 and not open_ended:
        return False  # no read-only function takes a write parameter first
    fn = resolve_callee(call.func, user_ns)
    return reads_its_arguments_only(fn) and not _reads_only_as_called(call, fn)


def _changeable(names: set[str], user_ns: Mapping[str, Any]) -> frozenset[str]:
    """The live names among *names* whose value a call could change in place."""
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


def users_callee(func: ast.expr, user_ns: Mapping[str, Any]) -> bool:
    """Whether the callee *func* names is code of the user's: a function or
    class defined in the notebook or a project module, or a method of an
    object of such a class. One cash cannot name without running something
    (``make()(x)``, ``steps[i](x)``) counts as one, to be safe; a builtin or
    a library's never does."""
    target = resolve_callee(func, user_ns, modules_only=False, builtins_fallback=True)
    if target is None:
        return not (isinstance(func, ast.Name) and hasattr(builtins, func.id))
    return _users_code(target)


def _users_code(target: Any) -> bool:
    """Whether calling *target* runs a function the user wrote."""
    if isinstance(target, functools.partial):
        target = target.func
    if inspect.ismethod(target):
        target = target.__func__
    if isinstance(target, type):
        target = inspect.getattr_static(target, "__init__", None)
    elif not isinstance(target, types.FunctionType) and not inspect.isroutine(target):
        target = inspect.getattr_static(type(target), "__call__", None)
    if isinstance(target, (staticmethod, classmethod)):
        target = target.__func__
    if not isinstance(target, types.FunctionType):
        return False
    try:
        target = inspect.unwrap(target)
    except ValueError:
        return True
    code = getattr(target, "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


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


def call_only_reads(call: ast.Call, arg_names: list[str], user_ns: dict, *, bound_by_now: bool = True) -> bool:
    """Whether *call* only reads the arguments *arg_names* name: an
    unshadowed builtin or a library function known to (``len(xs)``,
    ``np.mean(a)``), on pandas, numpy or builtin values. *bound_by_now*:
    see `_only_reads`."""
    return _only_reads(call, arg_names, user_ns, bound_by_now=bound_by_now)
