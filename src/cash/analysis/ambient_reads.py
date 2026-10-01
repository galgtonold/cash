"""Which calls read ambient state, and which of those reads reach a result.

An ambient read (the clock, the environment, a fresh UUID) is an input the
key cannot see. `ambient_call` names the read a call makes, however it is
spelled, including a call to one of the user's clock helpers (a function
whose body is log lines and one ``return time.time()``: `clock_helper_read`).
`log_only_ambient_reads` finds the reads whose value only ever reaches a log
line, including through the module's own log helpers (`log_helper_names`),
which a cache hit skipping cannot change.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from typing import Any

from .._memo import CODE_OBJECTS, LruMemo
from ..effects import (
    CLOCK_WHEN_ARG_CALLS,
    ENVIRON_NAMES,
    classify_call,
    dotted_name,
    environ_membership,
    environment_input,
)
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..install_paths import is_user_code_file
from ..source_reading import getsource
from .ast_util import called_names, resolve_callee
from .file_effects import get_base_name
from .helper_bindings import build_namespace, callee_chain
from .purity_flow import LogOnlyFlow, is_log_helper, is_log_line
from .purity_policy import AMBIENT_KINDS


def ambient_call(node: ast.Call, namespace: dict[str, Any] | None) -> str | None:
    """The ambient read *node* makes, spelled canonically, or None.

    The spelling in the source first (``datetime.now()``), then what its names
    are bound to in *namespace* (:func:`cash.effects.classify_call`): users
    wrote ``import datetime as _dt; _dt.datetime.now()``, ``from datetime
    import datetime as DateTime``, ``import time as _time``, ``import os as
    _os`` and ``pd.Timestamp.now()`` freezing a timestamp with no warning,
    while the canonical spellings warned. Also ``pd.to_datetime("today")`` and
    ``pd.Timestamp("now")``.
    """
    effect = classify_call(node, namespace)
    if effect is not None and effect.kind in AMBIENT_KINDS:
        if effect.name in CLOCK_WHEN_ARG_CALLS:
            return f"{effect.name}({node.args[0].value!r})"  # type: ignore[attr-defined]
        return effect.name
    helper = clock_helper_of(node, namespace)
    if helper is not None:
        inner = clock_helper_read(helper)
        shown = inner if inner.endswith(")") else f"{inner}()"  # type: ignore[union-attr]
        return f"{'.'.join(callee_chain(node.func) or ())}() (which returns {shown})"
    return None


def clock_helper_of(node: ast.Call, namespace: dict[str, Any] | None) -> Any:
    """The clock helper (`clock_helper_read`) *node* calls, or None.

    Called by name (``now()``) or through a module, a class or a method's own
    ``self`` (``clocks.now()``, ``Clock.now()``, ``self.stamp()``). Only the
    bare name was judged, and the helper's own read is left to the call site,
    so every dotted spelling froze the clock with no warning.
    """
    chain = callee_chain(node.func)
    if not namespace or not chain or chain[0] not in namespace:
        return None
    if len(chain) == 1:
        helper = namespace[chain[0]]
    else:
        helper = resolve_callee(node.func, namespace, modules_only=False)
    helper = getattr(helper, "__func__", helper)  # a bound method or classmethod
    return helper if clock_helper_read(helper) is not None else None


def method_namespace(func: Any, func_def: ast.AST, namespace: dict[str, Any]) -> dict[str, Any]:
    """*namespace* with a method's first parameter bound to its class.

    For the clock-helper judgment only (`clock_helper_of`): ``self.stamp()``
    names ``Class.stamp`` as far as its code goes. A plain function, a
    static method or a function nested in another function is left alone.
    """
    qualname = getattr(func, "__qualname__", "") or ""
    parts = qualname.split(".")
    args = getattr(func_def, "args", None)
    positional = (args.posonlyargs + args.args) if args is not None else []
    if len(parts) < 2 or "<locals>" in parts or not positional:
        return namespace
    owner: Any = getattr(func, "__globals__", {}).get(parts[0])
    for part in parts[1:-1]:
        owner = getattr(owner, part, None) if isinstance(owner, type) else None
    if not isinstance(owner, type):
        return namespace
    try:
        raw = inspect.getattr_static(owner, parts[-1])
    except AttributeError:
        return namespace
    if isinstance(raw, staticmethod):
        return namespace
    return {**namespace, positional[0].arg: owner}


def clock_helper_read(value: Any) -> str | None:
    """The ambient read a CLOCK HELPER returns, or None.

    A clock helper is a function of the user's whose body is log lines and one
    ``return <ambient read>``: ``def mark(name): print(..., file=sys.stderr);
    return time.perf_counter()``. Calling it IS the ambient read, so it is
    judged where it is called -- where ``t0 = mark("step")`` handed only to a
    ``done(name, t0)`` that prints it cannot reach a result.
    Inside the helper it is not
    reported at all when the helper is reached from a cached function.
    """
    code = getattr(value, "__code__", None)
    if not isinstance(value, types.FunctionType) or code is None:
        return None
    if getattr(value, "_cash_cached", False) is True:
        # Judged by its own analysis. And every cached function shares its
        # wrapper's code object, which the memo below is keyed by: one cached
        # `return time.time()` made every cached callee a clock read.
        return None
    if not is_user_code_file(code.co_filename):
        # A library function (`os.path.isdir`) is not the user's helper, and
        # reading its source inside a cached call would record the read.
        return None
    known = _CLOCK_HELPER_CACHE.get(code, _NOT_JUDGED)
    if known is not _NOT_JUDGED:
        return known
    _CLOCK_HELPER_CACHE[code] = None  # a helper that calls itself
    found = None
    try:
        tree = ast.parse(textwrap.dedent(getsource(value)))
        func_def = tree.body[0] if tree.body else None
        body = list(getattr(func_def, "body", []))
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if (
            body
            and isinstance(body[-1], ast.Return)
            and isinstance(body[-1].value, ast.Call)
            and all(
                isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and is_log_line(s.value) for s in body[:-1]
            )
        ):
            namespace = build_namespace(value)
            # A read the key folds (`environment_input`) is an input, not a
            # frozen value: the helper's own walk lists it.
            if environment_input(body[-1].value, namespace, resolve_constants=True) is None:
                found = ambient_call(body[-1].value, namespace)
    except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
        found = None
    _CLOCK_HELPER_CACHE[code] = found
    return found


#: code object -> the ambient read that clock helper returns, or None.
_CLOCK_HELPER_CACHE: LruMemo[Any, str | None] = LruMemo(CODE_OBJECTS)


#: A miss in `_CLOCK_HELPER_CACHE`, whose entries may be None.
_NOT_JUDGED = object()


def log_only_ambient_reads(
    func_def: ast.AST, func: Any = None, namespace: dict[str, Any] | None = None
) -> frozenset[int]:
    """ids of the ambient reads in *func_def* whose value is only logged.

    "Logged" includes being passed to one of the module's own log helpers
    (`log_helper_names`): one project counted ~20 KEY-AMBIENT-READ lines per
    worker start from ``_log(f"... {time.perf_counter() - t0:.2f}s")``,
    none of which could reach a result.
    """
    candidates = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            if ambient_call(node, namespace) is not None:
                candidates.append(node)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and get_base_name(node.value) in ENVIRON_NAMES
        ):
            candidates.append(node)
        elif isinstance(node, ast.Attribute) and dotted_name(node) in ENVIRON_NAMES:
            candidates.append(node)
        elif environ_membership(node) is not None:
            candidates.append(node)
    if not candidates:
        return frozenset()  # the common case pays for no parent map
    flow = LogOnlyFlow(func_def, log_helper_names(func_def, func))
    return frozenset(id(n) for n in candidates if flow.only_logged(n))


def log_helper_names(func_def: ast.AST, func: Any) -> frozenset[str]:
    """Names *func_def* calls that are, in *func*'s globals, log helpers."""
    module_ns = getattr(func, "__globals__", None)
    if not isinstance(module_ns, dict):
        return frozenset()
    return frozenset(name for name in called_names(func_def) if _is_log_helper_function(module_ns.get(name)))


def _is_log_helper_function(value: Any) -> bool:
    code = getattr(value, "__code__", None)
    if not isinstance(value, types.FunctionType) or code is None:
        return False
    known = _LOG_HELPER_CACHE.get(code)
    if known is None:
        try:
            tree = ast.parse(textwrap.dedent(getsource(value)))
            known = bool(tree.body) and is_log_helper(tree.body[0])
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            known = False
        _LOG_HELPER_CACHE[code] = known
    return known


#: code object -> "is it a log helper?". Code objects are immutable, so a
#: redefined helper is a new key.
_LOG_HELPER_CACHE: LruMemo[Any, bool] = LruMemo(CODE_OBJECTS)
