"""Writes a function makes into an argument its caller made: the out-argument
pattern, judged at the call sites a traced call ran.

``buf = np.empty(2); fill(x, buf)`` with ``def fill(x, out): out[0] = ...``
changes nothing that existed before the call: ``buf`` is the caller's own
new array. Read on its own, ``fill`` "changes the argument 'out' in place",
and a traced call that ran it (`cash.decorator.code_trace`) was refused as
one with a side effect a hit would skip: a cached function reaching pvlib's
solar position code was never stored.

Which objects a function was handed is a question about its callers, and a
body read on its own cannot answer it: the function may be reached through a
table, a registry, an argument or a library callback, with anyone's object.
So the trace records it. A top-level function that writes through a
subscript or an attribute (`writes_through_parameters`) keeps its start
event armed while a cached call is traced, and each start records where it
was called from: the caller's code and the instruction that called it. A
finding on its parameter is excused only when every recorded site is a call
written in a function of the user's (`site_hands_own`) that calls it by a
name or a module attribute and hands it, for that parameter, an object the
caller made itself (`purity_flow.receiver_is_fresh`), a constant, or leaves
an immutable default. A start from anywhere else -- a library calling it
back, a call through a table, ``*args`` -- keeps the finding.
"""

from __future__ import annotations

import ast
import dis
import re
import textwrap
import types
from typing import Any

from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..jit_functions import jit_python_function
from ..source_reading import own_source
from . import tree_memo
from .ast_util import resolve_callee
from .purity_flow import fresh_call_candidates, fresh_flow_nodes, receiver_is_fresh, resolve_fresh_call_candidates
from .purity_visitor import ARGUMENT_MUTATION_NOT_STORED

__all__ = ["mutated_parameter", "site_hands_own", "writes_through_parameters"]

_MUTATED = re.compile(
    r"changes (?:the argument|an element of the argument) '([^']+)' in place; "
    + re.escape(ARGUMENT_MUTATION_NOT_STORED)
)
_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, frozenset)
_STORES = frozenset({"STORE_SUBSCR", "DELETE_SUBSCR", "STORE_SLICE", "STORE_ATTR", "DELETE_ATTR"})
_RESUMABLE = 0x20 | 0x80 | 0x100 | 0x200


def mutated_parameter(issue: Any) -> str | None:
    """The parameter a finding says its function changes in place, or None."""
    found = _MUTATED.search(getattr(issue, "description", ""))
    return found.group(1) if found is not None else None


def writes_through_parameters(code: types.CodeType) -> bool:
    """Might *code*, a top-level function's, write into an object it was
    handed: it takes arguments and stores through a subscript or an
    attribute. Asked once per code, on its first start, so a scan of its
    instructions."""
    if not (code.co_argcount or code.co_kwonlyargcount) or "." in code.co_qualname:
        return False
    if code.co_flags & _RESUMABLE:
        return False
    return any(ins.opname in _STORES for ins in dis.get_instructions(code))


def site_hands_own(fn: types.FunctionType, param: str, caller: Any, caller_code: types.CodeType, lasti: int) -> bool:
    """Is the call at *lasti* in *caller_code* (the code of *caller*, a
    function of the user's) a direct call of *fn* that hands it, for
    *param*, an object the caller made itself?"""
    if type(caller) is not types.FunctionType or caller.__code__ is not caller_code:
        return False
    func_def = _function_def(caller)
    if func_def is None:
        return False
    call = _call_at(caller, caller_code, lasti, func_def)
    if call is None:
        return False
    namespace = _namespace(caller)
    callee = resolve_callee(call.func, namespace, modules_only=False)
    if (jit_python_function(callee) or callee) is not fn:
        return False
    candidates = tree_memo.fact(func_def, "fresh_call_candidates", fresh_call_candidates)
    calls = resolve_fresh_call_candidates(candidates, namespace)
    fresh = tree_memo.fact(func_def, ("fresh_flow", calls), lambda node: fresh_flow_nodes(node, calls)) | calls
    return param not in _shared_parameters(fn, call, fresh)


def _namespace(fn: types.FunctionType) -> dict[str, Any]:
    """What *fn*'s names resolve to: its globals, its closure cells over them."""
    code = fn.__code__
    if not code.co_freevars or not fn.__closure__:
        return fn.__globals__
    ns = dict(fn.__globals__)
    for name, cell in zip(code.co_freevars, fn.__closure__):
        try:
            ns[name] = cell.cell_contents
        except ValueError:  # an empty cell
            ns.pop(name, None)
    return ns


def _function_def(fn: types.FunctionType) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = tree_memo.parse(textwrap.dedent(own_source(fn)))
    except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node if node.name == fn.__code__.co_name else None
    return None


def _call_at(fn: types.FunctionType, code: types.CodeType, lasti: int, func_def: ast.AST) -> ast.Call | None:
    """The call expression in *fn*'s own source that the instruction at
    *lasti* runs, matched by the position the compiler gave it."""
    positions = list(code.co_positions())
    index = lasti // 2
    if not 0 <= index < len(positions):
        return None
    line, end_line, col, end_col = positions[index]
    if None in (line, end_line, col, end_col):
        return None
    try:
        raw = own_source(fn)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    first = raw.splitlines()[0] if raw else ""
    indent = len(first) - len(first.lstrip())
    shift = code.co_firstlineno - 1
    found = None
    for node in ast.walk(func_def):
        if (
            isinstance(node, ast.Call)
            and node.lineno + shift == line
            and node.end_lineno is not None
            and node.end_lineno + shift == end_line
            and node.col_offset + indent == col
            and node.end_col_offset is not None
            and node.end_col_offset + indent == end_col
        ):
            if found is not None:
                return None  # two calls at one position: cannot tell
            found = node
    return found


def _immutable(value: Any) -> bool:
    if isinstance(value, tuple):
        return all(_immutable(v) for v in value)
    return isinstance(value, _IMMUTABLE)


def _passes_own(arg: ast.AST, fresh_nodes: frozenset[int]) -> bool:
    return isinstance(arg, ast.Constant) or receiver_is_fresh(arg, fresh_nodes)


def _shared_parameters(fn: types.FunctionType, call: ast.Call, fresh_nodes: frozenset[int]) -> set[str]:
    """The parameters of *fn* that *call* may hand an object its caller did
    not make: every one when the call unpacks ``*args`` or ``**kwargs``."""
    code = fn.__code__
    positional = list(code.co_varnames[: code.co_argcount])
    keyword_only = list(code.co_varnames[code.co_argcount : code.co_argcount + code.co_kwonlyargcount])
    every = set(positional) | set(keyword_only)
    if any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords):
        return every
    shared: set[str] = set()
    given: set[str] = set()
    for i, arg in enumerate(call.args):
        if i >= len(positional):
            break  # into *args, which no finding names as a parameter
        given.add(positional[i])
        if not _passes_own(arg, fresh_nodes):
            shared.add(positional[i])
    for kw in call.keywords:
        given.add(kw.arg)
        if not _passes_own(kw.value, fresh_nodes):
            shared.add(kw.arg)
    defaults = dict(zip(positional[len(positional) - len(fn.__defaults__ or ()) :], fn.__defaults__ or ()))
    defaults.update(fn.__kwdefaults__ or {})
    for name in every - given:
        if name not in defaults or not _immutable(defaults[name]):
            shared.add(name)
    return shared
