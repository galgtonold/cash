"""A class a function names only to test what something is.

``isinstance(x, Kind)``, ``issubclass(t, Kind)``, ``except Kind:`` and a
parameter defaulting to ``(Kind,)`` that is only ever handed to such a test
ask whether an object is of the class. None of them runs the class's code or
reads what it holds: for a class whose metaclass leaves both checks to
``type``, the answer depends on the class object alone, and on the MRO of
the object tested, which comes from somewhere else (an argument, a global)
and is keyed there. What the class's methods read is input only to code
that calls them, so the helper walk keys such a class by its code
(`HelperWalk._queue_hash_only`) without folding the data its methods read
(`PurityReport.check_only_classes`).

``nilearn._utils.logger.log(..., object_classes=(NilearnBaseEstimator,))``
tests caller frames against the estimator base class; the class's HTML-doc
property reads ``__version__``, which moves on every commit of a source
checkout, so every cached function that logs missed on every commit.
"""

from __future__ import annotations

import ast
import builtins
import textwrap
import types
from typing import Any

from .._type_memo import mro_of
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import own_source
from . import tree_memo

__all__ = ["decides_checks_itself", "function_only_tests_against", "named_only_as_check_target"]

#: `function_only_tests_against` per ``(code, name)``: a new definition is a
#: new code object.
_TESTS_ONLY: dict[tuple[types.CodeType, str], bool] = {}

#: The builtin tests whose second argument is a class or a tuple of them.
_TESTS = ("isinstance", "issubclass")
#: Calls that read a function's own locals by a string.
_LOCALS_READERS = frozenset({"locals", "vars", "eval", "exec"})


def decides_checks_itself(cls: type) -> bool:
    """Does a metaclass of *cls* answer ``isinstance`` / ``issubclass``
    itself (``ABCMeta``'s registry and ``__subclasshook__``)? Then the
    answer reads what the class holds."""
    return any(
        "__instancecheck__" in vars(meta) or "__subclasscheck__" in vars(meta)
        for meta in mro_of(type(cls))
        if meta not in (type, object)
    )


def function_only_tests_against(func: Any, name: str, cls: type) -> bool:
    """Does plain function *func* name the global *name*, bound to class
    *cls*, only to test objects against it (`named_only_as_check_target`),
    with a metaclass that leaves the test to ``type``? False when its source
    cannot be read."""
    if type(func) is not types.FunctionType or decides_checks_itself(cls):
        return False
    code = func.__code__
    slot = (code, name)
    got = _TESTS_ONLY.get(slot)
    if got is None:
        try:
            src = textwrap.dedent(own_source(func))
        except SOURCE_RETRIEVAL_ERRORS:
            src = None
        got = False if src is None else _tests_only_in(src, code, name, func.__globals__)
        if len(_TESTS_ONLY) > 4096:
            _TESTS_ONLY.clear()
        _TESTS_ONLY[slot] = got
    return got


def _tests_only_in(src: str, code: types.CodeType, name: str, namespace: dict[str, Any]) -> bool:
    """`function_only_tests_against` read from *src*, the function's
    dedented text: what `named_only_as_check_target` answers for its first
    function definition, when that is the one *code* runs. Kept across
    processes (`text_store`) with the text, the name, the function's name
    and which of the tests its globals leave to the builtins -- all it
    reads -- so a fresh process does not parse and walk every helper that
    names a class again."""
    tests = _tests_in(namespace)

    def compute() -> bool:
        try:
            tree = tree_memo.parse(src, code.co_filename)
        except SyntaxError:
            return False
        func_def = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if func_def is None or func_def.name != code.co_name:
            return False
        return _only_check_target(func_def, name, tests)

    def decoded(value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("not a verdict")
        return value

    from . import text_store

    extra = "\0".join((name, code.co_name, *sorted(tests)))
    return text_store.kept("check-target-only", src, extra, compute, bool, decoded)


def _tests_in(namespace: dict[str, Any]) -> set[str]:
    """The builtin tests (`_TESTS`) a function with globals *namespace*
    calls by their names: those its globals do not rebind."""
    return {t for t in _TESTS if namespace.get(t, getattr(builtins, t)) is getattr(builtins, t)}


def named_only_as_check_target(func_def: ast.AST, name: str, namespace: dict[str, Any]) -> bool:
    """Is every read of *name* in *func_def* (its defaults and nested scopes
    included) a class an object is tested against: the second argument of
    ``isinstance`` / ``issubclass``, the type of an ``except`` clause, alone
    or in a tuple; or in the default of a parameter that is read only so?

    False when *name* or such a parameter is bound anywhere in it, or when
    it reads its own locals by a string (``locals()``, ``eval``).

    A function of the tree, *name* and which tests *namespace* leaves to
    the builtins alone, so it is a fact of the tree (`tree_memo.fact`),
    kept across processes under the text the tree was parsed from: the
    helper walk asks it of every class a body names, and a walk run again
    because a file it read changed elsewhere (a commit) parsed and walked
    each such body with a parent map again, for the same answers.
    """
    tests = _tests_in(namespace)
    ordered = ",".join(sorted(tests))
    return tree_memo.fact(
        func_def,
        ("check_target_only", name, ordered),
        lambda node: _only_check_target(node, name, tests),
        (f"check_target_only:{name}:{ordered}", _verdict_encoded, _verdict_decoded),
    )


def _verdict_encoded(_node: ast.AST, verdict: bool) -> bool:
    return verdict


def _verdict_decoded(_node: ast.AST, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("not a verdict")
    return value


def _only_check_target(func_def: ast.AST, name: str, tests: set[str]) -> bool:
    """`named_only_as_check_target`, the builtin tests the function calls
    by name given as *tests*."""
    if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return False
    parents = _parents(func_def)
    params = _parameters_defaulting_to(func_def, name, parents)
    if params is None:
        return False
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _LOCALS_READERS:
            return False
        if isinstance(node, (ast.Global, ast.Nonlocal)) and ({name} | params) & set(node.names):
            return False
        if isinstance(node, ast.arg) and node.arg in params and not _own_parameter(node, func_def):
            return False  # a nested scope's parameter of the same name
        if not isinstance(node, ast.Name):
            continue
        if node.id == name:
            if not isinstance(node.ctx, ast.Load):
                return False
            if not (_tested_against(node, parents, tests) or _in_default(node, parents, func_def)):
                return False
        elif node.id in params:
            if not isinstance(node.ctx, ast.Load) or not _tested_against(node, parents, tests, through_tuple=False):
                return False
    return True


def _parents(root: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


def _tested_against(node: ast.AST, parents: dict[int, ast.AST], tests: set[str], *, through_tuple: bool = True) -> bool:
    """Is *node* the class side of a test: ``isinstance(x, node)``,
    ``except node:``, or an element of a tuple that is one (*through_tuple*)?"""
    child, parent = node, parents.get(id(node))
    while through_tuple and isinstance(parent, ast.Tuple) and isinstance(parent.ctx, ast.Load):
        child, parent = parent, parents.get(id(parent))
    if isinstance(parent, ast.ExceptHandler):
        return parent.type is child
    return (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in tests
        and len(parent.args) == 2
        and not parent.keywords
        and parent.args[1] is child
    )


def _default_slots(func_def: ast.AST) -> list[tuple[ast.arg, ast.AST]]:
    """``(parameter, default expression)`` for each defaulted parameter of
    *func_def* itself."""
    args = func_def.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    slots = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults))
    slots.extend((kw, d) for kw, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None)
    return slots


def _in_default(node: ast.AST, parents: dict[int, ast.AST], func_def: ast.AST) -> bool:
    """Is *node* the whole default of a parameter of *func_def*, or an
    element of a tuple that is? (Which parameters: `_parameters_defaulting_to`.)"""
    child = node
    parent = parents.get(id(node))
    while isinstance(parent, ast.Tuple) and isinstance(parent.ctx, ast.Load):
        child, parent = parent, parents.get(id(parent))
    return parent is getattr(func_def, "args", None) and any(d is child for _, d in _default_slots(func_def))


def _parameters_defaulting_to(func_def: ast.AST, name: str, parents: dict[int, ast.AST]) -> set[str] | None:
    """The parameters of *func_def* whose default holds *name* (alone or in
    a tuple of names); None when a default holds it some other way."""
    params: set[str] = set()
    for param, default in _default_slots(func_def):
        reads = [n for n in ast.walk(default) if isinstance(n, ast.Name) and n.id == name]
        if not reads:
            continue
        for read in reads:
            parent = parents.get(id(read))
            while isinstance(parent, ast.Tuple) and isinstance(parent.ctx, ast.Load) and parent is not default:
                parent = parents.get(id(parent))
            if not (read is default or parent is default and isinstance(default, ast.Tuple)):
                return None
        params.add(param.arg)
    return params


def _own_parameter(node: ast.arg, func_def: ast.AST) -> bool:
    args = func_def.args  # type: ignore[attr-defined]
    own = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    return any(node is a for a in own if a is not None)
