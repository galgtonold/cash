"""Code that reaches the notebook's names through a string.

A statement's key names the functions its code names, and the globals
those functions read (`cache_key.called_function_dependencies`). Code can
reach a function without naming it: ``globals()['make'](3)``,
``eval('make(3)')``, ``vars()[name]``, ``getattr(sys.modules[__name__],
name)``, a frame's ``f_globals``. What ``make`` reads was then in no key:
``cfg = 2`` in the cell above and the notebook run again, cash served what
``make`` returned with ``cfg = 1``.

Which name the string spells is known only when the code runs, so a key
cannot name it. A statement whose code, or the code of a function it
reaches, looks the namespace up by a string is keyed so that no entry
matches it, and the runtime runs it uncached and says why. Both engines
answer from the same code objects and the same namespace.

What counts, read from the bytecode:

* naming ``globals``, ``eval``, ``exec`` or ``__import__``, or an
  ``import_module``, ``f_globals``, ``f_locals``, ``user_ns`` or
  ``user_global_ns`` attribute;
* at the top level of a statement, ``locals()`` or ``vars()``, which are
  the notebook's namespace there (in a function they are its own locals).
  ``exec("w = base * 2")`` there is read as the code it runs, as the
  statement analysis reads it;
* reading a value that IS the namespace: the notebook's dict
  (``NS = globals()`` in a cell above), ``sys.modules`` (also bound to a
  name of its own: ``from sys import modules``), or the module the
  notebook runs as (``import __main__``, then ``getattr(__main__, name)``,
  ``vars(__main__)``).

The code read is the statement's and that of the functions of the user's
it reaches, defined in a cell or in a module of their own
(`cache_key.called_function_dependencies`); a library's function is its
library's business.

``getattr(obj, name)`` on an object, ``vars(self)``, ``self.__dict__`` and
``model.modules()`` read no namespace, and do not count. Nor does
``getattr(mylib, name)`` on a module of the user's own code: the data of a
module the code reaches is keyed already (`callee_reach`).
"""

from __future__ import annotations

import ast
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from .._memo import CODE_OBJECTS, NOTEBOOK_STATEMENTS, LruMemo
from ..analysis.bytecode import instructions
from ..analysis.code_analyzer import exec_literal

__all__ = ["code_looks_up_by_string", "statement_looks_up_by_string"]

#: Builtins that read a namespace by a string wherever they are called.
_ALWAYS = frozenset({"globals", "eval", "exec", "__import__"})
#: Builtins that are the namespace at the top level, and a function's
#: own locals inside one.
_TOP_LEVEL = frozenset({"locals", "vars"})
#: Attributes that lead to a namespace read by a string.
_ATTRIBUTES = frozenset({"import_module", "f_globals", "f_locals", "user_ns", "user_global_ns"})

_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_ATTRIBUTE_LOADS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})
#: Instructions between a load and what uses it that change nothing here.
_FILLER = frozenset({"PUSH_NULL", "PRECALL", "CACHE", "NOP", "EXTENDED_ARG", "KW_NAMES"})


class _Facts(NamedTuple):
    """What one code object, its nested scopes included, says by itself."""

    #: It names an accessor that reads a namespace by a string wherever it is.
    always: bool
    #: The globals it loads: one may hold the namespace itself.
    loaded: frozenset[str]
    #: Globals it reads ``.modules`` of: ``sys.modules``, when one is ``sys``.
    modules_of: frozenset[str]


_FACTS: LruMemo[Any, _Facts] = LruMemo(CODE_OBJECTS)
_STATEMENT_CODES: LruMemo[str, tuple[types.CodeType, ...] | None] = LruMemo(NOTEBOOK_STATEMENTS)


def _scopes(code: types.CodeType) -> Iterable[types.CodeType]:
    stack = [code]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(c for c in current.co_consts if isinstance(c, types.CodeType))


def _facts(code: types.CodeType) -> _Facts:
    found = _FACTS.get(code)
    if found is not None:
        return found
    always = False
    loaded: set[str] = set()
    modules_of: set[str] = set()
    for scope in _scopes(code):
        top_level = scope.co_name == "<module>"
        instrs = [ins for ins in instructions(scope) if ins[0] not in _FILLER]
        for i, (opname, name) in enumerate(instrs):
            if not isinstance(name, str):
                continue
            after = instrs[i + 1] if i + 1 < len(instrs) else None
            if opname in _ATTRIBUTE_LOADS:
                if name in _ATTRIBUTES:
                    always = True
                continue
            if opname not in _GLOBAL_LOADS:
                continue
            loaded.add(name)
            if name in _ALWAYS:
                always = True
            elif name in _TOP_LEVEL and top_level and after is not None and after[0].startswith("CALL"):
                always = True  # ``locals()``, ``vars()``: the notebook's names
            if after is not None and after[0] in _ATTRIBUTE_LOADS and after[1] == "modules":
                modules_of.add(name)
    found = _Facts(always, frozenset(loaded), frozenset(modules_of))
    try:
        _FACTS[code] = found
    except TypeError:
        pass
    return found


def _is_notebook_module(value: Any, namespace: Mapping[str, Any]) -> bool:
    return isinstance(value, types.ModuleType) and getattr(value, "__dict__", None) is namespace


def _holds_a_namespace(value: Any, namespace: Mapping[str, Any]) -> bool:
    """Is *value* a namespace a string can be looked up in: the notebook's
    own, the module table, or the notebook's module?"""
    return value is namespace or value is sys.modules or _is_notebook_module(value, namespace)


def code_looks_up_by_string(code: types.CodeType | None, namespace: Mapping[str, Any]) -> bool:
    """Does *code* (with its nested scopes) reach a name through a string,
    its globals resolved in *namespace*? See the module docstring."""
    if code is None:
        return False
    facts = _facts(code)
    if facts.always:
        return True
    get = namespace.get
    if any(_holds_a_namespace(get(name), namespace) for name in facts.loaded if name in namespace):
        return True
    return any(get(name) is sys.modules.get("sys") for name in facts.modules_of if name in namespace)


class _ExecLiterals(ast.NodeTransformer):
    """Takes ``exec("w = base * 2")`` out of a statement's top level, keeping
    its text: the statement analysis reads it as the assignment it runs
    (`code_analyzer.exec_literal`), so its names are the statement's own
    and keyed as such. Code in a nested scope is left as it is."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def _leave(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Lambda = _leave
    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _leave

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if exec_literal(node) is None:
            return node
        self.texts.append(node.args[0].value)  # type: ignore[attr-defined]
        return ast.copy_location(ast.Constant(value=None), node)


def _statement_codes(code: str) -> tuple[types.CodeType, ...] | None:
    """What running the statement *code* runs, compiled: itself without its
    ``exec`` literals, and each literal's text. A ``def`` runs no body, and
    a decorator on it is read as its own code. None when it does not
    compile (a magic not transformed)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        definition = tree.body[0]
        tree.body = [ast.Expr(value=d) for d in definition.decorator_list]
    literals = _ExecLiterals()
    tree = ast.fix_missing_locations(literals.visit(tree))
    try:
        codes = [compile(tree, "<cash-statement>", "exec", dont_inherit=True)]
    except (SyntaxError, ValueError, TypeError):
        return None
    for text in literals.texts:
        more = _statement_codes(text)
        if more is None:
            return None
        codes.extend(more)
    return tuple(codes)


def statement_looks_up_by_string(code: str, namespace: Mapping[str, Any] | None) -> bool:
    """Does the statement *code* itself reach a name through a string?

    Its compiled form is read as a function's is (`_statement_codes`); a
    statement that does not compile reaches none this way."""
    if not code or namespace is None:
        return False
    if code in _STATEMENT_CODES:
        codes = _STATEMENT_CODES.get(code)
    else:
        codes = _statement_codes(code)
        _STATEMENT_CODES[code] = codes
    return any(code_looks_up_by_string(compiled, namespace) for compiled in codes or ())
