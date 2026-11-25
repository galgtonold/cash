"""Cash's one model of Python's scoping (`cash.analysis.scope_model`) says
what Python itself says, for every kind of statement and expression.

Checked two ways over one corpus of snippets (`tests/_scope_oracle.py`):
against the ``symtable`` module, scope by scope and name by name, and against
the compiled bytecode, for what reaches the module's namespace. A kind of
syntax node this Python has and the corpus does not use fails the last test
here, so new syntax gets a snippet (and the model a rule) when it arrives.
"""

from __future__ import annotations

import ast
import sys

import pytest

from cash.analysis import scope_model as sm
from tests import _scope_oracle as oracle

pytestmark = pytest.mark.timeout(120)


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


@pytest.mark.parametrize("source", oracle.symtable_snippets())
def test_the_model_agrees_with_symtable(source):
    assert oracle.differences(source) == []


@pytest.mark.parametrize("source", oracle.snippets())
def test_the_model_agrees_with_the_compiled_code(source):
    compiled = oracle.compiled_names(source)
    model = sm.scope_model(ast.parse(source))
    now, later = model.bindings(), model.bindings(later=True)

    stored = {name for name, hows in now.items() if hows != {sm.DELETE}}
    assert stored == {name for name in compiled.stores if not _dunder(name)}

    stored_later = {name for name, hows in later.items() if hows != {sm.DELETE}}
    assert stored_later == compiled.stores_later

    # `except E as e` unbinds `e` when the handler ends.
    deleted = {name for found in (now, later) for name, hows in found.items() if hows & {sm.DELETE, sm.EXCEPT}}
    assert deleted == compiled.deletes

    # An annotation Python does not evaluate (quoted, inside a function,
    # under `from __future__ import annotations`) loads nothing; the model
    # still names what it reads.
    loaded = {name for name in compiled.loads if not _dunder(name)}
    assert {n for n in model.global_reads(annotations=False) if not _dunder(n)} <= loaded
    assert loaded <= model.global_reads()

    assert bool(model.star_imports) == compiled.star


def test_a_comprehension_variable_is_not_a_module_binding_on_any_python():
    # symtable folds a comprehension into the scope around it from 3.12 on;
    # the bytecode check above is what holds this on those versions.
    model = sm.scope_model(ast.parse("ys = [x for x in xs]"))
    assert set(model.bindings()) == {"ys"}
    assert model.global_reads() == {"xs"}


def test_a_name_of_a_class_body_is_not_seen_from_its_methods():
    model = sm.scope_model(ast.parse("class C:\n    x = 1\n    def m(self):\n        return x"))
    assert model.global_reads() == {"x"}


def test_when_python_evaluates_each_annotation_is_told_apart():
    source = "def f(a: Eager, b: 'Quoted') -> Ret:\n    v: Never = 1\nx: AtModule = 2\n"
    kinds = {
        name: scope.evaluated
        for scope in sm.scope_model(ast.parse(source)).scopes
        if scope.kind == sm.ANNOTATION
        for name in scope.reads
    }
    assert kinds == {
        "Eager": sm.EVALUATED,
        "Ret": sm.EVALUATED,
        "Quoted": sm.QUOTED,
        "Never": sm.NEVER,
        "AtModule": sm.EVALUATED,
    }


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters are Python 3.12 syntax")
def test_a_type_parameter_is_local_to_its_definition_and_its_bound_is_read():
    model = sm.scope_model(ast.parse("def ident[T: Bound](x: T) -> T:\n    return x"))
    assert model.global_reads() == {"Bound"}
    assert set(model.bindings()) == {"ident"}


def test_a_star_import_is_reported_as_binding_names_unseen():
    model = sm.scope_model(ast.parse("from os.path import *\nfrom . import x"))
    assert model.star_imports == ["os.path"]


def test_the_locals_of_a_function_and_every_name_a_tree_binds():
    tree = ast.parse(
        "def f(a, *rest, k=1):\n"
        "    global g\n"
        "    g = 1\n"
        "    import os.path\n"
        "    match a:\n"
        "        case [first, *others]: pass\n"
        "    for i in a: pass\n"
        "    ys = [(w := j) for j in a]\n"
        "    def inner(p): q = 1\n"
    )
    assert sm.function_locals(tree.body[0]) == {
        "a", "rest", "k", "os", "first", "others", "i", "ys", "w", "inner",
    }  # fmt: skip
    assert sm.bound_names(tree) >= {"f", "g", "j", "p", "q", "first", "others"}


def test_events_come_in_the_order_the_code_runs():
    model = sm.scope_model(ast.parse("total = total + step\nfor row in rows:\n    seen = row"))
    steps = [(e.kind, e.name) for e in model.events if e.kind != sm.SITE]
    assert steps == [
        (sm.READ, "total"), (sm.READ, "step"), (sm.BIND, "total"),
        (sm.READ, "rows"), (sm.BIND, "row"), (sm.READ, "row"), (sm.BIND, "seen"),
    ]  # fmt: skip


def test_code_a_call_runs_in_place_is_walked_where_the_call_stands():
    def inline(call):
        if isinstance(call.func, ast.Name) and call.func.id == "exec":
            return ast.parse(call.args[0].value)
        return None

    tree = ast.parse("exec('w = base * 2')")
    assert set(sm.scope_model(tree, inline).bindings()) == {"w"}
    assert set(sm.scope_model(tree).bindings()) == set()


def test_an_expression_nested_thousands_deep_is_walked_without_recursion():
    terms = 3000
    # Built by hand: some Pythons' own parser gives up on text this deep.
    total: ast.expr = ast.Name(id="a", ctx=ast.Load())
    for _ in range(terms - 1):
        total = ast.BinOp(left=total, op=ast.Add(), right=ast.Name(id="a", ctx=ast.Load()))
    tree = ast.Module(body=[ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=total)], type_ignores=[])
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        model = sm.scope_model(tree)
    finally:
        sys.setrecursionlimit(limit)
    assert sum(1 for e in model.events if e.kind == sm.READ) == terms
    assert set(model.bindings()) == {"x"}


def _node_kinds(base: type) -> set[type]:
    return {
        kind
        for kind in vars(ast).values()
        if isinstance(kind, type) and issubclass(kind, base) and kind is not base and not kind.__subclasses__()
    }


def test_the_corpus_uses_every_kind_of_syntax_node_this_python_has():
    used: set[type] = set()
    for source in oracle.snippets():
        used |= {type(node) for node in ast.walk(ast.parse(source))}
    bases = [ast.stmt, ast.expr, ast.pattern, ast.excepthandler, ast.comprehension, ast.arguments, ast.withitem]
    bases += [getattr(ast, name) for name in ("type_param",) if hasattr(ast, name)]
    every = set().union(*(_node_kinds(base) | {base} for base in bases))
    every = {kind for kind in every if kind.__subclasses__() == [] and not _is_abstract(kind)}
    # Nodes old Pythons parsed into; current ones keep the names for old code.
    retired = {
        "Num",
        "Str",
        "Bytes",
        "NameConstant",
        "Ellipsis",
        "Index",
        "ExtSlice",
        "Suite",
        "AugLoad",
        "AugStore",
        "Param",
    }
    missing = sorted({kind.__name__ for kind in every - used} - retired)
    assert missing == [], f"add a snippet to tests/_scope_oracle.py for: {missing}"


def _is_abstract(kind: type) -> bool:
    return kind in (ast.stmt, ast.expr, ast.pattern, ast.excepthandler)


def test_a_single_def_does_not_count_its_own_name_among_what_it_binds():
    node = ast.parse("def walk(n, acc=()):\n    for child in n: walk(child)\n    import os").body[0]
    assert sm.bound_names(node) == {"n", "acc", "child", "os"}


def test_the_globals_a_function_rebinds_follow_the_scope_that_declares_them():
    from cash.analysis.callee_effects import source_global_rebinds

    assert source_global_rebinds(
        "def f():\n    global a, b, c, d, untouched\n    a = 1\n    import b\n    def c(): pass\n    for d in (): pass\n"
    ) == {"a", "b", "c", "d"}
    assert source_global_rebinds("def f(v):\n    global m\n    match v:\n        case [m, *_]: pass\n") == {"m"}
    # A local of an inner function is not the global an outer one declares.
    assert (
        source_global_rebinds("def f():\n    global g\n    def inner():\n        g = 1\n    return g\n") == frozenset()
    )
