"""Copying a statement's syntax tree never follows a ``.parent`` link.

IPython's tracebacks (``stack_data`` / ``executing``) set ``.parent`` on every
node of each file they show, the interpreter-wide ``ast.Load()`` and
``ast.Store()`` nodes that every ``ast.parse`` result shares included. A
``copy.deepcopy`` of a seven-node call then copied the whole file the last
traceback showed: after an error in pandas, 0.6 s for each call a statement
made, for the rest of the session. Pins the work, not the behaviour: a
refactor may expect this to fail.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.code_analyzer import magic_python
from tests._cell_driver import run_cash_cell


class _Followed:
    """Counts every attempt to copy or walk into it."""

    def __init__(self) -> None:
        self.copies = 0

    def __deepcopy__(self, memo):
        self.copies += 1
        return self

    def __reduce_ex__(self, protocol):
        self.copies += 1
        return (_Followed, ())


@pytest.fixture
def parent_on_shared_nodes():
    """A ``.parent`` on the shared context nodes, as a traceback leaves it."""
    followed = _Followed()
    shared = {type(node.ctx): node.ctx for node in ast.walk(ast.parse("x = y")) if hasattr(node, "ctx")}
    assert set(shared) == {ast.Load, ast.Store}
    had = {kind: node.__dict__.get("parent", _Followed) for kind, node in shared.items()}
    for node in shared.values():
        node.parent = followed
    try:
        yield followed
    finally:
        for kind, node in shared.items():
            if had[kind] is _Followed:
                node.__dict__.pop("parent", None)
            else:
                node.parent = had[kind]


def test_copy_tree_copies_the_fields_and_shares_the_parent(parent_on_shared_nodes):
    from cash.analysis.ast_util import copy_tree  # the one tree copier; new with this test

    call = ast.parse("f(a, g(b) + 1, k=c[0])").body[0].value
    copied = copy_tree(call)
    assert ast.dump(copied) == ast.dump(call)
    assert copied is not call and copied.args[1] is not call.args[1]
    copied.args[0] = ast.Name(id="z", ctx=ast.Load())
    assert ast.unparse(call) == "f(a, g(b) + 1, k=c[0])"  # the original is untouched
    assert parent_on_shared_nodes.copies == 0


def test_a_magic_line_is_read_without_copying_the_parent(parent_on_shared_nodes):
    tree = ast.parse("for i in r:\n    get_ipython().run_line_magic('time', 'acc.append(i * k)')")
    spliced = magic_python(tree)
    assert "acc.append(i * k)" in ast.unparse(spliced)
    assert "acc.append" not in ast.unparse(tree).replace("'acc.append(i * k)'", "")
    assert parent_on_shared_nodes.copies == 0


def test_a_cell_with_calls_never_copies_the_parent(cash_magics, parent_on_shared_nodes):
    run_cash_cell(cash_magics, "def tidy(s):\n    return s.strip().lower()")
    run_cash_cell(cash_magics, "line = ' A b '\nr = tidy(line)\nparts = [tidy(p) for p in line.split()]")
    run_cash_cell(
        cash_magics, "out = []\nfor p in line.split():\n    if len(tidy(p)) > 0:\n        out.append(tidy(p + 'X'))"
    )
    ns = cash_magics.shell.user_ns
    assert ns["r"] == "a b" and ns["parts"] == ["a", "b"] and ns["out"] == ["ax", "bx"]
    assert parent_on_shared_nodes.copies == 0
