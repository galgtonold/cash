"""A plain statement -- names bound to names, constants and what operators
and literal containers make of them (``plain_statement``) -- holds no call,
attribute or subscript, so the analyses that only collect those answer it
without walking it. Each answers exactly what its walk does."""

from __future__ import annotations

import ast
import os

import pytest

from cash.analysis import ast_util, callee_effects, mutations
from cash.analysis.ast_util import plain_statement
from cash.notebook import callee_reach

PLAIN = [
    "x = 1",
    "y = a + b",
    "z = [a, b]",
    "a",
    "w = v = -a if flag else (b, {1: a, 'k': gen})",
    "t = not a and b in items or mod is None",
    "u = {a, b} | {1}",
    "pass",
    "s = 'text' * n",
    "g = gen",
    "m = mod",
    "f = helper",
    "o = obj + obj",
]

NOT_PLAIN = [
    "x = f(1)",
    "x = a.b",
    "x = a[0]",
    "a, b = c",
    "x += 1",
    "x = {**d}",
    "x = [*a]",
    "x = f'{a}'",
    "x = [i for i in a]",
    "x = lambda: 1",
    "import os",
    "del x",
    "global x",
    "for i in a: pass",
    "x: int = 1",
    "x = (y := 2)",
    "a.b = 1",
    "a[0] = 1",
    "def f(): pass",
]


@pytest.mark.parametrize("code", PLAIN)
def test_plain(code):
    assert plain_statement(ast.parse(code))


@pytest.mark.parametrize("code", NOT_PLAIN)
def test_not_plain(code):
    assert not plain_statement(ast.parse(code))


def test_none_is_not_plain():
    assert not plain_statement(None)


class _Adds:
    def __add__(self, other):
        LOG.append(other)
        return self


LOG: list = []


def _helper(acc=[]):  # a mutable default, as a notebook's may be
    acc.append(1)


def _namespace():
    def produce():
        yield 1

    return {
        "a": 1,
        "b": 2,
        "n": 3,
        "flag": True,
        "items": [1, 2],
        "gen": produce(),
        "mod": os,
        "helper": _helper,
        "obj": _Adds(),
        "v": 0,
    }


def _answers(code: str) -> dict:
    tree = ast.parse(code)
    namespace = _namespace()
    return {
        "top_level_calls": mutations.top_level_calls(tree),
        "statement_calls": callee_effects.statement_calls(tree),
        "call_arg_targets": callee_effects.call_arg_targets(tree),
        "function_arg_mutations": callee_effects.function_arg_mutations(tree, lambda name: None),
        "called_names": ast_util.called_names(tree),
        "handed_names": ast_util.handed_names(tree, "no_control_bodies"),
        "calls_in": ast_util.calls_in(tree),
        "called_dotted_names": ast_util.called_dotted_names(tree),
        "module_state_writes": callee_reach.module_state_writes(code, namespace),
        "process_state_writes": callee_reach.process_state_writes(code, namespace),
    }


@pytest.mark.parametrize("code", PLAIN)
def test_each_answers_what_its_walk_does(code, monkeypatch):
    skipping = _answers(code)
    for module in (ast_util, callee_effects, mutations, callee_reach):
        monkeypatch.setattr(module, "plain_statement", lambda tree: False)
    walking = _answers(code)
    assert skipping == walking
