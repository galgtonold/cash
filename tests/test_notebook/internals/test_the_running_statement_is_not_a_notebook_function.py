"""Whether a notebook function exists is answered without counting the
statement cash runs, and once while no code can have made one.

The walk for the notebook functions a value holds (`held_functions`) stops
early when none exists (`notebook_functions_exist`). Since Python 3.11
``exec`` makes a function of the code it runs, so a call keyed inside a
running statement found that statement and walked a whole networkx graph to
its end (p20, 1.1 s a statement). A statement's checks asked the question
up to 13 times, 0.035-0.07 s each beside a big graph; a no is now kept
until code could have made a function -- an ``exec``, a function made from
code -- and never taken while code with the notebook's globals is running.
"""

from __future__ import annotations

import gc
import sys
import types

import pytest

from cash.notebook import held_functions
from cash.notebook.compiled_source import CASH_FILENAME_PREFIX

nx = pytest.importorskip("networkx")


def test_the_running_statement_is_not_a_notebook_function():
    namespace: dict = {"__name__": "__main__", "nfe": held_functions.notebook_functions_exist}
    code = compile("answer = nfe(globals())", CASH_FILENAME_PREFIX + "test>", "exec")
    exec(code, namespace)
    assert namespace["answer"] is False
    code = compile("answer = nfe(globals())", "<other>", "exec")
    exec(code, namespace)
    # Since 3.11 ``exec`` makes a function of the code it runs; 3.10 makes
    # none, so there no statement is a function to leave out.
    makes_a_function = sys.version_info >= (3, 11)
    assert namespace["answer"] is makes_a_function, "only cash's own statements are left out"


def test_a_lambda_made_by_the_running_statement_is_seen():
    """A no is not taken while code with the notebook's globals runs: it
    can make a function after the question without a new ``exec``."""
    namespace: dict = {"__name__": "__main__", "nfe": held_functions.notebook_functions_exist}
    code = compile(
        "first = nfe(globals()); h = [lambda: 1]; second = nfe(globals())", CASH_FILENAME_PREFIX + "t>", "exec"
    )
    exec(code, namespace)
    assert namespace["first"] is False
    assert namespace["second"] is True


def test_a_no_is_kept_until_code_could_make_a_function(monkeypatch):
    namespace: dict = {"__name__": "__main__"}
    asked = []
    real = gc.get_referrers
    monkeypatch.setattr(held_functions.gc, "get_referrers", lambda *a: asked.append(1) or real(*a))
    assert held_functions.notebook_functions_exist(namespace) is False
    assert held_functions.notebook_functions_exist(namespace) is False
    assert len(asked) == 1, "the no was not kept"
    # A function made from code, without an exec.
    held = [types.FunctionType((lambda: 0).__code__, namespace)]
    assert held_functions.notebook_functions_exist(namespace) is True
    held.clear()
    assert held_functions.notebook_functions_exist(namespace) is False
    exec("def made():\n    return 1\nkeep = [made]\ndel made", namespace)
    assert held_functions.notebook_functions_exist(namespace) is True


def test_a_call_keyed_inside_a_statement_does_not_walk_a_whole_graph(monkeypatch):
    """p20: a call keyed inside ``print(round(nx.density(G), 6))`` found
    the running statement as a notebook function and walked all of ``G``
    (1.1 s); the walk now stops at the existence check."""
    graph = nx.gnm_random_graph(3000, 20_000, seed=1)
    visited = [0]
    real = held_functions._visit_group

    def counting(kind, group, *args):
        visited[0] += len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    namespace: dict = {"__name__": "__main__", "held": held_functions.held_notebook_code, "G": graph}
    code = compile("found = held(G, globals(), 'a lineage of G')", CASH_FILENAME_PREFIX + "t>", "exec")
    exec(code, namespace)
    assert namespace["found"].codes == ()
    assert visited[0] < 2 * held_functions.EXISTENCE_CHECK_AT, visited[0]
