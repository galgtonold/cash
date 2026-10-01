"""The decorator, the notebook and the call units agree on which globals a
function changes.

The decorator warns when a cached function reads a module global that some
function in the module changes; the notebook makes a callee's changed globals
outputs of the statement that calls it, and a call unit keys on them. All
three read ``free_vars_mutated_in_function``. A name the function binds itself
(a loop or ``with`` target, an annotated local) is not a global, every
mutating method counts (``popitem``, ``setdefault``, ``inplace=True``), and a
helper's writes count for its caller however deep the call.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap
import time

import pytest

from cash.analysis.callee_effects import callee_global_mutations, source_global_mutations
from cash.analysis.code_analyzer import CodeAnalyzer
from cash.analysis.mutable_globals import modified_globals_in_source
from cash.analysis.purity_analyzer import PurityAnalyzer
from cash.notebook.call_key import callee_mutated_globals

pytestmark = pytest.mark.core


CASES = [
    ("def f(rows):\n    for r in rows:\n        r.append(1)\n", set()),
    ("def f():\n    acc: list = []\n    acc.append(1)\n", set()),
    ("def f():\n    with make() as buf:\n        buf.append(1)\n", set()),
    ("def f():\n    [x.append(1) for x in rows]\n", set()),
    ("def f():\n    try:\n        pass\n    except E as e:\n        e.args = ()\n", set()),
    ("def f():\n    REG.popitem()\n", {"REG"}),
    ("def f():\n    CACHE.setdefault(1, 2)\n", {"CACHE"}),
    ("def f():\n    DF.drop(columns=['a'], inplace=True)\n", {"DF"}),
    ("def f():\n    global N\n    N = N + 1\n", {"N"}),
    ("def f():\n    def g():\n        G.append(1)\n    return g\n", {"G"}),
    ("def f():\n    n = []\n    def g():\n        n.append(1)\n    return g\n", set()),
]


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_notebook_and_decorator_agree(source, expected):
    assert set(source_global_mutations(source)) == expected
    assert set(modified_globals_in_source(source)) == expected


def test_a_local_of_the_callee_is_not_an_output_of_the_statement():
    sources = {"f": "def f(data):\n    acc: list = []\n    for r in data:\n        acc.append(r)\n    return acc\n"}
    _, outputs = CodeAnalyzer.analyze_code_block(
        "x = f(data)", resolve_source=sources.get, user_ns={"acc": [], "r": [0], "data": [1]}
    )
    assert outputs == {"x"}


def test_a_helper_called_by_the_callee_counts():
    sources = {
        "outer": "def outer():\n    middle()\n",
        "middle": "def middle():\n    inner()\n",
        "inner": "def inner():\n    LOG.append(1)\n",
    }
    assert callee_global_mutations(ast.parse("outer()"), sources.get) == {"LOG"}


def test_mutual_recursion_ends():
    sources = {"a": "def a():\n    b()\n", "b": "def b():\n    a()\n    LOG.append(1)\n"}
    assert callee_global_mutations(ast.parse("a()"), sources.get) == {"LOG"}


def _module(tmp_path, monkeypatch, text: str):
    name = f"globals_mod_{time.monotonic_ns()}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_a_call_unit_sees_a_helpers_global_write(tmp_path, monkeypatch):
    module = _module(
        tmp_path,
        monkeypatch,
        """
        LOG = []

        def inner():
            LOG.append(1)

        def outer():
            inner()
            return len(LOG)
        """,
    )
    assert callee_mutated_globals(module.outer) == ("LOG",)


def test_the_decorator_reports_a_global_another_function_pops(tmp_path, monkeypatch):
    module = _module(
        tmp_path,
        monkeypatch,
        """
        REG = {"a": 1, "b": 2}

        def drop_one():
            REG.popitem()

        def size():
            return len(REG)
        """,
    )
    issues = PurityAnalyzer().analyze(module.size).issues
    assert [(i.kind, i.subject) for i in issues] == [("mutable_global", "REG")]
