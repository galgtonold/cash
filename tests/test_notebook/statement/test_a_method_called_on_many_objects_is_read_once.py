"""A method called on each of many objects of the user's class is read once.

``vals = [r.value() for r in rows]``, a ``for`` loop calling ``r.value()``,
or a helper handed ``rows`` that does: the method each object binds is the
same function. It was read and its source parsed once per object, on every
run: 20,000 rows took 5-10 s of analysis per run of the cell. What a method
does that depends on its object (it changes the object, or calls what the
object holds) is still followed for every object.
"""

from __future__ import annotations

import ast
import linecache

import pytest

import cash.analysis.handed_callables as hc
from cash.analysis.handed_callables import handed_callable_values, handed_callables, iteration_bindings

SETUP = """
LOG = []
class Row:
    def __init__(self, i):
        self.i = i
        self.d = {'a': i}
        self.hook = None
    def value(self):
        return self.i * 2
    def bump(self):
        self.i += 1
        return self.i
    def run_hook(self):
        return self.hook(self.i)
def record(v):
    LOG.append(v)
    return v
def total(rows):
    return sum(r.value() for r in rows)
rows = [Row(i) for i in range(3000)]
first = rows[0]
"""


def _namespace(code: str) -> dict:
    filename = f"<many-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    return namespace


@pytest.fixture
def namespace():
    return _namespace(SETUP)


@pytest.fixture
def reads(monkeypatch):
    count = [0]
    original = hc._changes_its_first_argument

    def counting(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hc, "_changes_its_first_argument", counting)
    return count


@pytest.mark.parametrize(
    "statement",
    [
        "vals = [r.value() for r in rows]",
        "acc = 0\nfor r in rows:\n    acc += r.value()",
        "out = total(rows)",
    ],
)
def test_the_method_is_read_once_not_once_per_object(namespace, reads, statement):
    tree = ast.parse(statement)
    hc.new_analysis_round()
    handed_callables(tree, namespace)
    bound = iteration_bindings(tree)
    values = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        values.extend(handed_callable_values(call, namespace, bound))
    assert reads[0] <= 3
    assert len(values) <= 3


def test_a_method_changing_its_object_still_names_each_one(namespace):
    found = handed_callables(ast.parse("vals = [r.bump() for r in rows]"), namespace)
    assert {"rows", "first"} <= found.receivers


def test_a_method_calling_what_its_object_holds_is_followed_for_each_object(namespace):
    namespace["rows"][1234].hook = namespace["record"]
    found = handed_callables(ast.parse("vals = [r.run_hook() for r in rows]"), namespace)
    assert any("LOG.append" in source for source in found.sources)


def test_an_object_hiding_the_method_in_its_own_dict_is_read(namespace):
    namespace["rows"][2000].value = namespace["record"]
    found = handed_callables(ast.parse("vals = [r.value() for r in rows]"), namespace)
    assert any("LOG.append" in source for source in found.sources)
