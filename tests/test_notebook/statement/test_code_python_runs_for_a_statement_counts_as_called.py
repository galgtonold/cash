"""Code of the user's that a statement runs without calling it by a bare name
counts as called: what it changes of the globals is the statement's.

A method called on an object or a class (a static or class method, a hook
reached through ``super()``), a context manager's ``__enter__``/``__exit__``,
a property, the special methods of an object the statement names, the
functions a closure wraps (a decorator without ``functools.wraps``, a
``compose``), a ``functools.partial``, ``np.vectorize`` or
``operator.methodcaller`` called by its own name, a ``singledispatch``
implementation. Each was left out, the statement served from the cache, and
the global list the code appends to stayed empty.
"""

from __future__ import annotations

import ast
import functools
import inspect
import linecache

import pytest

from cash.analysis.callee_effects import callee_global_mutations
from cash.analysis.handed_callables import handed_callables, hands_a_state_changing_callable

SETUP = """
import functools, operator
from functools import singledispatch
LOG = []
def write_row(r):
    LOG.append(r)
    return 1
class Store:
    def process(self, r):
        LOG.append(r)
        return 1
    @staticmethod
    def static(r):
        return write_row(r)
    @classmethod
    def klass(cls, r):
        return write_row(r)
    def pure(self, r):
        return len(r)
class Base:
    def run(self, r):
        return self.hook(r)
    def hook(self, r):
        return 0
class Job(Base):
    def run(self, r):
        return super().run(r)
    def hook(self, r):
        LOG.append(r)
        return 1
class timer:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        LOG.append('t')
class Bag:
    def __getitem__(self, k):
        LOG.append(k)
        return 1
    @property
    def value(self):
        LOG.append('p')
        return 1
class Plain:
    def __getitem__(self, k):
        return k
def with_retry(fn):
    def call(*a):
        return fn(*a)
    return call
def compose(f, g):
    return lambda x: f(g(x))
def clean(v):
    return v.strip()
@with_retry
def save_row(v):
    LOG.append(v)
    return v
@singledispatch
def note(x):
    return x
@note.register
def _(x: str):
    LOG.append(x)
    return x
store = Store()
job = Job()
bag = Bag()
plain = Plain()
pipe = compose(write_row, clean)
pz = functools.partial(write_row, 'Z')
mc = operator.methodcaller('process', 'Z')
"""


def _namespace(code: str) -> dict:
    filename = f"<implicit-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def namespace():
    return _namespace(SETUP)


def _changed_globals(statement: str, namespace: dict) -> frozenset[str]:
    tree = ast.parse(statement)
    handed = handed_callables(tree, namespace)

    def resolve(name):
        value = namespace.get(name)
        if callable(value) and hasattr(value, "__code__"):
            return inspect.getsource(value)
        return None

    return callee_global_mutations(tree, resolve, extra_sources=handed.sources)


@pytest.mark.parametrize(
    "statement",
    [
        "n = store.process('r')",
        "n = Store.static('r')",
        "n = Store.klass('r')",
        "n = job.run('r')",
        "with timer():\n    x = 1",
        "a = bag['k']",
        "a = bag.value",
        "out = save_row('Z')",
        "out = pipe(' Z ')",
        "out = pz()",
        "out = mc(store)",
        "out = operator.methodcaller('process', 'Z')(store)",
        "out = note('Z')",
    ],
)
def test_the_code_python_runs_counts_as_called(namespace, statement):
    assert "LOG" in _changed_globals(statement, namespace)


@pytest.mark.parametrize("statement", ["n = store.pure('r')", "a = plain['k']", "x = 1"])
def test_code_changing_no_global_adds_nothing(namespace, statement):
    assert "LOG" not in _changed_globals(statement, namespace)


def test_a_call_into_a_closure_over_a_writer_is_not_served(namespace):
    """The call cache asks the same of the callee: the wrapper calls the
    function it closes over, which appends."""
    assert hands_a_state_changing_callable(namespace["save_row"], ("Z",), {})
    assert hands_a_state_changing_callable(namespace["note"], ("Z",), {})
    assert not hands_a_state_changing_callable(functools.partial(len), ("Z",), {})
