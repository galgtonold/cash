"""Code of the user's that a statement runs without calling it by a bare
name runs again on every Run All, as in a plain kernel: a method called on
an object or a class (also a static or class method, and a hook reached
through ``super()``); a context manager's ``__exit__``, a property, a
special method (``bag['k']``, ``len(bag)``, ``bag + 1``, ``f'{bag}'``,
iterating, ``in``); a function a wrapper made without ``functools.wraps``
or a ``compose`` closure calls; a ``functools.partial``, ``np.vectorize``
or ``operator.methodcaller`` called by its own name.

Each was served from the cache on the second Run All when it was slow, so
the global list it appends to stayed empty. And on the first Run All, a
cell appending to the list directly re-ran the cell making the list in the
middle of the run, wiping what the cells above had added.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = """import time, collections
LOGS = collections.defaultdict(list)
def write_row(case, r):
    LOGS[case].append(r)
    return len(r)
class Store:
    def process(self, r):
        time.sleep(0.03)
        LOGS['method'].append(r)
        return len(r)
    def via_helper(self, r):
        time.sleep(0.03)
        return write_row('helper', r)
    @staticmethod
    def static(r):
        time.sleep(0.03)
        return write_row('static', r)
    @classmethod
    def klass(cls, r):
        time.sleep(0.03)
        return write_row('classmethod', r)
    def pure(self, r):
        time.sleep(0.03)
        return len(r)
class Base:
    def run(self, r):
        time.sleep(0.03)
        return self.hook(r)
    def hook(self, r):
        return 0
class Job(Base):
    def run(self, r):
        return super().run(r) + 1
    def hook(self, r):
        LOGS['super'].append(r)
        return 1
store = Store()
job = Job()
import functools, operator
import numpy as np
from functools import singledispatch
def slow(n):
    time.sleep(0.03)
    return n
class timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        return self
    def __exit__(self, *a):
        LOGS['with'].append(self.name)
        return False
class Bag:
    def __getitem__(self, k):
        LOGS['get'].append(k)
        return 2
    def __len__(self):
        LOGS['len'].append(1)
        return 1
    def __iter__(self):
        LOGS['iter'].append(1)
        return iter([1, 2])
    def __add__(self, o):
        LOGS['add'].append(o)
        return 5
    def __contains__(self, k):
        LOGS['in'].append(k)
        return True
    def __format__(self, spec):
        LOGS['fmt'].append(spec)
        return 'B'
    @property
    def value(self):
        LOGS['prop'].append(1)
        return 7
bag = Bag()
def record(case, v):
    time.sleep(0.03)
    LOGS[case].append(v)
    return 1
def with_retry(fn):
    def call(*a):
        return fn(*a)
    return call
def compose(f, g):
    return lambda x: f(g(x))
@with_retry
def save_row(v):
    time.sleep(0.03)
    LOGS['decorated'].append(v)
    return v
@singledispatch
def note(x):
    return x
@note.register
def _(x: str):
    time.sleep(0.03)
    LOGS['singledispatch'].append(x)
    return x
class H:
    def rec(self, v):
        time.sleep(0.03)
        LOGS['methodcaller'].append(v)
        return 1
h = H()
safe = with_retry(functools.partial(record, 'wrapper'))
pipe = compose(functools.partial(record, 'compose'), str.strip)
pz = functools.partial(record, 'partial', 'Z')
vz = np.vectorize(functools.partial(record, 'vectorize'), otypes=[int])
mc = operator.methodcaller('rec', 'Z')"""

STATEMENTS = {
    "method": "a = store.process('x')",
    "helper": "b = store.via_helper('x')",
    "static": "c = Store.static('x')",
    "classmethod": "d = Store.klass('x')",
    "super": "e = job.run('x')",
    "with": "with timer('fit'):\n    f = slow(1)",
    "get": "g1 = slow(bag['k'])",
    "len": "g2 = slow(len(bag))",
    "iter": "g3 = slow(sum(bag))",
    "add": "g4 = slow(bag + 1)",
    "in": "g5 = slow('k' in bag)",
    "fmt": "g6 = slow(f'{bag}')",
    "prop": "g7 = slow(bag.value)",
    "decorated": "h1 = save_row('Z')",
    "wrapper": "h2 = safe('Z')",
    "compose": "h3 = pipe(' Z ')",
    "partial": "h4 = pz()",
    "vectorize": "h5 = vz(['Z'])",
    "methodcaller": "h6 = mc(h)",
    "singledispatch": "h7 = note('Z')",
}
STATE = "sorted((k, len(v)) for k, v in LOGS.items())"
WANT = repr(sorted((case, 1) for case in STATEMENTS))


def test_every_method_runs_again_on_a_run_all(nb_runner):
    nb_runner.create_notebook([SETUP, *STATEMENTS.values()])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT


WIPE_SETUP = """import time, operator
from functools import singledispatch
TIMINGS = []
class timer:
    def __init__(self, name): self.name = name
    def __enter__(self): return self
    def __exit__(self, *a): TIMINGS.append(self.name); return False
class Memo:
    def __getitem__(self, k):
        TIMINGS.append(k)
        return k * 2
memo = Memo()
@singledispatch
def note(x): return x
@note.register
def _(x: str): TIMINGS.append(x); return x
class Rec:
    def add(self, v): TIMINGS.append(v); return v
rec = Rec()
def slow(n):
    time.sleep(0.05); return n"""


@pytest.mark.parametrize(
    "cell, want",
    [
        ("with timer('load'):\n    x = slow(1)", ["load", "total"]),
        ("a = memo['load']", ["load", "total"]),
        ("b = note('load')", ["load", "total"]),
        ("c = operator.methodcaller('add', 'load')(rec)", ["load", "total"]),
    ],
)
def test_a_cell_appending_below_keeps_what_the_cell_above_added(nb_runner, cell, want):
    """First Run All: the cell above appends through code cash did not see
    as a call; the cell appending directly below re-ran ``TIMINGS = []`` in
    the middle of the run."""
    nb_runner.create_notebook([WIPE_SETUP, cell, "TIMINGS.append('total')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("TIMINGS") == repr(want)


def test_a_method_changing_nothing_stays_cached(nb_runner):
    """Control: a slow method that changes no global is served."""
    nb_runner.create_notebook(["%cash_badge print", SETUP, "p = store.pure('xyz')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_raw_output(3)
