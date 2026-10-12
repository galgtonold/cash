"""A function of the user's that a statement reaches without naming it runs
again when the notebook runs again.

Picked out of a list or dict and called (``table[key](3)``, ``fns[i](3)``,
``table.get(key)(3)``, ``[f(3) for f in fns]``), wrapped in a
``functools.partial`` with another function, a method or stored function a
helper calls on what it is given (``model.predict(v)``, ``s.fn(v)``), a
lambda's default (``lambda v, g=record: g(v)``), the items a ``map`` or
``reduce`` function calls (``map(lambda f: f(3), fns)``), handed in a
``**kwargs``, a namedtuple or a subclass of ``dict`` or ``list``, a
callback an object holds (``bus.emit(3)``, also from a helper) and a
logging handler of the user's: the statement, or the call inside it, was
served from the cache on the next Run All, and what the function appends
to was left empty, where a plain kernel fills it again.
"""

from __future__ import annotations

import ast
import linecache
import warnings

import pytest

from cash.analysis.handed_callables import handed_callables, hands_a_state_changing_callable
from cash.analysis.namespace_effects import statement_calls_user_writer
from tests._cell_driver import run_cash_cell

DEFINITIONS = """
import time, functools, collections, logging
LOG = []
def record(v):
    LOG.append(v)
    return v * 2
def double(v):
    return v * 2
class Step:
    def __init__(self, fn):
        self.fn = fn
class Model:
    def predict(self, v):
        LOG.append(v)
        return v + 1
class Bus:
    def __init__(self):
        self.subs = []
    def subscribe(self, cb):
        self.subs.append(cb)
    def emit(self, x):
        return [cb(x) for cb in self.subs]
class ListHandler(logging.Handler):
    def emit(self, rec):
        LOG.append(rec.getMessage())
def run_steps(steps, v):
    return [s.fn(v) for s in steps]
def run_model(model, v):
    return model.predict(v)
def apply_(f, v):
    return f(v)
def run_fn(f, v):
    return f(v)
def call3(f):
    return f(3)
def run_kw(**hooks):
    return [h(3) for h in hooks.values()]
def run_iter(steps):
    return [s(3) for s in steps]
def run_vals(d):
    return [s(3) for s in d.values()]
def fire(x):
    return bus.emit(x)
def compute(x):
    lg.info('computing %s', x)
    return x * 2
Pipe = collections.namedtuple('Pipe', 'pre post')
class Registry(dict):
    pass
class Steps(list):
    pass
fns = [double, record]
table = {'d': double, 'r': record}
steps = [Step(double), Step(record)]
key = 'r'
i = 1
p = functools.partial(apply_, record)
m = Model()
hooks = {'a': double, 'b': record}
nt = Pipe(double, record)
reg = Registry(a=double, b=record)
ls = Steps([double, record])
bus = Bus()
bus.subscribe(record)
lg = logging.getLogger('cash-test-reached-indirectly')
lg.setLevel(logging.INFO)
lg.propagate = False
lg.handlers.clear()
lg.addHandler(ListHandler())
lines = ['a ', ' b']
nums = [1, 2, 3]
"""

STATEMENTS = {
    "variable-key": "out = table[key](3)",
    "get-a-variable-key": "out = table.get(key)(3)",
    "variable-index": "out = fns[i](3)",
    "comprehension": "out = [f(3) for f in fns]",
    "comprehension-over-items": "out = {n: f(3) for n, f in table.items()}",
    "comprehension-over-a-tuple": "out = [g(3) for g in (double, record)]",
    "comprehension-over-objects": "out = [s.fn(3) for s in steps]",
    "loop-over-a-dict": "out = {}\nfor name in table:\n    out[name] = table[name](3)",
    "partial-with-another-function": "y = run_fn(p, 3)",
    "stored-function-a-helper-calls": "y = run_steps(steps, 3)",
    "method-a-helper-calls": "y = run_model(m, 3)",
    "lambda-default": "out = list(map(lambda v, g=record: g(v), [3]))",
    "reduce-over-functions": "out = functools.reduce(lambda acc, f: f(acc), fns, 3)",
    "map-calling-its-items": "out = list(map(lambda f: f(3), fns))",
    "map-a-helper-calling-its-items": "out = list(map(call3, fns))",
    "sorted-key-calling-its-items": "out = len(sorted(fns, key=lambda f: f(0)))",
    "callback-an-object-holds": "out = bus.emit(3)",
    "callback-from-a-helper": "out = fire(3)",
    "logging-handler": "out = compute(3)",
    "double-star": "out = run_kw(**hooks)",
    "namedtuple": "out = run_iter(nt)",
    "dict-subclass": "out = run_vals(reg)",
    "list-subclass": "out = run_iter(ls)",
}


def _namespace(code: str) -> dict:
    """*code* run as a cell would be: its functions keep their source."""
    filename = f"<reached-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def namespace():
    return _namespace(DEFINITIONS)


@pytest.mark.parametrize("case", list(STATEMENTS))
def test_the_statement_reaches_the_function(namespace, case):
    found = handed_callables(ast.parse(STATEMENTS[case]), namespace)
    # ``emit`` is the logging handler's method.
    assert {f.__name__ for f in found.functions} & {"record", "predict", "emit"}, found
    assert found.unreadable == ()


@pytest.mark.parametrize(
    "statement",
    [
        "out = [x.strip() for x in lines]",
        "out = [double(x) for x in nums]",
        "out = [n * 2 for n in nums]",
        "out = getattr(functools, 'reduce')",
        "out = list(map(lambda x: x.strip(), lines))",
        "out = table['d'](3)",
    ],
)
def test_what_reaches_no_function_of_theirs_finds_none(namespace, statement):
    """Controls: methods of plain values, functions called by name, a
    constant key that picks a function changing nothing."""
    found = handed_callables(ast.parse(statement), namespace)
    assert {f.__name__ for f in found.functions} <= {"double"}
    assert found.unreadable == ()


def test_a_function_whose_item_cash_cannot_tell_is_unreadable():
    """What a function of the user's returns is known only by running it:
    the statement runs uncached."""
    found = handed_callables(ast.parse("out = [f(3) for f in make()]"), _namespace("def make():\n    return []\n"))
    assert found.unreadable and "cannot tell which function it is" in found.unreadable[0]


@pytest.mark.parametrize(
    ("callee", "args", "kwargs"),
    [
        ("run_fn", ("p", 3), {}),
        ("run_steps", ("steps", 3), {}),
        ("run_model", ("m", 3), {}),
        ("fire", (3,), {}),
        ("compute", (3,), {}),
        ("run_iter", ("nt",), {}),
    ],
)
def test_a_call_reaching_it_is_not_served(namespace, callee, args, kwargs):
    """The call cache's question: the call runs every time."""

    def value(v):
        return namespace[v] if isinstance(v, str) else v

    fn = namespace[callee]
    assert hands_a_state_changing_callable(fn, tuple(map(value, args)), {k: value(v) for k, v in kwargs.items()})


def test_a_call_reaching_nothing_that_changes_state_is_served(namespace):
    """Controls: the same helpers handed functions that change nothing."""
    ns = namespace
    assert not hands_a_state_changing_callable(ns["run_fn"], (ns["double"], 3), {})
    assert not hands_a_state_changing_callable(ns["run_steps"], ([ns["Step"](ns["double"])], 3), {})
    assert not hands_a_state_changing_callable(ns["call3"], (ns["double"],), {})


@pytest.mark.parametrize(
    "statement", ["_ = [f() for f in [export]]", "_ = bus.emit()", "_ = run_steps(steps)", "export()"]
)
def test_a_writer_reached_without_its_name_is_a_writer(statement):
    """The upstream check re-runs an edited writer's statement for a cell
    that reads its file: it has to know the statement writes."""
    ns = _namespace(
        "def export():\n    with open('out.csv', 'w') as fh:\n        fh.write('x')\n"
        "class Bus:\n    def __init__(self):\n        self.subs = []\n"
        "    def emit(self):\n        return [c() for c in self.subs]\n"
        "class Step:\n    def __init__(self, fn):\n        self.fn = fn\n"
        "def run_steps(steps):\n    return [s.fn() for s in steps]\n"
        "bus = Bus()\nbus.subs.append(export)\nsteps = [Step(export)]\n"
    )
    assert statement_calls_user_writer(statement, ns, appends=False) == "export"


SLEEP = 0.07
RERUN_CASES = {
    "variable-key": ("table = {'r': record}\nkey = 'r'", "out = table[key](3)"),
    "comprehension": ("fns = [record]", "out = [f(3) for f in fns]"),
    "comprehension-over-objects": ("steps = [Step(record)]", "out = [s.fn(3) for s in steps]"),
    "partial-with-another-function": ("p = functools.partial(apply_, record)", "y = run_fn(p, 3)"),
    "method-a-helper-calls": ("m = Model()", "y = run_model(m, 3)"),
    "namedtuple": ("nt = Pipe(record, record)", "out = run_iter(nt)"),
}
RERUN_DEFINITIONS = f"""import time, functools, collections
LOG = []
def record(v):
    time.sleep({SLEEP})
    LOG.append(v)
    return v * 2
class Step:
    def __init__(self, fn):
        self.fn = fn
class Model:
    def predict(self, v):
        time.sleep({SLEEP})
        LOG.append(v)
        return v + 1
class Bus:
    def __init__(self):
        self.subs = []
    def subscribe(self, cb):
        self.subs.append(cb)
    def emit(self, x):
        return [cb(x) for cb in self.subs]
def run_model(model, v):
    return model.predict(v)
def apply_(f, v):
    return f(v)
def run_fn(f, v):
    return f(v)
def run_iter(steps):
    return [s(3) for s in steps]
Pipe = collections.namedtuple('Pipe', 'pre post')"""


@pytest.mark.parametrize("case", list(RERUN_CASES))
def test_a_second_run_all_calls_it_again(cash_magics, mock_shell, case):
    holders, statement = RERUN_CASES[case]
    cells = [RERUN_DEFINITIONS, holders, statement]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = []
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
            counts.append(len(mock_shell.user_ns["LOG"]))
    assert counts[1] == counts[0] > 0
