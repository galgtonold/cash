"""A function of the user's that a statement reaches without naming it runs
again on every Run All, as in a plain kernel: picked out of a dict or list
and called, called for each item a comprehension iterates, wrapped in a
``functools.partial`` with another function, a method a helper calls on
what it is given, a lambda's default, the items a ``reduce`` calls, a
namedtuple of functions, a callback an object holds (also fired from a
helper), and a logging handler of the user's.

The statements, or the calls inside them, were served from the cache on
the second Run All: the functions never ran and the lists they append to
stayed empty.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = """import time, functools, collections, logging
LOGS = collections.defaultdict(list)
def record(v, case):
    time.sleep(0.03)
    LOGS[case].append(v)
    return v * 2
def double(v, case):
    return v * 2
class Step:
    def __init__(self, fn):
        self.fn = fn
class Model:
    def predict(self, v, case):
        time.sleep(0.03)
        LOGS[case].append(v)
        return v + 1
class Bus:
    def __init__(self):
        self.subs = []
    def subscribe(self, cb):
        self.subs.append(cb)
    def emit(self, x, case):
        return [cb(x, case) for cb in self.subs]
class ListHandler(logging.Handler):
    def emit(self, rec):
        LOGS['logging'].append(rec.getMessage())
def run_model(model, v, case):
    time.sleep(0.03)
    return model.predict(v, case)
def apply_(f, v, case):
    return f(v, case)
def run_fn(f, v, case):
    time.sleep(0.03)
    return f(v, case)
def run_iter(steps, case):
    time.sleep(0.03)
    return [s(3, case) for s in steps]
def fire(x):
    return bus.emit(x, 'helper')
def compute(x):
    time.sleep(0.03)
    lg.info('computing %s', x)
    return x * 2
Pipe = collections.namedtuple('Pipe', 'pre post')"""

HOLDERS = """table = {'d': double, 'r': record}
key = 'r'
fns = [double, record]
steps = [Step(double), Step(record)]
p = functools.partial(apply_, record)
m = Model()
nt = Pipe(double, record)
bus = Bus()
bus.subscribe(record)
lg = logging.getLogger('cash-reached-indirectly')
lg.setLevel(logging.INFO)
lg.propagate = False
lg.handlers.clear()
lg.addHandler(ListHandler())"""

STATEMENTS = {
    "variable-key": "a = table[key](3, 'variable-key')",
    "comprehension": "b = [f(3, 'comprehension') for f in fns]",
    "objects": "c = [s.fn(3, 'objects') for s in steps]",
    "partial": "d = run_fn(p, 3, 'partial')",
    "method": "e = run_model(m, 3, 'method')",
    "default": "g = list(map(lambda v, h=record: h(v, 'default'), [3]))",
    "reduce": "r = functools.reduce(lambda acc, f: f(acc, 'reduce'), fns, 3)",
    "namedtuple": "n = run_iter(nt, 'namedtuple')",
    "callback": "o = bus.emit(3, 'callback')",
    "helper": "q = fire(3)",
    "logging": "t = compute(3)",
}
STATE = "sorted((k, len(v)) for k, v in LOGS.items())"
WANT = repr(sorted((case, 1) for case in STATEMENTS))


def test_every_function_reached_runs_again_on_a_run_all_and_after_a_restart(nb_runner):
    nb_runner.create_notebook([SETUP, HOLDERS, *STATEMENTS.values()])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT


def test_a_comprehension_over_plain_values_stays_cached(nb_runner):
    """Control: a comprehension calling methods of plain values and a
    function by name that changes nothing is served from the cache."""
    cells = [
        "%cash_badge print",
        "import time\ndef slow(x):\n    time.sleep(0.03)\n    return x.strip()\nlines = [' a', 'b ', ' c ']",
        "out = [slow(x) for x in lines] + [x.upper() for x in lines]",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_raw_output(3)


STATEFUL = {
    "function-attribute": (
        [
            "LOG = []",
            "def fattr(v):\n    fattr.calls += 1\n    LOG.append(v)\n    return v\nfattr.calls = 0",
            "y = list(map(fattr, [1, 2]))",
            "z = fattr(5)",
        ],
        "(fattr.calls, LOG)",
        (3, [1, 2, 5]),
    ),
    "default-argument": (
        [
            "def tag(s, _seen=[]):\n    _seen.append(s)\n    return f'{s}{len(_seen)}'",
            "a = list(map(tag, ['x', 'y']))",
            "b = tag('z')",
        ],
        "(a, b)",
        (["x1", "y2"], "z3"),
    ),
}


@pytest.mark.parametrize("case", list(STATEFUL))
def test_a_function_keeping_state_on_itself_handed_to_map_runs_once_per_run_all(nb_runner, case):
    """A function that keeps state on itself (a counter attribute, a list in
    its default) handed to ``map``: the second Run All re-ran the cells above
    the call by name in the middle of the run, so the state was reset or the
    function was called twice."""
    cells, check, want = STATEFUL[case]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert nb_runner.peek(check) == repr(want)
