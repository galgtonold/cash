"""A callable handed to a call runs again on every Run All, as in a plain
kernel: a dict entry (``ops['dbl']``), a bound method
(``tracker.record``), an object with ``__call__``, a closure, a list of steps
a helper runs, and a function of the user's module that keeps state in the
module (``s.apply(helpers.record)``, ``s.map(count)``), after a restart
too.

The statements were served from the cache and the callables never ran: the
lists and counters they fill stayed empty (``60 0 0 0 0 0`` where plain
Jupyter prints ``60 60 60 60 60 60``).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ROWS = 20
CELLS = [
    f"import time, pandas as pd, numpy as np\ns = pd.Series(np.arange({ROWS}))",
    "seen1 = []\ndef f1(v):\n    time.sleep(0.015); seen1.append(v); return v * 2",
    "r1 = s.apply(f1)",
    "seen2 = []\ndef f2(v):\n    time.sleep(0.015); seen2.append(v); return v * 2\nops = {'dbl': f2}",
    "r2 = s.apply(ops['dbl'])",
    "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
    "    def record(self, v):\n        time.sleep(0.015); self.seen.append(v); return v * 2\ntracker = Tracker()",
    "r3 = s.apply(tracker.record)",
    "class Counter:\n    def __init__(self):\n        self.n = 0\n"
    "    def __call__(self, v):\n        time.sleep(0.015); self.n += 1; return v * 2\ncounter = Counter()",
    "r4 = s.apply(counter)",
    "def make_memo():\n    calls = []\n    def g(v):\n        time.sleep(0.015); calls.append(v); return v * 2\n"
    "    g.calls = calls\n    return g\nmemo = make_memo()",
    "r5 = s.apply(memo)",
    "seen6 = []\ndef f6(v):\n    time.sleep(0.015); seen6.append(v); return v\n"
    "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\nsteps = [f6]",
    f"r6 = run_steps(steps, list(range({ROWS})))",
    "print(len(seen1), len(seen2), len(tracker.seen), counter.n, len(memo.calls), len(seen6))",
]
EXPECTED = " ".join([str(ROWS)] * 6)
#: What the last cell prints, read from the kernel: a cached print replays
#: the stdout of the run that stored it.
STATE = "(len(seen1), len(seen2), len(tracker.seen), counter.n, len(memo.calls), len(seen6))"
STATE_EXPECTED = repr((ROWS,) * 6)

HELPERS = """import time
SEEN = []
STATS = {"n": 0}
def record(v):
    time.sleep(0.015)
    SEEN.append(v)
    return v * 2
def count(v):
    time.sleep(0.015)
    STATS["n"] += 1
    return v
"""
MODULE_CELLS = [
    f"import pandas as pd, numpy as np\nimport handedhelpers\nfrom handedhelpers import count\ns = pd.Series(np.arange({ROWS}))",
    "handedhelpers.SEEN.clear()\nhandedhelpers.STATS['n'] = 0",
    "r = s.apply(handedhelpers.record)",
    "r2 = s.map(count)",
    "print(len(handedhelpers.SEEN), handedhelpers.STATS['n'])",
]


def test_a_second_run_all_runs_every_handed_callable(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(len(CELLS)).strip() == EXPECTED
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == STATE_EXPECTED
    assert nb_runner.get_output(len(CELLS)).strip() == EXPECTED


@pytest.mark.fresh_kernel
def test_a_module_function_handed_to_a_call_sets_the_module_state_again(nb_runner, tmp_path):
    (tmp_path / "handedhelpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(MODULE_CELLS)
    nb_runner.start_kernel()
    state = "(len(handedhelpers.SEEN), handedhelpers.STATS['n'])"
    for _ in range(3):
        nb_runner.run_all()
        assert nb_runner.peek(state) == repr((ROWS, ROWS))
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek(state) == repr((ROWS, ROWS))
