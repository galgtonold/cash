"""A callable picked by an expression and handed to a call runs again on
every Run All and after a restart, as in a plain kernel: ``ops.get('dbl')``,
``ops[key]`` with a variable key, ``f if flag else g``, ``getattr(tracker,
'record')``, and functions kept in an object of the user's class or in a
``collections.deque`` that a helper of theirs runs.

The statements were served from the cache and the callables never ran: the
lists they fill stayed empty where plain Jupyter fills them again.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ROWS = 20
CELLS = [
    f"import time, collections, pandas as pd, numpy as np\ns = pd.Series(np.arange({ROWS}))",
    "seen1 = []\ndef f1(v):\n    time.sleep(0.015); seen1.append(v); return v * 2\nops = {'dbl': f1}",
    "r1 = s.apply(ops.get('dbl'))",
    "seen2 = []\ndef f2(v):\n    time.sleep(0.015); seen2.append(v); return v * 2\nops2 = {'dbl': f2}\nkey = 'dbl'",
    "r2 = s.apply(ops2[key])",
    "seen3 = []\ndef f3(v):\n    time.sleep(0.015); seen3.append(v); return v * 2\nflag = True",
    "r3 = s.apply(f3 if flag else abs)",
    "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
    "    def record(self, v):\n        time.sleep(0.015); self.seen.append(v); return v * 2\ntracker = Tracker()",
    "r4 = s.apply(getattr(tracker, 'record'))",
    "seen5 = []\ndef f5(v):\n    time.sleep(0.015); seen5.append(v); return v\n"
    "class Pipeline:\n    def __init__(self, steps):\n        self.steps = steps\n"
    "def run(pipe, data):\n    return [st(x) for x in data for st in pipe.steps]\npipe = Pipeline([f5])",
    f"r5 = run(pipe, list(range({ROWS})))",
    "seen6 = []\ndef f6(v):\n    time.sleep(0.015); seen6.append(v); return v\n"
    "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\n"
    "steps6 = collections.deque([f6])",
    f"r6 = run_steps(steps6, list(range({ROWS})))",
    "print(len(seen1), len(seen2), len(seen3), len(tracker.seen), len(seen5), len(seen6))",
]
#: Read from the kernel: a cached print replays the stdout of the run that
#: stored it.
STATE = "(len(seen1), len(seen2), len(seen3), len(tracker.seen), len(seen5), len(seen6))"
WANT = repr((ROWS,) * 6)


def test_every_picked_callable_runs_again_after_a_run_all_and_a_restart(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek(STATE) == WANT
