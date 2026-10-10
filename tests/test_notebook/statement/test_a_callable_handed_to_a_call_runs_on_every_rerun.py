"""A callable handed to a call runs again when the notebook runs again.

``s.apply(f)`` runs ``f`` as much as ``f(v)`` does. Handed as a dict entry
(``ops['dbl']``), a bound method (``tracker.record``), an object whose class
defines ``__call__`` (``counter``), a closure (``memo``) or in a list a helper
runs (``steps``), the statement was served from the cache on the next Run
All: the callable never ran and what it fills stayed empty, where a plain
kernel fills it again. The bare name (``s.apply(f)``) was already right.
"""

from __future__ import annotations

import ast
import warnings

import pytest

from tests._cell_driver import run_cash_cell

#: Each element sleeps so the statement is worth caching.
SLEEP = 0.07

SETUP = f"import time, pandas as pd\ns = pd.Series(range(4))\nSLEEP = {SLEEP}"

CASES = {
    "dict-entry": (
        "seen = []\ndef f(v):\n    time.sleep(SLEEP); seen.append(v); return v * 2\nops = {'dbl': f}",
        "r = s.apply(ops['dbl'])",
        "len(seen)",
    ),
    "bound-method": (
        "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
        "    def record(self, v):\n        time.sleep(SLEEP); self.seen.append(v); return v * 2\n"
        "tracker = Tracker()",
        "r = s.apply(tracker.record)",
        "len(tracker.seen)",
    ),
    "method-through-another-method": (
        "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
        "    def record(self, v):\n        time.sleep(SLEEP); self._note(v); return v * 2\n"
        "    def _note(self, v):\n        self.seen.append(v)\n"
        "tracker = Tracker()",
        "r = s.apply(tracker.record)",
        "len(tracker.seen)",
    ),
    "callable-instance": (
        "class Counter:\n    def __init__(self):\n        self.n = 0\n"
        "    def __call__(self, v):\n        time.sleep(SLEEP); self.n += 1; return v * 2\n"
        "counter = Counter()",
        "r = s.apply(counter)",
        "counter.n",
    ),
    "closure": (
        "def make_memo():\n    calls = []\n    def g(v):\n        time.sleep(SLEEP); calls.append(v); return v * 2\n"
        "    g.calls = calls\n    return g\nmemo = make_memo()",
        "r = s.apply(memo)",
        "len(memo.calls)",
    ),
    "list-a-helper-runs": (
        "seen = []\ndef f(v):\n    time.sleep(SLEEP); seen.append(v); return v\n"
        "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\nsteps = [f]",
        "r = run_steps(steps, list(range(4)))",
        "len(seen)",
    ),
    "list-of-bound-methods": (
        "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
        "    def record(self, v):\n        time.sleep(SLEEP); self.seen.append(v); return v\n"
        "tracker = Tracker()\n"
        "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\nsteps = [tracker.record]",
        "r = run_steps(steps, list(range(4)))",
        "len(tracker.seen)",
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_second_run_all_calls_it_again(cash_magics, mock_shell, case):
    definitions, statement, check = CASES[case]
    cells = [SETUP, definitions, statement]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    assert eval(check, mock_shell.user_ns) == 4


def test_a_method_that_changes_nothing_is_still_served(cash_magics, mock_shell, tmp_path):
    """Control: a handed method that leaves its object alone keeps the cache.
    Its runs are counted in a file, which a hit does not replay."""
    log = tmp_path / "runs"
    definitions = (
        "import os\nclass Model:\n    def __init__(self, k):\n        self.k = k\n"
        "    def predict(self, v):\n        time.sleep(SLEEP)\n"
        f"        fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "        os.write(fd, b'x'); os.close(fd)\n        return v * self.k\n"
        "model = Model(3)"
    )
    cells = [SETUP, definitions, "r = s.apply(model.predict)"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    assert list(mock_shell.user_ns["r"]) == [0, 3, 6, 9]
    assert log.read_bytes() == b"xxxx", "the second run should have been served from the cache"


def _namespace(code: str) -> dict:
    """*code* run as a cell would be: its functions keep their source."""
    import linecache

    filename = f"<handed-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)  # noqa: S102 - the test's own cell text
    return namespace


@pytest.mark.parametrize(
    ("statement", "receivers"),
    [
        ("r = s.apply(tracker.record)", {"tracker"}),
        ("r = s.apply(tracker.peek)", set()),
        ("r = run_steps([tracker.record], data)", {"tracker"}),
        ("r = take(big, tracker.peek)", set()),
        ("r = list(map(seen.append, data))", {"seen"}),
    ],
)
def test_the_objects_a_handed_callable_changes(statement, receivers):
    from cash.analysis.handed_callables import handed_callables

    ns = _namespace(
        "seen = []\nbig = list(range(10))\n"
        "class Tracker:\n    def __init__(self):\n        self.seen = []\n"
        "    def record(self, v):\n        self.seen.append(v)\n"
        "    def peek(self, v):\n        return len(self.seen)\n"
        "tracker = Tracker()\n"
        "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\n"
        "def take(rows, fn):\n    return [fn(r) for r in rows]\n"
    )
    assert set(handed_callables(ast.parse(statement), ns).receivers) == receivers


def test_a_handed_user_function_without_source_runs_uncached():
    from cash.analysis.handed_callables import handed_callables

    ns: dict = {"__name__": "__main__"}
    exec("def f(v):\n    return v\n", ns)  # noqa: S102 - no source on purpose
    found = handed_callables(ast.parse("r = s.apply(f)"), ns)
    assert found.unreadable and "'f'" in found.unreadable[0]


def test_a_call_handed_a_list_of_writers_is_not_served():
    from cash.analysis.handed_callables import hands_a_state_changing_callable

    ns = _namespace(
        "seen = []\ndef f(v):\n    seen.append(v)\ndef g(v):\n    return v\n"
        "def run_steps(steps, data):\n    return [st(x) for x in data for st in steps]\n"
        "def total(rows):\n    return sum(rows)\n"
    )
    run_steps, total = ns["run_steps"], ns["total"]
    assert hands_a_state_changing_callable(run_steps, ([ns["f"]], [1]), {})
    assert not hands_a_state_changing_callable(run_steps, ([ns["g"]], [1]), {})
    # A list the callee does not call into is data, and not looked into.
    assert not hands_a_state_changing_callable(total, ([ns["f"]],), {})

