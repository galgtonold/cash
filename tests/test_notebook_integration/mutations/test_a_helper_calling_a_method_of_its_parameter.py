"""A helper that hands an argument, or a global, to a method of its own
parameter: ``def shuf(r, xs): r.shuffle(xs)``, ``def mix(r):
r.shuffle(DECK)``, ``def run(m, xs): m.apply(xs)``.

``r`` is bound only when the helper runs, and was taken for a name that
fails before the call runs, so what the call is handed counted as only
read. A cached ``mix(random.Random(SEED))`` was served with ``DECK`` left
unshuffled after a restart, and the old order after ``SEED`` was edited.
What the caller hands such a helper directly is watched as every argument
of a function of the user's is; these pin it.

Each step (Run All, an edit, a restart, the edit undone) is compared with
plain Python running the same cells.
"""

import sys
import types

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600), pytest.mark.fresh_kernel]

FLOORS = "import cash\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)"

HELPERS = """
import random, time
def mix(r):
    time.sleep(0.2)
    r.shuffle(DECK)
def shuf(r, xs):
    time.sleep(0.2)
    r.shuffle(xs)
class Scaler:
    def __init__(self, k):
        self.k = k
    def apply(self, xs):
        xs[:] = [v * self.k for v in xs]
def run(m, xs):
    time.sleep(0.2)
    m.apply(xs)
def run_global(m):
    time.sleep(0.2)
    m.apply(DECK)
"""

RNG = ("SEED = 1", "SEED = 2", "random.Random(SEED)")
SCALER = ("K = 2", "K = 3", "Scaler(K)")

#: name -> (where the helper is, its setup, the statement calling it)
CASES = {
    "global-statement": ("notebook", RNG, "mix(OBJ)"),
    "global-assigned": ("notebook", RNG, "n = mix(OBJ)"),
    "global-loop": ("notebook", RNG, "for _ in range(2):\n    mix(OBJ)"),
    "global-branch": ("notebook", RNG, "if True:\n    mix(OBJ)"),
    "global-comprehension": ("notebook", RNG, "n = [mix(OBJ) for _ in range(2)]"),
    "global-users-method": ("notebook", SCALER, "n = run_global(OBJ)"),
    "argument-assigned": ("notebook", RNG, "n = shuf(OBJ, data)"),
    "argument-loop": ("notebook", RNG, "for _ in range(2):\n    n = shuf(OBJ, data)"),
    "argument-users-method": ("notebook", SCALER, "n = run(OBJ, data)"),
    "module-argument-assigned": ("module", RNG, "n = shuf(OBJ, data)"),
    "module-argument-users-method": ("module", SCALER, "n = run(OBJ, data)"),
}


def _cells(case: str, edited: bool) -> list[str]:
    where, (first, other, make), statement = CASES[case]
    head = HELPERS if where == "notebook" else "import random\nfrom pm_helpers import *"
    return [
        FLOORS,
        head,
        other if edited else first,
        "data = list(range(8))\nDECK = data",
        statement.replace("OBJ", make),
        "top = data[:4]\nprint(top)",
    ]


def _plain(cells: list[str]) -> str:
    module = types.ModuleType("pm_helpers")
    exec(HELPERS, module.__dict__)
    sys.modules["pm_helpers"] = module
    try:
        namespace: dict = {"__name__": "__main__"}
        for code in cells[1:]:
            exec(code, namespace)
        return repr(namespace["top"])
    finally:
        sys.modules.pop("pm_helpers", None)


@pytest.mark.parametrize("case", list(CASES))
def test_every_run_matches_plain_python(nb_runner, case):
    plan = [("run", False), ("run", False), ("edit", True), ("restart", True), ("edit", False), ("restart", False)]
    nb_runner.create_notebook(_cells(case, False))
    (nb_runner.work_dir / "pm_helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.start_kernel()
    got, want = [], []
    for step, edited in plan:
        cells = _cells(case, edited)
        if step == "edit":
            nb_runner.set_cell_source(3, cells[2])
        if step == "restart":
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.run_all()
        got.append(nb_runner.peek("top").strip())
        want.append(_plain(cells))
    assert got == want
