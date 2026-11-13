"""A statement that shuffles a global in place through a path the effect
analysis did not follow: a method of the user's class
(``Mixer().go(SEED)``), a lambda the helper calls at once
(``(lambda: rng.shuffle(DECK))()``), or a list kept in a module of the
user's (``hhm.MODDECK``).

The statement was taken for one that changes nothing, so after a restart it
was served from the cache and the global kept its first order:
``[0, 1, 2, 3]`` where plain Python has ``[3, 6, 1, 5]``. Each step (Run
All, again, an edit, a restart, the edit undone, a restart) is compared with
plain Python running the same cells, the module kept from one Run All to the
next as a kernel keeps it.
"""

import sys
import types

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600), pytest.mark.fresh_kernel]

FLOORS = "import cash\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)"

HHM = """
import random, time
MODDECK = list(range(8))
def mix(seed):
    time.sleep(0.2)
    random.Random(seed).shuffle(MODDECK)
def _inner(seed):
    random.Random(seed).shuffle(MODDECK)
def mix_inner(seed):
    time.sleep(0.2)
    _inner(seed)
class ModMixer:
    def go(self, seed):
        time.sleep(0.2)
        random.Random(seed).shuffle(MODDECK)
"""

HELPERS = """
import random, time
import hhm
class Mixer:
    def go(self, seed):
        time.sleep(0.2)
        random.Random(seed).shuffle(DECK)
class Holder:
    def __init__(self, k):
        self.k = k
    def go(self, seed):
        time.sleep(0.2)
        random.Random(seed).shuffle(self.k)
def mix(seed):
    time.sleep(0.2)
    (lambda: random.Random(seed).shuffle(DECK))()
def mix_module(seed):
    time.sleep(0.2)
    random.Random(seed).shuffle(hhm.MODDECK)
def mix_module_lambda(seed):
    time.sleep(0.2)
    (lambda: random.Random(seed).shuffle(hhm.MODDECK))()
class ModuleMixer:
    def go(self, seed):
        time.sleep(0.2)
        random.Random(seed).shuffle(hhm.MODDECK)
def mix_through_method(seed):
    time.sleep(0.2)
    Mixer().go(seed)
"""

#: name -> (the statement, what the last cell reads)
CASES = {
    "users-method": ("Mixer().go(SEED)", "DECK"),
    "users-method-bound": ("m = Mixer()\nm.go(SEED)", "DECK"),
    "users-method-of-attribute": ("Holder(DECK).go(SEED)", "DECK"),
    "users-method-in-loop": ("for _ in range(2):\n    Mixer().go(SEED)", "DECK"),
    "users-method-in-helper": ("mix_through_method(SEED)", "DECK"),
    "called-lambda": ("mix(SEED)", "DECK"),
    "called-lambda-in-loop": ("for _ in range(2):\n    mix(SEED)", "DECK"),
    "called-lambda-on-module-global": ("mix_module_lambda(SEED)", "hhm.MODDECK"),
    "users-method-on-module-global": ("ModuleMixer().go(SEED)", "hhm.MODDECK"),
    "module-function": ("hhm.mix(SEED)", "hhm.MODDECK"),
    "helper-on-module-global": ("mix_module(SEED)", "hhm.MODDECK"),
    "module-function-assigned": ("n = hhm.mix(SEED)", "hhm.MODDECK"),
    "module-function-imported": ("from hhm import mix as hmix\nhmix(SEED)", "hhm.MODDECK"),
    "module-function-inner": ("hhm.mix_inner(SEED)", "hhm.MODDECK"),
    "module-class-method": ("hhm.ModMixer().go(SEED)", "hhm.MODDECK"),
    "module-global-in-statement": ("random.Random(SEED).shuffle(hhm.MODDECK)", "hhm.MODDECK"),
    "module-function-in-loop": ("for _ in range(2):\n    hhm.mix(SEED)", "hhm.MODDECK"),
}


def _cells(case: str, seed: int) -> list[str]:
    statement, read = CASES[case]
    return [
        FLOORS,
        HELPERS,
        f"SEED = {seed}",
        "DECK = list(range(8))",
        statement,
        f"top = {read}[:4]\nprint(top)",
    ]


def _plain(cells: list[str], module: types.ModuleType) -> str:
    """Run All in a plain kernel whose import of ``hhm`` is *module*: kept
    from one Run All to the next, as a kernel keeps it until a restart."""
    sys.modules["hhm"] = module
    try:
        namespace: dict = {"__name__": "__main__"}
        for code in cells[1:]:
            exec(code, namespace)
        return repr(namespace["top"])
    finally:
        sys.modules.pop("hhm", None)


@pytest.mark.parametrize("case", list(CASES))
def test_every_run_matches_plain_python(nb_runner, case):
    plan = [("run", 1), ("run", 1), ("restart", 1), ("edit", 2), ("restart", 2), ("edit", 1), ("restart", 1)]
    nb_runner.create_notebook(_cells(case, 1))
    (nb_runner.work_dir / "hhm.py").write_text(HHM, encoding="utf-8")
    nb_runner.start_kernel()
    got, want = [], []
    module = None
    for step, seed in plan:
        cells = _cells(case, seed)
        if step == "restart" or module is None:
            module = types.ModuleType("hhm")
            exec(HHM, module.__dict__)
        if step == "edit":
            nb_runner.set_cell_source(3, cells[2])
        if step == "restart":
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.run_all()
        got.append(nb_runner.peek("top").strip())
        want.append(_plain(cells, module))
    assert got == want


READER = """
import time
class Summer:
    def total(self, seed):
        time.sleep(0.3)
        return sum(DECK) + seed
"""


def test_a_method_that_only_reads_a_global_is_served_after_a_restart(nb_runner):
    """Following a method of the user's class into its body finds nothing it
    changes: the statement is still served after a restart."""
    cells = [FLOORS + "\n%cash_badge print", READER, "SEED = 1", "DECK = list(range(8))", "s = Summer().total(SEED)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("s").strip() == "29"
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert shows_cached(out), out
    assert nb_runner.peek("s").strip() == "29"
