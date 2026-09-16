"""Calls inside a loop cached as one unit are still cached per call.

Round 25's r25s5 fitted one model per store and department in the loop they
would naturally write::

    for (store, dept), g in feats.groupby(['store', 'department']):
        models[(store, dept)] = fit_series(g)
        scores[(store, dept)] = score(models[(store, dept)], g)

360 iterations and two statements put the loop past the threshold where it
runs as ONE unit, and calls inside a unit never reached the interceptor: fixing
one store's data re-fitted all 720 models, slower than without cash. A
comprehension would have been cached per call; the loop was not.

Inside the unit, a name the loop binds or writes has no lineage of its own --
the unit's is the whole loop's -- so it is treated like a comprehension's
variable: an argument reading it is keyed on its value, and a call whose callee
reads it as a global is not intercepted.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(300)]

N = 130

SETUP = (
    "import os, time\n"
    "def fit(values):\n"
    "    fd = os.open('fits.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'fit|')\n"
    "    os.close(fd)\n"
    # Above the many-cheap-calls guard's 50 ms, as the tester's fits were:
    # below it a few calls are timed plain, by design.
    "    time.sleep(0.06)\n"
    "    return round(sum(values) / len(values), 6)"
)
DATA = ("BUMP = {bump}\n"
        "groups = {{k: [float(k + i + (0.5 if k in BUMP else 0)) for i in range(4)] for k in range(%d)}}" % N)
LOOP = (
    "models = {}\n"
    "sizes = {}\n"
    "for key, g in groups.items():\n"
    "    models[key] = fit(g)\n"
    "    sizes[key] = len(g)\n"
    "print('TOTAL', round(sum(models.values()), 6), sum(sizes.values()))"
)


def _fits(runner) -> int:
    log = Path(runner.work_dir) / "fits.log"
    return log.read_text().count("fit|") if log.exists() else 0


def _total(bump):
    total = 0.0
    for k in range(N):
        vals = [float(k + i + (0.5 if k in bump else 0)) for i in range(4)]
        total += round(sum(vals) / 4, 6)
    return f"TOTAL {round(total, 6)} {4 * N}"


def test_fixing_one_group_refits_only_that_group(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, DATA.format(bump="[]"), LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _total([]) in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _fits(nb_runner) == N

    nb_runner.set_cell_source(3, DATA.format(bump="[7]"))
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert _total([7]) in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _fits(nb_runner) == N + 1, "every group was re-fitted"


GLOBAL_READER = (
    "import time\n"
    "def scaled(values):\n"
    "    time.sleep(0.005)\n"
    "    return sum(values) * FACTOR\n"
    f"rows = {{k: [1.0, 2.0] for k in range({N})}}"
)
GLOBAL_LOOP = (
    "out = {}\n"
    "extra = {}\n"
    "for k, v in rows.items():\n"
    "    FACTOR = k % 3\n"
    "    out[k] = scaled(v)\n"
    "    extra[k] = FACTOR\n"
    "print('SUM', sum(out.values()))"
)


def test_a_callee_reading_a_loop_name_as_a_global_is_not_served_stale(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", GLOBAL_READER, GLOBAL_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = sum(3.0 * (k % 3) for k in range(N))
    assert f"SUM {want}" in nb_runner.get_output(3), nb_runner.get_output(3)


# Past 1 MiB an argument passed by name is not hashed per call, so the key
# would hold `STATE`'s lineage -- from before the loop, as the unit updates none.
MUTATED = (
    "import time\n"
    "import numpy as np\n"
    "def weigh(state, v):\n"
    "    time.sleep(0.06)\n"
    "    return float(state[0]) * v\n"
    "def bump(state):\n"
    "    state += 1\n"
    "STATE = np.zeros(200_000)\n"
    f"items = list(range({N}))"
)
MUTATED_LOOP = (
    "res = {}\n"
    "seen = {}\n"
    "for i in items:\n"
    "    bump(STATE)\n"
    "    res[i] = weigh(STATE, 1)\n"
    "    seen[i] = i\n"
    "print('SUM', sum(res.values()))"
)


def test_a_global_another_call_in_the_loop_mutates_is_keyed_on_its_value(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", MUTATED, MUTATED_LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = float(sum(range(1, N + 1)))
    assert f"SUM {want}" in nb_runner.get_output(3), nb_runner.get_output(3)
