"""A statement that draws from a generator held in a variable moves it on.

Seen in a notebook::

    rng = np.random.default_rng(7)               # cell 2
    d = {k: draw(k, rng) for k in range(3)}      # cell 3, slow enough to store

Editing cell 3 to ``range(6)`` and running it alone drew from the ``rng`` the
first run had already moved. That value was stored under the key of a draw
from the fresh ``rng``: the draw moved the generator without rebinding it, so
``rng`` kept its lineage and both states keyed alike. Re-running cell 2 to
reseed and then cell 3 restored the moved-generator values, which no
top-to-bottom run gives, and so did Restart & Run All.

These pin that a draw gives the generator's variable a new lineage, on a run
and on a hit alike, so a statement reading a fresh generator never keys like
one reading a moved one: the spellings that draw without rebinding (a
comprehension, a plain assignment, a helper imported from a module, a helper
reading the generator as a global, a loop) and Restart & Run All. The oracle is the
same draws made top to bottom without cash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

np = pytest.importorskip("numpy")

SETUP = "import cash\n%cash_on"
SEED = "import numpy as np, json, time\nrng = np.random.default_rng(7)"

# Each draw sleeps so the statement is worth storing: a cheap one is never
# written, and the stale entry never exists.
DRAW = """def draw(k, r):
    time.sleep(0.2)
    return round(float(r.normal()) + k, 9)
"""

COMPREHENSION = DRAW + 'd = {{k: draw(k, rng) for k in range({n})}}\nprint("RESULT", json.dumps(d))'
ASSIGNMENT = DRAW + 'x = draw({n}, rng)\nprint("RESULT", json.dumps({{"0": x}}))'
# The generator is never named in the statement: the helper reads it.
GLOBAL_HELPER = """def draw(k):
    time.sleep(0.2)
    return round(float(rng.normal()) + k, 9)
d = {{k: draw(k) for k in range({n})}}
print("RESULT", json.dumps(d))"""

MODULE = "carrier_draw_helpers"
MODULE_SOURCE = """import time


def draw(k, r):
    time.sleep(0.2)
    return round(float(r.normal()) + k, 9)
"""
FROM_MODULE = (
    f"from {MODULE} import draw\n" + 'd = {{k: draw(k, rng) for k in range({n})}}\nprint("RESULT", json.dumps(d))'
)


def _drawn(ks):
    """What drawing once per k from a fresh ``default_rng(7)`` gives."""
    r = np.random.default_rng(7)
    return {str(i): round(float(r.normal()) + k, 9) for i, k in enumerate(ks)}


def _comprehension(n):
    return _drawn(range(n))


def _assignment(n):
    return _drawn([n])


def _result(nb_runner):
    out = nb_runner.get_output(3)
    line = next((line for line in out.splitlines() if line.startswith("RESULT ")), None)
    assert line is not None, out
    return json.loads(line[len("RESULT ") :])


def _edit_rerun_reseed(nb_runner, cell, first, second, oracle):
    """Run all, edit the draw and run it alone, reseed and run it again,
    then Restart & Run All, checking each result against *oracle*."""
    nb_runner.create_notebook([SETUP, SEED, cell.format(n=first)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _result(nb_runner) == oracle(first)

    nb_runner.set_cell_source(3, cell.format(n=second))
    nb_runner.run_cell(3)
    # On its own the edited cell first rebuilds the generator it reads.
    assert _result(nb_runner) == oracle(second), "the edited draw started from a moved generator"

    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert _result(nb_runner) == oracle(second), "after a reseed the draw was served a moved generator's values"

    nb_runner.restart()
    nb_runner.run_all()
    assert _result(nb_runner) == oracle(second), "Restart & Run All served a moved generator's values"


def test_a_comprehension_drawing_from_a_held_generator(nb_runner):
    _edit_rerun_reseed(nb_runner, COMPREHENSION, 3, 6, _comprehension)


def test_a_plain_assignment_drawing_from_a_held_generator(nb_runner):
    _edit_rerun_reseed(nb_runner, ASSIGNMENT, 0, 1, _assignment)


def test_a_helper_from_a_module_drawing_from_a_held_generator(nb_runner):
    Path(nb_runner.work_dir, f"{MODULE}.py").write_text(MODULE_SOURCE, encoding="utf-8")
    _edit_rerun_reseed(nb_runner, FROM_MODULE, 3, 6, _comprehension)


def test_a_helper_reading_the_generator_as_a_global(nb_runner):
    _edit_rerun_reseed(nb_runner, GLOBAL_HELPER, 3, 6, _comprehension)


def test_a_restored_draw_moves_its_generator_on_too(nb_runner):
    """A hit on the draw gives ``rng`` the lineage its run gave it.

    Without that, the warm Run All below keyed the next draw like one
    reading the fresh generator and stored the moved generator's value under
    that key, and the next draw was served it once the first draw was gone.
    """
    nxt = (
        "def next_draw(r):\n"
        "    time.sleep(0.2)\n"
        "    return round(float(r.normal()), 9)\n"
        "e = next_draw(rng)\n"
        'print("NEXT", e)'
    )
    nb_runner.create_notebook([SETUP, SEED, COMPREHENSION.format(n=3), nxt])
    nb_runner.start_kernel()
    r = np.random.default_rng(7)
    first = {str(k): round(float(r.normal()) + k, 9) for k in range(3)}
    after = round(float(r.normal()), 9)
    fresh = round(float(np.random.default_rng(7).normal()), 9)

    for _ in range(2):  # cold, then warm: the draw is restored
        nb_runner.run_all()
        assert _result(nb_runner) == first
        assert f"NEXT {after}" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.set_cell_source(3, 'd = {}\nprint("RESULT", json.dumps(d))')
    nb_runner.run_all()
    assert f"NEXT {fresh}" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.restart()
    nb_runner.run_all()
    assert f"NEXT {fresh}" in nb_runner.get_output(4), nb_runner.get_output(4)


def test_a_loop_drawing_from_a_held_generator(nb_runner):
    """A loop's body runs statement by statement; the loop as a whole moves
    the generator on, so the slow draw after it keys on the moved one."""
    loop = "d = {}\nfor k in range(3):\n    d[k] = float(rng.normal())"
    after = DRAW + 'x = draw({n}, rng)\nprint("AFTER", x)'

    def expected(n):
        r = np.random.default_rng(7)
        [r.normal() for _ in range(3)]
        return f"AFTER {round(float(r.normal()) + n, 9)}"

    def check(n, why):
        assert expected(n) in nb_runner.get_output(4), f"{why}: {nb_runner.get_output(4)}"

    nb_runner.create_notebook([SETUP, SEED, loop, after.format(n=0)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    check(0, "top to bottom")
    nb_runner.set_cell_source(4, after.format(n=1))
    nb_runner.run_cell(4)
    check(1, "the edited draw started elsewhere than after the loop")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    check(1, "after a reseed the draw was served another position's value")
    nb_runner.restart()
    nb_runner.run_all()
    check(1, "Restart & Run All served another position's value")
