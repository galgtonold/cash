"""Hidden state cash cannot see or restore — one live limitation, three fixed.

This file started (2026-06-28 correctness probe sweep) as a pair of "fundamental"
xfails. Three of the four have since been fixed, which is the point of keeping
them here: the corpus is a regression record, not a monument.

Still failing (non-strict xfail):

* ``test_function_hidden_global_mutation_rerun`` — a cell calls ``tick()``, which
  mutates a global the cell never names. Cash cannot see the mutation, so an
  isolated re-run does not restore the global to its cell-entry base and the call
  advances it again. Genuinely fundamental to source/lineage-based caching, and it
  self-disables on ``run_all`` (the producer cell re-runs first).

Fixed, kept as regression tests:

* ``test_exhausted_generator_rerun`` — CAS-118 / CAS-50: producers of consumed
  unrestorable inputs are re-executed on an isolated re-run, so the generator is
  re-seeded instead of being observed empty.
* ``test_global_keyword_mutation_rerun`` and ``test_mutable_default_arg_rerun`` —
  CAS-49 family via CAS-93: definition statements always re-execute.
"""
import pytest

pytestmark = pytest.mark.upstream


@pytest.mark.xfail(reason="Hidden global mutation via a function the cell does not "
                          "name: cash cannot see that tick() mutates the global "
                          "`c`, so on an isolated re-run `c` is not restored to its "
                          "cell-entry base and the call advances it again. The "
                          "@stateful decorator forces re-execution but still reads "
                          "the advanced global. Fundamental impure-function limit.",
                   strict=False)
def test_function_hidden_global_mutation_rerun(nb_runner):
    nb_runner.create_notebook([
        "c = {'n': 0}\ndef tick():\n    c['n'] += 1\n    return c['n']",
        "r = tick()\nprint(r)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "1" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert nb_runner.get_output(2).strip().endswith("1"), nb_runner.get_output(2)


# FIXED (CAS-118, CAS-50): an upstream generator is exhausted after first
# consumption and cannot be pickled/restored — so cash now re-executes the
# PRODUCER of a consumed unrestorable input on an isolated re-run. `g` is
# re-seeded and `list(g)` sees the full sequence again instead of an empty
# generator.
def test_exhausted_generator_rerun(nb_runner):
    nb_runner.create_notebook([
        "g = (i for i in range(3))",
        "vals = list(g)\nprint(vals)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[0, 1, 2]" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert "[0, 1, 2]" in nb_runner.get_output(2), nb_runner.get_output(2)


# FIXED (CAS-49 family, via CAS-93): definition statements always re-execute
# now, so the isolated re-run re-runs `def inc()` and the upstream chain
# re-seeds `g` — the hidden global mutation no longer accumulates.
def test_global_keyword_mutation_rerun(nb_runner):
    nb_runner.create_notebook([
        "g = 0\ndef inc():\n    global g\n    g += 1",
        "inc()\nprint(g)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "1" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert nb_runner.get_output(2).strip().endswith("1"), nb_runner.get_output(2)


# FIXED (CAS-49 family, via CAS-93): definition statements always re-execute
# now, so the isolated re-run recreates the function object — and with it a
# FRESH mutable default — instead of reusing the accumulated one.
def test_mutable_default_arg_rerun(nb_runner):
    nb_runner.create_notebook([
        "def acc(x, bucket=[]):\n    bucket.append(x)\n    return bucket",
        "r = acc(1)\nprint(r)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[1]" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert nb_runner.get_output(2).strip().endswith("[1]"), nb_runner.get_output(2)
