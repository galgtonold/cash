"""Documented known limitations: hidden state cash cannot see or restore.

Both are fundamental to source/lineage-based caching (any such system shares
them) and self-disable on ``run_all`` (the producer cell re-runs first). They are
recorded as non-strict xfails so the corpus flags it loudly if either ever
starts passing. Discovered by the correctness probe sweep (2026-06-28).
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


@pytest.mark.xfail(reason="Stateful iterator: an upstream generator is exhausted "
                          "after first consumption and cannot be pickled/restored, "
                          "so an isolated re-run of `list(g)` sees an empty "
                          "generator. Fundamental unrestorable-state limit.",
                   strict=False)
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
