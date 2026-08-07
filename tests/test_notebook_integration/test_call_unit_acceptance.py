"""The iterator-collapse guard, end to end, on the DEFAULT (no-directive) path.

CAS-243's acceptance table (produced by hand-decorating ``compute``) has two
halves, and both are already pinned elsewhere with assertions on call counts,
not wall-clock:

- **The fold/reorder half** ("append costs one call, every reorder after that
  costs zero, a genuinely new value costs one call again") is pinned by
  ``test_loop_reorder_reuse_claim.py::test_accumulator_fold_reordering_is_free_by_default``
  and ``test_cache_calls_directive.py::test_reordering_an_accumulator_fold_costs_nothing_with_no_directive``.
- **The append-in-a-loop-reuses-on-rerun half** is pinned by
  ``test_cache_calls_directive.py::test_append_loop_caches_the_call_with_no_directive``
  and ``test_loop_reorder_reuse_claim.py::test_subscript_store_and_undirected_append_both_reuse_by_default``.

Duplicating either here would be exactly the "fourth near-duplicate" this
task was warned against, so this file does not re-assert them.

What is genuinely NOT pinned anywhere with an assertion: the CAS-243 §1a
"iterator collapse" scenario end to end, through the real per-iteration
``CallUnit`` key-building wiring (``arg_digests`` discriminating a computed
argument by its evaluated VALUE, not a hidden object's id-stable lineage), on
the interception-is-on-by-default path (no ``# @cash:cache-calls`` directive
anywhere). The key-construction logic itself is already unit-tested directly
(``tests/test_notebook/test_call_unit_key.py::test_a_computed_argument_discriminates_by_VALUE_not_lineage``),
and a probe (``zzprobe_hidden_state_arg.py``, in this directory) exercised a
real-kernel shape of it -- but only under the OLD opt-in
``# @cash:cache-calls`` directive, and with no assertion at all ("this probe
reports, it does not gate").

**The loop body shape matters and is not interchangeable with the brief's.**
The brief's own proposed body, ``out.append(compute(next(it)))``, is a bare
``Expr(Call)`` loop body -- exactly the shape
``control_structures/processor.py``'s accumulator single-unit fast path
(``cacheable_accumulator_loop``) claims, per
``test_call_unit_loop_vars_real_kernel.py``'s own module docstring. Verified
directly here (mutation-tested, see the task report): with that body, the
WHOLE loop is cached as one unit and the individual ``compute()`` calls never
go through per-iteration ``CallUnit`` interception at all -- the badge shows
one ``COMPUTED: for _ in range(3): -> RAM+DISK`` line and no ``sub-call`` /
``[intercepted]`` entries. A test built on that shape would pass or fail
identically whether ``call_cache_key``'s ``arg_digests`` channel works or is
deleted outright, which is exactly the "rerun-shaped test can pass regardless
of the behaviour under test" trap this task's brief warned about. The fix is
the two-statement body below (``v = compute(next(it))`` then
``out.append(v)``), which is never eligible for the single-statement fast
path and forces real per-iteration decomposition -- the badge for this shape
shows three separate ``sub-call compute(next(it)): 0/1 hit`` lines and an
``@cash.cache: compute() [intercepted]`` summary, proving the call actually
went through ``CallUnit``.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above the cost model's ~10ms floor, or `compute` is never stored at all and
# every assertion below would hold whether or not this feature works.
_SLEEP = 0.2

DEFS = f"""\
import time
CALLS = []
def compute(v):
    CALLS.append(v)
    time.sleep({_SLEEP})
    return v * 10
"""

# `CALLS.clear()` at the top makes every subsequent `print` a stable per-run
# view rather than a running total.
#
# It is NOT a count of real executions, and must not be read as one. `CALLS`
# is a global written from inside `compute`'s body, so CAS-260/265 captures it
# per call and restores it on a hit: a served call reproduces its append
# without executing. That is the point of the feature -- the observable state
# matches the uncached oracle either way -- and it means an execution count
# needs an instrument OUTSIDE the cached region (see the rerun test below).
#
# Two-statement body (`v = compute(next(it))`, then `out.append(v)`) -- NOT
# the brief's bare `out.append(compute(next(it)))`. See the module docstring:
# a bare-append body is eligible for the accumulator single-unit fast path,
# which caches the whole loop and never reaches per-iteration `CallUnit`
# interception for the calls inside it.
LOOP = """\
CALLS.clear()
it = iter(range(3))
out = []
for _ in range(3):
    v = compute(next(it))
    out.append(v)
print('OUT', out, 'CALLS', CALLS)
"""


def test_loop_carried_hidden_state_matches_the_uncached_oracle(nb_runner):
    """First run, no pre-existing cache, no directive.

    A key that ignored the evaluated argument (relying on ``it``'s lineage
    alone) would collapse all three iterations onto one entry and serve
    iteration 1's value three times: ``OUT [0, 0, 0]``. The correct answer,
    matching the cash-off oracle below, is ``OUT [0, 10, 20]`` with all three
    ``compute`` calls genuinely executed.
    """
    nb_runner.create_notebook(["import cash\n%cash_on\n", DEFS, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(3)
    assert "OUT [0, 10, 20] CALLS [0, 1, 2]" in output, output


def test_loop_carried_hidden_state_matches_the_no_cash_oracle(nb_runner):
    """Independent confirmation that ``[0, 10, 20]`` is correct -- cash off entirely.

    Without this, the test above could be trivially "satisfied" by any bug
    that always produces ``[0, 10, 20]`` regardless of caching -- it needs an
    oracle run that never touches cash's caching machinery at all.
    """
    nb_runner.create_notebook(["import cash\n", DEFS, LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert "OUT [0, 10, 20] CALLS [0, 1, 2]" in nb_runner.get_output(3)


def test_loop_carried_hidden_state_reuses_on_an_unchanged_rerun(nb_runner, tmp_path):
    """Re-running the identical loop cell must replay from cache.

    ``it = iter(range(3))`` is itself part of the cell, so a fresh iterator is
    built every run; a correct cache still serves ``OUT [0, 10, 20]`` without
    executing ``compute`` again. A collapsed key would instead calcify at the
    wrong ``[0, 0, 0]`` and replay THAT stably, which the two tests above rule
    out for the first run -- this one checks that correctness survives a
    second execution, not just the first.

    Reuse is proven by a counter OUTSIDE the cached region, not by ``CALLS``.
    This test previously asserted ``CALLS []`` on the rerun, reading "the
    callee's append is missing" as "the callee did not run". CAS-260/265 makes
    that inference invalid in the direction of correctness: a served call now
    restores the global its body wrote, so ``CALLS`` reads ``[0, 1, 2]`` on a
    rerun -- byte-identical to the uncached oracle pinned by
    ``test_loop_carried_hidden_state_matches_the_no_cash_oracle`` -- while the
    work is still skipped. The old assertion required the cached run to
    DIVERGE from the oracle, and would now be satisfied only by the bug.

    ``os.open``/``os.write``, not ``builtins.open``, which ``FileAccessTracker``
    patches -- a tracked file dependency changes every run, so the entry is
    never fresh and the instrument silently disables what it measures.
    """
    ticks = tmp_path / "real.log"
    defs_ticked = DEFS.replace(
        "import time\n",
        "import time, os\n"
        "def _tick():\n"
        f"    fd = os.open(r'{ticks}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n",
    ).replace("    CALLS.append(v)\n", "    CALLS.append(v)\n    _tick()\n")
    assert "_tick()" in defs_ticked, "the instrument was not spliced into compute()"

    nb_runner.create_notebook(["import cash\n%cash_on\n", defs_ticked, LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT [0, 10, 20] CALLS [0, 1, 2]" in nb_runner.get_output(3)
    cold = ticks.read_bytes()
    assert len(cold) == 3, f"the cold run did not execute compute() three times: {len(cold)}"

    nb_runner.run_cell(3)
    output = nb_runner.get_output(3)
    assert "OUT [0, 10, 20] CALLS [0, 1, 2]" in output, (
        f"the rerun diverged from the uncached oracle:\n{output}"
    )
    rerun = len(ticks.read_bytes()) - len(cold)
    assert rerun == 0, (
        f"compute() ran {rerun} more time(s) on an unchanged rerun of the loop cell"
    )
