"""CAS-261 step 2: a cheap many-iteration loop is learned and then split.

The band, measured on HEAD before this landed (n=124, warm rerun, wall clock
against a cash-off arm):

    body 0.1ms  ->  22ms off vs 215ms on   (9.8x SLOWER)
    body 1ms    -> 137ms off vs 324ms on   (2.4x slower)
    body 2.5ms  -> 320ms off vs 617ms on   (1.9x slower)

Every one re-ran all 124 calls AND paid ~190-300ms of per-iteration
decomposition overhead. Not a missed optimisation -- cash being worse than
not using cash. It falls between two floors: n=124 is under the single-unit
threshold so the loop decomposes, and each call is under
``call_unit._COST_FLOOR_S`` so nothing is stored per call either.

**The split is applied by the SIMULATOR, not the runtime.** The re-execution
planner runs the statements ``upstream/virtual_lineage.py`` modelled, so
splitting that model is what makes a head and a tail actually execute. Three
earlier attempts split in the runtime instead and each produced a silent
stale value; ``test_lineage_check_control_structure.py::
test_for_loop_still_reexecutes_when_inputs_change`` is the case that caught
them and must stay green.

Measured as counted real executions, never wall clock -- a timing assertion
would measure the machine under parallel load rather than the policy.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SETUP = "import cash\n%cash_on\n"
SETUP_OFF = "import cash\n"
UNRELATED = "unrelated = 1\n"

# Under the single-unit threshold (n>=125 for a 1-statement body), so this
# loop decomposes -- the band's shape.
_N = 124
# The runtime measures this many iterations before judging, and it becomes
# the split point. The head re-runs on every warm pass, so warm cost is
# bounded by it.
_K = 5


def _compute_def(counter, ms=1.0):
    """Busy-wait, not sleep: Windows sleep granularity (~1-15ms) would swamp
    a 1ms body and make the split verdict depend on the OS timer."""
    return (
        "import time\n"
        "def _busy(ms):\n"
        "    t = time.perf_counter() + ms / 1000.0\n"
        "    while time.perf_counter() < t:\n"
        "        pass\n"
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        f"    _busy({ms})\n"
        "    return v * 10\n"
    )


def _loop(n=_N):
    # Items start at 1, never 0: compute(0) returns `0 * 10`, which CPython
    # interns to the argument object itself and trips the call unit's
    # arg-identity refusal -- one permanently-uncached call that mimics a
    # caching bug (CAS-263).
    return f"for t in list(range(1, {n + 1})):\n    out.append(compute(t))\nprint('OUT', len(out))"


def _cells(counter, setup=SETUP, n=_N, ms=1.0):
    """SETUP(1) / unrelated(2) / compute def(3) / seed(4) / loop(5)."""
    return [setup, UNRELATED, _compute_def(counter, ms), "out = []\n", _loop(n)]


LOOP_CELL = 5
UNRELATED_CELL = 2
DEF_CELL = 3


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def test_a_learned_loop_splits_and_then_costs_only_its_head(nb_runner, tmp_path):
    """Three runs, because learning costs one:

    * **run 1** -- the loop is unknown, so it decomposes and stores
      per-iteration entries. The verdict is recorded at the end.
    * **run 2** -- the verdict now exists, so the loop splits. The TAIL has
      never been stored (run 1 wrote per-iteration entries, not a tail), so
      it is a cold miss and runs. Measured: 123 of 124.
    * **run 3 onward** -- the tail hits; only the head runs.

    That one-run lag is inherent to learning from measurement: nothing can
    know a loop is worth splitting until it has been seen. Asserting on run 2
    would have looked like a failure of the split rather than the cost of
    learning, which is why this test is explicit about all three.

    Before this landed, every rerun measured 124 -- the whole loop, forever.
    """
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N, f"baseline did not run all {_N} items: {cold}"

    nb_runner.run_cell(LOOP_CELL)          # run 2: splits, tail stores
    after_learning = _n(counter)

    nb_runner.run_cell(LOOP_CELL)          # run 3: tail hits
    steady = _n(counter) - after_learning
    assert steady <= _K, (
        f"a settled rerun re-ran {steady}/{_N} calls; expected at most the "
        f"{_K}-iteration head. Without a split this loop caches nothing and "
        "is slower than running with cash off."
    )
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_the_split_preserves_the_value(nb_runner, tmp_path):
    """Head + tail must produce exactly what the undivided loop produced,
    including the leaked loop variable."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter) + [
        "print('SUM', sum(out), 'LAST', out[-1], 'T', t, 'LEN', len(out))"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = nb_runner.get_output(6)
    assert expected == (f"SUM {sum(range(1, _N + 1)) * 10} LAST {_N * 10} "
                        f"T {_N} LEN {_N}"), expected

    nb_runner.run_cell(LOOP_CELL)
    nb_runner.run_cell(6)
    assert nb_runner.get_output(6) == expected, (
        f"the split changed the loop's result:\n  cold: {expected}\n"
        f"  warm: {nb_runner.get_output(6)}"
    )


def test_a_dependency_edit_still_recomputes(nb_runner, tmp_path):
    """Negative control. Without it every assertion above would pass just as
    happily if the split were caching too eagerly."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.set_cell_source(DEF_CELL, _compute_def(counter).replace("v * 10", "v * 20"))
    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"dependency edit only re-ran {warm}/{_N} calls"
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_an_expensive_body_is_never_split(nb_runner, tmp_path):
    """The gate protecting CAS-259's incremental reuse.

    A loop whose calls clear the cost floor is served by per-call caching,
    which gives INCREMENTAL reuse -- append one item, re-run one call. A
    split's tail is all-or-nothing: its key covers the whole remaining
    iterable, so an append re-runs it entirely. Splitting an expensive-bodied
    loop would silently trade a better mechanism for a worse one.

    Measured via an append: incremental reuse costs 1 real call; a split
    would cost the whole tail.
    """
    counter = tmp_path / "calls.log"
    n = 60
    nb_runner.create_notebook(_cells(counter, n=n, ms=20.0))   # 20ms >> 3ms floor
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == n

    nb_runner.set_cell_source(LOOP_CELL, _loop(n + 1))
    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == 1, (
        f"append re-ran {warm}/{n + 1} calls, expected 1. An expensive-bodied "
        "loop must keep per-call incremental reuse (CAS-259) rather than "
        "being split into an all-or-nothing tail."
    )


def test_oracle_no_caching(nb_runner, tmp_path):
    """Cash off: the same shape re-runs everything. Pins the harness, so a
    zero above means reuse rather than a broken counter."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter, setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"oracle only re-ran {warm}/{_N}; the harness is not measuring real work"
