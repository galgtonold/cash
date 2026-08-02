"""CAS-261 step 2: a many-cheap-iteration loop promotes to a single unit.

The band this closes, measured on HEAD before the fix (n=124, warm rerun,
wall-clock against a cash-off arm):

    body 0.1ms  ->  22ms cash-off vs 215ms cash-on   (9.8x SLOWER)
    body 1ms    -> 137ms cash-off vs 324ms cash-on   (2.4x slower)
    body 2.5ms  -> 320ms cash-off vs 617ms cash-on   (1.9x slower)

Every one of those re-ran all 124 calls AND paid ~190-300ms of per-iteration
decomposition overhead. That is not a missed optimisation -- it is cash being
worse than not using cash. It sits between two floors: n=124 is below the
single-unit threshold (n>=125 for a 1-statement body) so the loop decomposes,
and each call is below ``call_unit._COST_FLOOR_S`` (3ms) so nothing is stored
per call either.

Why the fix is a whole-loop unit and NOT per-call storage
---------------------------------------------------------
Storing N calls saves ``N x body`` but costs ``N x hit``, so the ratio is per
call and no aggregate total rescues a body below the ~1.2ms break-even --
that is CAS-261's rejected Option 2, and
``test_loop_edit_rerun_matrix.py::test_sub_break_even_calls_are_not_stored_
individually`` is the guard against reintroducing it. A single unit is
different in kind: ONE store amortises over every iteration, so the saving
scales with N while the cost does not.

Measured as counted real executions of the loop body's callee, never wall
clock -- a timing assertion here would measure the machine under parallel
load rather than the policy.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SETUP = "import cash\n%cash_on\n"
SETUP_OFF = "import cash\n"
UNRELATED = "unrelated = 1\n"

# Below the single-unit threshold (n>=125 for a 1-statement body), so this
# loop decomposes and is the band's shape.
_N = 124
# The promotion probe runs this many iterations before deciding. The head
# re-runs on every warm pass, so warm cost is bounded by it.
_PROBE = 5


def _compute_def(counter):
    """A callee far below the 3ms call floor: nothing is ever stored per call."""
    return (
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    return v * 10\n"
    )


def _loop(n=_N):
    # Items start at 1, never 0: compute(0) would return `0 * 10`, which
    # CPython interns to the argument object itself and trips the call unit's
    # arg-identity refusal -- one permanently-uncached call that mimics a
    # caching bug (CAS-263).
    return f"for t in list(range(1, {n + 1})):\n    out.append(compute(t))\nprint('OUT', len(out))"


def _cells(counter, setup=SETUP, n=_N):
    """SETUP(1) / unrelated(2) / compute def(3) / seed(4) / loop(5)."""
    return [setup, UNRELATED, _compute_def(counter), "out = []\n", _loop(n)]


LOOP_CELL = 5
SEED_CELL = 4
UNRELATED_CELL = 2


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def test_cheap_many_iteration_loop_promotes(nb_runner, tmp_path):
    """An unchanged warm rerun must cost at most the promotion probe head.

    Before step 2 this measured 124 -- the full loop, every time.
    """
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N, f"baseline did not run all {_N} items: {cold}"

    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm <= _PROBE, (
        f"warm rerun re-ran {warm}/{_N} calls; expected at most the {_PROBE}-"
        "iteration promotion probe. The loop is below BOTH the single-unit "
        "threshold and the per-call cost floor, so without promotion it "
        "caches nothing and is slower than running with cash off."
    )
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_promotion_does_not_break_the_value(nb_runner, tmp_path):
    """Head + tail must produce exactly what an undivided loop produces."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter) + [
        "print('SUM', sum(out), 'LAST', out[-1], 'T', t)"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = nb_runner.get_output(6)
    # sum(1..124)*10, last element, and the leaked loop variable.
    assert expected == f"SUM {sum(range(1, _N + 1)) * 10} LAST {_N * 10} T {_N}", expected

    nb_runner.run_cell(LOOP_CELL)
    nb_runner.run_cell(6)
    assert nb_runner.get_output(6) == expected, (
        "promoted rerun changed the loop's result:\n"
        f"  cold: {expected}\n  warm: {nb_runner.get_output(6)}"
    )


def test_promoted_loop_still_invalidates_on_a_real_dependency_edit(nb_runner, tmp_path):
    """Negative control: editing the callee must recompute.

    Without this, every assertion above would pass just as happily if
    promotion were caching too eagerly.
    """
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter))
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.set_cell_source(3, _compute_def(counter).replace("v * 10", "v * 20"))
    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"dependency edit only re-ran {warm}/{_N} calls"
    assert f"OUT {_N}" in nb_runner.get_output(LOOP_CELL)


def test_expensive_body_is_not_promoted(nb_runner, tmp_path):
    """The gate that protects CAS-259's incremental reuse.

    A loop whose calls clear the cost floor is served by per-call caching,
    which gives INCREMENTAL reuse -- append one item, re-run one call. A
    whole-loop unit is all-or-nothing: its key covers the whole iterable, so
    an append re-runs everything. Promoting an expensive-bodied loop would
    therefore silently trade a better mechanism for a worse one.

    Measured via an append: incremental reuse costs 1 real call, promotion
    would cost all of them.
    """
    counter = tmp_path / "calls.log"
    expensive = (
        "import time\n"
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    time.sleep(0.02)\n"       # 20ms: well clear of the 3ms floor
        "    return v * 10\n"
    )
    n = 60
    nb_runner.create_notebook([SETUP, UNRELATED, expensive, "out = []\n", _loop(n)])
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
        "being promoted to an all-or-nothing whole-loop unit."
    )


def test_oracle_no_caching(nb_runner, tmp_path):
    """Cash off: the same shape re-runs everything. Pins the harness."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_cells(counter, setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N

    nb_runner.run_cell(LOOP_CELL)
    warm = _n(counter) - cold
    assert warm == _N, f"oracle only re-ran {warm}/{_N}; harness is not measuring real work"
