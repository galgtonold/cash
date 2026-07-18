"""Loop-carried accumulator recurrences skip their expensive work on a warm
re-run (CAS-204 — investigated, found already correct, pinned here).

CAS-204 was filed off a round-8 gate report that the "running-balance equity
accumulator" (``acc = acc + work(e); curve.append(acc)``) was *not cached*, and
proposed widening the CAS-145 ``cacheable_accumulator_loop`` shape gate to admit
it. Measured against the real-kernel oracle with an EXTERNAL call counter, that
premise did not hold: the recurrence already caches in every arrangement tried
(seeds as two statements, as a tuple assign, in a previous cell, with a
statement between, and adjacent to another loop — the reporter's own A/B shape).

What actually happens, per the ``%cash_badge print`` output:

    COLD  COMPUTED: acc = acc + slow(e)   x5     (5 real calls)
    WARM  RESTORED: acc = acc + slow(e)   x5     (0 real calls, "saved 0.15s")
          COMPUTED: curve.append(acc)     x5     (re-runs, by design)

The expensive recurrence statement is cached per-iteration, so the work is
skipped; the cheap ``curve.append(acc)`` re-executes because it is an in-place
mutation that must rebuild the list (restoring it instead would be the alias
hazard CAS-184 covers). Reading that per-statement "in-place mutation" badge as
"the loop is not cached" is what produced the report.

So the gate was NOT widened — doing so would have added risk to a load-bearing
correctness boundary for zero speedup. These tests pin the invariant that
matters: the body's expensive call must not re-fire on a warm re-run. The
counter is the witness; the badge is not (it is itself restored on a hit).
"""
import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations, pytest.mark.timeout(180)]

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time"


def _slow_def(counter):
    return (
        "def slow(e):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    time.sleep(0.03)\n"
        "    return e * 10"
    )


def _cold_warm_calls(nb_runner, tmp_path, loop_src, tail, loop_idx=4):
    """Run the notebook, then re-run ONLY the loop cell. Returns (cold, warm)
    counts of real body calls, measured from outside the kernel."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(
        [SETUP, _slow_def(counter), "items = [1, 2, 3, 4, 5]", loop_src, tail]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = len(counter.read_bytes()) if counter.exists() else 0
    tail_out = nb_runner.get_output(loop_idx + 1)

    nb_runner.run_cell(loop_idx)
    warm = (len(counter.read_bytes()) if counter.exists() else 0) - cold
    return cold, warm, tail_out


def test_append_accumulator_skips_work_on_rerun(nb_runner, tmp_path):
    """CAS-145 control: the plain append accumulator."""
    cold, warm, tail = _cold_warm_calls(
        nb_runner, tmp_path,
        "out = []\nfor e in items:\n    out.append(slow(e))",
        "print(f'out={out}')")
    assert "out=[10, 20, 30, 40, 50]" in tail
    assert (cold, warm) == (5, 0), f"append accumulator: cold={cold} warm={warm}"


def test_scalar_reduction_skips_work_on_rerun(nb_runner, tmp_path):
    """``total = total + slow(e)`` — loop-carried scalar, no list."""
    cold, warm, tail = _cold_warm_calls(
        nb_runner, tmp_path,
        "total = 0\nfor e in items:\n    total = total + slow(e)",
        "print(f'total={total}')")
    assert "total=150" in tail
    assert (cold, warm) == (5, 0), f"scalar reduction: cold={cold} warm={warm}"


def test_scalar_reduction_augassign_skips_work_on_rerun(nb_runner, tmp_path):
    """The ``total += slow(e)`` form of the same recurrence."""
    cold, warm, tail = _cold_warm_calls(
        nb_runner, tmp_path,
        "total = 0\nfor e in items:\n    total += slow(e)",
        "print(f'total={total}')")
    assert "total=150" in tail
    assert (cold, warm) == (5, 0), f"scalar AugAssign: cold={cold} warm={warm}"


def test_equity_curve_recurrence_skips_work_on_rerun(nb_runner, tmp_path):
    """THE CAS-204 shape: a running balance appended to a curve each iteration.

    Values are asserted element-wise too — a recurrence restored out of order
    would corrupt the running total, not just the final sum.
    """
    cold, warm, tail = _cold_warm_calls(
        nb_runner, tmp_path,
        "curve = []\nacc = 0\nfor e in items:\n"
        "    acc = acc + slow(e)\n    curve.append(acc)",
        "print(f'curve={curve} acc={acc}')")
    assert "curve=[10, 30, 60, 100, 150] acc=150" in tail
    assert (cold, warm) == (5, 0), f"equity curve: cold={cold} warm={warm}"


def test_equity_curve_augassign_skips_work_on_rerun(nb_runner, tmp_path):
    """AugAssign form of the equity curve."""
    cold, warm, tail = _cold_warm_calls(
        nb_runner, tmp_path,
        "curve = []\nacc = 0\nfor e in items:\n"
        "    acc += slow(e)\n    curve.append(acc)",
        "print(f'curve={curve} acc={acc}')")
    assert "curve=[10, 30, 60, 100, 150] acc=150" in tail
    assert (cold, warm) == (5, 0), f"equity curve AugAssign: cold={cold} warm={warm}"


def test_equity_curve_caches_with_seeds_in_a_previous_cell(nb_runner, tmp_path):
    """Seeds living in a different cell from the loop still skip the work."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook([
        SETUP, _slow_def(counter), "items = [1, 2, 3, 4, 5]",
        "curve = []\nacc = 0",
        "for e in items:\n    acc = acc + slow(e)\n    curve.append(acc)",
        "print(f'curve={curve} acc={acc}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = len(counter.read_bytes())
    assert "curve=[10, 30, 60, 100, 150] acc=150" in nb_runner.get_output(6)

    nb_runner.run_cell(5)
    warm = len(counter.read_bytes()) - cold
    assert (cold, warm) == (5, 0), f"cross-cell seeds: cold={cold} warm={warm}"
