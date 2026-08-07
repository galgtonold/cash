"""Accumulator-loop caching (CAS-145 fast path; superseded by CAS-259).

A *pure* accumulator loop — ``out = []`` then ``for e in it: out.append(f(e))``
— is byte-identical to the comprehension ``out = [f(e) for e in it]`` yet used
to be refused caching entirely, because the ``append`` reads as an in-place
mutation ("In-place mutation on: out"; 1.0x). CAS-145 fixed that by matching
the narrow shape and routing the WHOLE loop through the statement cache as
ONE unit, capturing BOTH the accumulator AND the leaked loop variable as
outputs.

**CAS-259 removed that fast path.** Its cache key included the iterable's
lineage, so appending a SINGLE item invalidated the whole unit and re-ran
EVERY call — as expensive as no caching at all (see
``test_accumulator_single_statement_append_incremental.py`` for that
regression guard). The shape now decomposes per-iteration like any other
loop: ``out.append(slow(e))`` is still an in-place mutation and still
re-executes every run, but CAS-243's call interception now caches ``slow(e)``
itself, so the real work is still skipped on an unchanged rerun.

The correctness gates this file pins, post-CAS-259:
  * #1 the call is cached (interception hit, counted from OUTSIDE the
    kernel), result byte-identical to a no-cash run;
  * #2 / #3 a REAL kernel restart — and here the picture genuinely changed,
    though NARROWER than it first looked. The disk tier itself is not the
    issue: ``CallUnit._store`` writes through the exact same
    ``backend.set(key, value, metadata)`` the statement path uses, and
    ``TieredBackend`` promotes an entry to disk either when its metadata sets
    ``force_persist`` OR when the entry clears the backend's own generic
    compute-cost floor (~1s) regardless of any annotation. An EXPENSIVE
    intercepted call (e.g. a 1.2s ``slow``) survives a restart today, with or
    without ``@cash:persist`` — confirmed against a real kernel. What is
    actually lost was narrower: ``CallUnit._store`` never consulted the
    ``persist`` annotation and always wrote the same sparse, fixed metadata
    (``execution_time``/``timestamp``), so it could never set
    ``force_persist`` — cache *policy* did not propagate down to call units.
    For a call CHEAP enough to also miss the generic floor (this file's
    ``slow`` sleeps 0.03s, deliberately below it), ``@cash:persist`` could not
    override that, so it did NOT survive a restart.

    **The follow-up question this file used to pose — "should ``persist``, TTL
    and the size model propagate from the statement path down into
    ``CallUnit``?" — has since been answered "yes" for two of the three:** TTL
    in CAS-268, ``persist`` in CAS-269. So the pair of restart tests below now
    pins the propagating behaviour (``persist`` → restored, no directive →
    fully recomputed), with the no-directive arm as the control that keeps the
    other honest about ``slow`` staying below the generic floor. The RESULT is
    asserted correct in BOTH arms, which is the property these tests existed
    for in the first place. The size/cost model still does not propagate;

  * #4 a loop with a genuine side effect is NOT cached and re-fires every run
    (the wrong-result guard — unaffected by CAS-259, side effects still
    refuse via the normal per-statement pipeline);
  * #5 a pre-seeded / non-empty accumulator's prefix is not dropped or
    doubled (unaffected by CAS-259 — there is no shape-match to lose);
  * #6 / #7 an upstream edit of the iterable / body function invalidates.

``slow`` lives in its OWN cell (not the ``%cash_on`` setup cell) so cash tracks
its lineage and source hash — an input without a lineage would refuse the loop
for an unrelated reason ("Input variable missing lineage").
"""
import asyncio

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.mutations, pytest.mark.timeout(120)]

# ``print`` badge so the RESTORED status lands in the cell's text output.
SETUP = (
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print\n"
    "import time"
)
# Slow enough that the loop clears the "too cheap to cache" floor.
SLOW = "def slow(e):\n    time.sleep(0.03)\n    return e * 10"


def _n(path):
    """Real call count, from OUTSIDE the kernel — badge text can drift, a
    file byte count cannot."""
    return len(path.read_bytes()) if path.exists() else 0


def _slow_def(counter):
    """Like ``SLOW``, but also logs each real call to *counter* so tests can
    count real executions instead of trusting the badge's own bookkeeping."""
    return (
        "def slow(e):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    time.sleep(0.03)\n"
        "    return e * 10"
    )


def _restart(nb_runner):
    """Restart the kernel in place and re-inject the notebook path."""
    nb_runner.restart()
    nb_runner._inject_notebook_path()


# ---------------------------------------------------------------------------
# #1 Core: the accumulator loop caches, restores on an isolated re-run, and the
#    result is byte-identical to a plain (no-cash) run.
# ---------------------------------------------------------------------------

def test_accumulator_loop_caches(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook([
        SETUP,
        _slow_def(counter),
        "items = [1, 2, 3, 4, 5]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 5, f"baseline did not run all 5 iterations: {cold} calls"

    # Byte-identical to a plain kernel: slow(e) == e*10.
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Isolated re-run of the loop cell must skip ALL real work. Post-CAS-259
    # the whole-loop `RESTORED` badge is gone — `out.append(slow(e))` is a
    # genuine in-place mutation and re-executes every run — but CAS-243 call
    # interception must still cache `slow()` itself, in RAM, per iteration.
    # Real work is counted from OUTSIDE the kernel so a badge-text change
    # alone can't fake this passing.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    warm = _n(counter) - cold
    assert warm == 0, f"accumulator loop did not cache: {warm} real calls, badge={out!r}"
    assert "[intercepted]" in out, (
        f"badge does not name interception as the cache mechanism: {out!r}"
    )

    nb_runner.run_cell(5)
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #1b THE regression guard, added after a review caught it live: above the
#     cost check's own single-unit threshold (>50 iterations, >1s estimated
#     overhead), decomposition never runs at all -- so caching for this shape
#     depends ENTIRELY on the chosen single-unit branch itself being
#     cacheable. CAS-259 originally shipped without wiring ``force_outputs``
#     back into that branch (``for_handler.py``'s single-unit branch), so
#     every large/cheap accumulator loop was refused outright by the
#     in-place-mutation detector and got ZERO caching from EITHER mechanism
#     -- strictly worse than the pre-CAS-259 baseline, which cached it fine.
#     This test's absence is the entire reason that shipped; see the CAS-259
#     task report for the measured before/after.
# ---------------------------------------------------------------------------

def test_large_accumulator_loop_single_unit_still_caches(nb_runner, tmp_path):
    """150 iterations x 1 body statement -> 150*1*0.008 = 1.2s estimated
    overhead, clearing BOTH the >50-iteration and >1s-overhead cost-check
    thresholds -> the cost check chooses the single-unit branch, not
    decomposition. An isolated re-run must still skip ALL real work."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook([
        SETUP,
        _slow_def(counter),
        "items = list(range(150))",
        "out = []\nfor e in items:\n    out.append(slow(e))",
        "print(f'len={len(out)} last={out[-1]}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 150, f"baseline did not run all 150 iterations: {cold} calls"
    assert "len=150 last=1490" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Isolated re-run of the loop cell must skip ALL real work, counted from
    # OUTSIDE the kernel so a badge-text change alone can't fake this
    # passing. This is the single-unit path (unlike test #1's 5-item loop,
    # which decomposes and relies on call interception instead) -- so the
    # badge must show a genuine RESTORED whole-unit entry, not
    # `[intercepted]` sub-calls.
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    warm = _n(counter) - cold
    assert warm == 0, f"large accumulator loop did not cache: {warm} real calls, badge={out!r}"
    assert "RESTORED" in out, f"badge does not show a whole-unit restore: {out!r}"

    nb_runner.run_cell(5)
    assert "len=150 last=1490" in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #2 A REAL kernel restart, both arms. ``persist`` propagates into
#    ``CallUnit`` since CAS-269, so the annotated arm is RESTORED and the
#    bare arm is fully recomputed -- and the bare arm is what proves the
#    annotated one is measuring the annotation rather than the backend's
#    generic compute floor. What must NOT regress either way is correctness:
#    the loop must land the exact right answer, restored or recomputed.
# ---------------------------------------------------------------------------

def _restart_and_rerun_loop(nb_runner, tmp_path, annotation):
    """Cold run, real restart, re-run the loop cell. Returns re-executions.

    ``annotation`` is the directive on the FOR line (or ``""``), scoped there
    rather than to the too-cheap seed statement above it.
    """
    counter = tmp_path / "calls.log"
    loop = "out = []\n"
    if annotation:
        loop += f"{annotation}\n"
    loop += "for e in items:\n    out.append(slow(e))"
    nb_runner.create_notebook([
        SETUP,
        _slow_def(counter),
        "items = [1, 2, 3, 4, 5]",
        loop,
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5)
    cold = _n(counter)
    assert cold == 5

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on
    nb_runner.run_cell(2)  # def slow (re-derives the same source hash)
    nb_runner.run_cell(3)  # items (re-derives the same lineage)

    nb_runner.run_cell(4)
    warm = _n(counter) - cold

    # Correctness is the property that must hold in BOTH arms, restored or
    # recomputed. It is what this pair originally existed to pin.
    nb_runner.run_cell(5)
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)
    return warm


def test_persist_makes_a_cheap_accumulator_loop_survive_a_restart(nb_runner, tmp_path):
    """``# @cash:persist`` now reaches the call entry (CAS-269).

    This assertion used to read ``warm == 5``, pinning the opposite. That was
    not a mistake at the time -- it snapshotted a real gap, deliberately, with
    a note that "a future change that silently alters this either direction
    gets caught rather than passing unnoticed". CAS-269 is that change, so the
    snapshot is now the thing to update rather than defend.

    Why the annotation is load-bearing HERE and nowhere else in this file:
    ``TieredBackend`` promotes an entry to disk when its metadata sets
    ``force_persist`` OR when it clears the backend's own generic compute
    floor. An EXPENSIVE intercepted call (~1.2s) survives a restart with or
    without the directive. ``slow`` sleeps 0.03s -- deliberately below that
    floor -- so ``persist`` is the only thing that can make this durable, which
    is exactly what makes the control below meaningful.
    """
    warm = _restart_and_rerun_loop(nb_runner, tmp_path, "# @cash:persist")
    assert warm == 0, (
        f"`# @cash:persist` did not reach the loop's call entries: {warm} of 5 "
        "calls re-ran after a restart"
    )


def test_without_persist_a_cheap_accumulator_loop_recomputes(nb_runner, tmp_path):
    """NON-VACUITY for the test above, and the ``slow`` cost floor is why.

    Without this, a ``slow`` that had drifted above ``TieredBackend``'s generic
    compute floor would make the ``persist`` test pass on the floor alone,
    proving nothing about whether the annotation reaches a call entry.
    """
    warm = _restart_and_rerun_loop(nb_runner, tmp_path, "")
    assert warm == 5, (
        f"expected all 5 sub-floor calls to re-run after a restart with no "
        f"persist directive, got {warm} -- `slow` may have grown expensive "
        "enough to clear TieredBackend's generic compute floor on its own"
    )


# ---------------------------------------------------------------------------
# #3 Loop-var-leak guard: the leaked loop variable ``e`` must end up correct
#    (equal to the LAST item) after a restart, NOT stale or missing. Before
#    CAS-259 this was verified via a cache HIT restoring it; post-CAS-259
#    this shape fully recomputes after a restart (see test #2 above), so the
#    guard now verifies the same thing via a correct per-iteration REBIND
#    instead — ``bind_target_values`` runs on every iteration regardless of
#    caching, so a genuinely broken rebind (not merely a lost cache) is what
#    this test would now catch.
# ---------------------------------------------------------------------------

def test_loop_variable_correct_after_restart_recompute(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [10, 20, 30]",
        "out = []\n"
        "# @cash:persist\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        # no-cache so this always re-reads the live ``e`` rather than replaying.
        "# @cash:no-cache\n"
        "print(f'e={e}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "e=30" in nb_runner.get_output(5), nb_runner.get_output(5)

    _restart(nb_runner)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)

    # After the restart ``e`` does NOT exist. Re-running the loop cell now
    # fully recomputes it (no disk-persisted call, see test #2) — ``e`` must
    # still come out correct from that recompute's per-iteration rebind.
    nb_runner.run_cell(4)

    nb_runner.run_cell(5)
    out5 = nb_runner.get_output(5)
    assert "e=30" in out5, f"leaked loop variable wrong after restart recompute: {out5!r}"


# ---------------------------------------------------------------------------
# #4 Wrong-result guard: a loop body that appends AND performs a side effect is
#    NOT matched (the body is not a single append), so it is still refused and
#    the side effect fires on EVERY run — proven by an append-mode file that
#    grows. The file is read from the TEST side (no reader cell) so cash's
#    file-writer re-execution machinery never double-counts it.
# ---------------------------------------------------------------------------

def test_side_effect_loop_not_cached(nb_runner, tmp_path):
    sink = tmp_path / "sink.txt"
    sink_s = str(sink).replace("\\", "/")
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        f"SINK = r'{sink_s}'\nitems = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))\n"
        "    with open(SINK, 'a') as f:\n"
        "        f.write(str(e) + '\\n')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert sink.read_text().count("\n") == 3, sink.read_text()

    # Re-run the loop cell: the side effect (file append) must fire AGAIN — the
    # loop is not cached, so the file grows to 6 lines. A wrongly-cached loop
    # would skip the write and leave it at 3.
    nb_runner.run_cell(4)
    loop_out = nb_runner.get_output(4)
    assert "RESTORED" not in loop_out, f"side-effect loop was wrongly cached: {loop_out!r}"
    assert sink.read_text().count("\n") == 6, (
        f"side effect was skipped on re-run (loop wrongly cached): "
        f"{sink.read_text()!r}"
    )


# ---------------------------------------------------------------------------
# #5 A pre-seeded / non-empty accumulator is NOT matched (caching a partial
#    accumulator would drop or double its prefix). Falls back to today's
#    behaviour, which still produces the correct value.
# ---------------------------------------------------------------------------

def test_preseeded_accumulator_not_cached(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        # non-empty seed -> not the fresh-empty shape -> not matched.
        "out = [0]\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[0, 10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Isolated re-run: not cached (no RESTORED), and the value stays correct
    # (the seeded prefix is preserved, not doubled).
    nb_runner.run_cell(4)
    assert "RESTORED" not in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(5)
    assert "out=[0, 10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)


def test_prior_cell_seed_not_cached(nb_runner):
    """The seed lives in a PRIOR cell, so the loop's immediately-preceding
    top-level sibling is not ``out = []`` -> not matched."""
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []",  # seed in its own cell
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(6), nb_runner.get_output(6)

    nb_runner.run_cell(5)
    assert "RESTORED" not in nb_runner.get_output(5), nb_runner.get_output(5)


# ---------------------------------------------------------------------------
# #6 Editing the iterable upstream invalidates the cached loop.
# ---------------------------------------------------------------------------

def test_iterable_edit_invalidates(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Edit the upstream iterable -> the loop must re-run and downstream reflect it.
    nb_runner.set_cell_source(3, "items = [1, 2, 3, 4]")
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "out=[10, 20, 30, 40]" in out, f"iterable edit not reflected: {out!r}"


# ---------------------------------------------------------------------------
# #7 Editing the body function's definition invalidates the cached loop.
# ---------------------------------------------------------------------------

def test_body_function_edit_invalidates(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        SLOW,
        "items = [1, 2, 3]",
        "out = []\n"
        "for e in items:\n"
        "    out.append(slow(e))",
        "print(f'out={out}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out=[10, 20, 30]" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Redefine slow: e * 100 instead of e * 10. The loop's cache key includes the
    # function source hash, so it must re-run.
    nb_runner.set_cell_source(2, SLOW.replace("return e * 10", "return e * 100"))
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "out=[100, 200, 300]" in out, f"body-function edit not reflected: {out!r}"
