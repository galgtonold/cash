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
    actually lost is narrower: ``CallUnit._store`` never consults the
    ``persist`` annotation and always writes the same sparse, fixed metadata
    (``execution_time``/``timestamp``), so it can never set
    ``force_persist`` — cache *policy* (this branch's Task 5 report already
    documented the same gap for TTL and the size/cost model) doesn't
    propagate down to call units. For a call CHEAP enough to also miss the
    generic floor (this file's ``slow`` sleeps 0.03s, deliberately below it),
    ``@cash:persist`` can no longer override that and force it durable, so it
    does NOT survive a restart. The RESULT stays correct either way (a full,
    correct recompute — not corruption or a stale value); these two tests
    were rewritten to pin exactly that ("correct but fully recomputed", not
    "restored") for the cheap/sub-floor case specifically. This is a known,
    accepted consequence of CAS-259 surfaced during its own test triage, not
    something this suite is hiding — see the CAS-259 task report for the
    follow-up question it raises (should ``persist`` — and TTL, and the size
    model — propagate from the statement path down into ``CallUnit``, so it
    can override the floor for cheap calls too?);
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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(nb_runner.client.km._async_restart_kernel(now=True))
    loop.run_until_complete(nb_runner.client.kc._async_wait_for_ready(timeout=30))
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
# #2 CAS-259 consequence: a REAL kernel restart no longer restores this shape
#    from disk. The disk tier is fine; ``persist`` (like TTL and the
#    size/cost model, per this branch's Task 5 report) just doesn't
#    propagate from the statement path into ``CallUnit`` -- see module
#    docstring. What must NOT regress is correctness: the recompute must
#    still land the exact right answer.
# ---------------------------------------------------------------------------

def test_accumulator_loop_recomputes_correctly_after_restart(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook([
        SETUP,
        _slow_def(counter),
        "items = [1, 2, 3, 4, 5]",
        # persist directive on the FOR line so it scopes to the loop unit, not
        # the (too-cheap) seed above it. Kept even though it can no longer
        # make this shape survive a restart, to prove that too (see below).
        "out = []\n"
        "# @cash:persist\n"
        "for e in items:\n"
        "    out.append(slow(e))",
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
    # NOT because interception is RAM-only -- it isn't; CallUnit writes
    # through the same backend.set() the statement path uses, and an
    # EXPENSIVE intercepted call (e.g. 1.2s) survives a restart just fine
    # (TieredBackend promotes to disk once its generic ~1s compute floor is
    # cleared, `@cash:persist` or not). ``slow`` here sleeps only 0.03s --
    # deliberately cheap -- and CallUnit's metadata carries neither
    # `force_persist` (the `persist` annotation never reaches CallUnit._store)
    # nor a cost-model family, so it falls through to that generic floor and
    # does NOT clear it. So all 5 calls genuinely re-run. Asserted explicitly
    # (not just "don't crash") so a future change that silently alters this
    # either direction gets caught rather than passing unnoticed.
    assert warm == 5, f"expected all 5 calls to re-run post-restart, got {warm}"

    nb_runner.run_cell(5)
    assert "out=[10, 20, 30, 40, 50]" in nb_runner.get_output(5), nb_runner.get_output(5)


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
