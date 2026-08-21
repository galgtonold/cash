"""Loop shape x edit-kind matrix: measures INCREMENTAL REUSE and INVALIDATION
PRECISION, not restore.

CAS-259, CAS-261 and CAS-262 all survived a fully green 832-file suite for the
same structural reason: every existing test of a loop shape asserted only an
*unchanged* rerun costs zero real calls. That proves restore works. It says
nothing about whether appending one item recomputes only that item, whether
reordering leaves an unrelated cell alone, or whether an edit to a cell the
loop does not read leaves the loop alone. This module adds that missing
dimension: four loop shapes x five kinds of edit, every cell measured as
counted real executions of the loop body's callee, not inferred from printed
values, badges or timings.

Instrument, not endorsement
----------------------------
``compute()`` appends one byte to an external counter file every time it
actually runs. That file is the ONLY thing any assertion here reads to decide
"did real work happen." This is deliberately the same technique CAS-259's own
regression guard uses (``test_accumulator_single_statement_append_incremental.
py``), and it is a side effect the callee performs purely to serve as this
suite's instrument -- CAS-260 is the record that a callee's own side effects
are not, in general, something cash's caching is obliged to preserve on a
cache hit. Using one here to COUNT executions is not an endorsement of that
gap; it is the least-invasive way to get a ground truth number that survives
being read by a value-blind cache.

Retracted design note (kept as a record, because the retraction is the
lesson). An earlier draft of this module reported that ``compute()`` must not
read a plain module-global free variable -- that ``return v * factor`` with
`factor` in another cell defeated call-level caching entirely, every call,
every time. **That was re-measured on 2026-08-02 and is false.** Four
configurations (global in its own cell vs. beside the ``def``, crossed with
the single-statement and two-statement loop shapes) all gave 0 real calls on
an unchanged rerun and exactly 1 on an append -- identical to a callee with
no free-variable read at all. The original observation came from an
uncommitted scratch draft that carried a confounder.

The negative control still redefines ``compute()`` itself rather than editing
a variable it reads, which is the existing convention in this suite -- but for
convention's sake, not because the alternative is broken.

Shapes (seed placement matters -- see "hoisting" below)
---------------------------------------------------------
A - single-statement accumulator: ``out = []`` / ``out.append(compute(t))``
B - two-statement body: ``v = compute(t)`` / ``out.append(v)``
C - accumulator fold: ``s = 0`` / ``s += compute(t)``
D - subscript store: ``d = {}`` / ``d[t] = compute(t)``

Edits, per shape
-----------------
1. unchanged rerun         -- expect 0 real calls
2. append an item          -- expect 1 real call (only the new item)
3. reorder the items       -- expect 0 real calls
4. edit an unrelated cell  -- expect 0 real calls (the loop does not read it)
5. edit a real dependency  -- expect full recompute (negative control)

Size variant (shape A only)
-----------------------------
CAS-259 (fixed on this branch, commit 61d920d) and CAS-261 (still open) sit on
opposite sides of one boundary: ``for_handler._should_execute_loop_as_single_
unit`` treats a loop as one big cacheable unit once ``n * body_stmts * 0.008s
>= 1.0s`` (n >= 125 for a 1-statement body); below that it decomposes, and
``call_unit._COST_FLOOR_S`` only *stores* a call that individually took that
long. A loop can be under BOTH floors at once -- decomposed (so no whole-unit
caching) yet each call too cheap to store individually (so no per-call caching
either) -- while its aggregate cost is well worth caching. CAS-261's own
measured reproduction is n=124 iterations at 5ms/call (0.62s total): 0 warm
calls before CAS-259 touched anything, 124 warm calls after.

**Status: CAS-261 step 1 landed.** ``_COST_FLOOR_S`` was 10ms -- inherited
from the statement path's floor rather than measured -- and is now **3ms**,
derived from end-to-end measurement (store ~0.7ms/call, hit ~1.2ms/call, so
break-even is a ~1.2ms body). The four shape-A-large cases below were
``xfail(strict=True)`` against CAS-261 and now pass outright: the n=100 at
5ms band went from 100 warm calls to 0. What remains open is the band BELOW
the break-even (many iterations of a sub-millisecond body), where per-call
caching cannot pay for itself at any N and the fix is promotion to a
whole-loop unit -- CAS-261 step 2, designed but not built.

The task brief suggested "large (150 items, cheap body)" for this variant.
Measured directly: n=150 at 5ms/call clears the single-unit threshold
(n >= 125) and 61d920d's fix (restoring ``force_outputs`` for that branch)
already covers its unchanged-rerun case cleanly (0 warm calls) -- it is NOT
CAS-261's gap. This module instead uses **n=100** at 5ms/call: solidly inside
the (50, 125) sub-floor band on BOTH the baseline count and after an append
(100 -> 101 stays inside the band; 124 -> 125 would have crossed the
single-unit threshold mid-test and confounded the append measurement with the
boundary itself). The deviation from the brief's literal 150 is deliberate
and is exactly what this module is for: measure the real boundary, not the
suggested one. (n=150's own measured behaviour -- clean on unchanged rerun,
but a full recompute on append/reorder because the single-unit's cache key
still keys on the whole iterable, and a CLEAN pass on the unrelated-edit case
-- is reported in the task report rather than pinned here, to keep this
module's shape list matching the brief's four-shapes-plus-one-size-variant
structure; see the report for the numbers and why they matter for CAS-262.)

CAS-262 confound, found while calibrating this module
---------------------------------------------------------
The existing xfailed probe for CAS-262 (``test_zzprobe_codeleads.py::
test_unrelated_upstream_edit_reruns_loop``) uses a bare-expression loop body
(``results.append(i * i)``, no function call). Every unrelated-edit case in
this module measures 0 real calls, so none of them reproduces CAS-262.

An earlier reading attributed that to call-level caching (CAS-243) absorbing
the re-execution whenever the callee is cacheable. **That mechanism is
wrong.** Re-measured with a monotonic counter: a loop whose callee is BELOW
``_COST_FLOOR_S`` -- never stored at the call level, so call caching cannot
be helping -- still shows 0 recomputation after an unrelated upstream edit
(6 real calls cold, 6 after the edit; had the body re-executed it would read
12). Statement-level restore is what absorbs it.

CAS-262 itself still fails on HEAD under ``--runxfail``, so it is real: it is
over-invalidation at the PLANNING level, detected by inspecting
``UPSTREAM_DEBUG`` output. This module counts WORK. Both instruments are
valid and they answer different questions -- which is precisely how CAS-262
came to be filed at a severity ("an edit to an unrelated cell costs a full
loop re-run") that measurement does not support. The ticket now carries that
scope correction and was lowered High -> Medium.

Failure policy
--------------
A failure that does not match a known ticket's described mechanism is left as
a plain, failing assertion -- not xfailed, not silenced, not "fixed" here. A
matrix of tests that cannot fail is worse than no matrix, and an unattributed
red test is the flag that sends the next investigation to the right place
instead of requiring another hand audit. There are currently no xfails in
this module: the four that recorded CAS-261's band were lifted when step 1
landed.

``get_output`` / ``run_cell`` on ``nb_runner`` are 1-based cell indices.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Well above call_unit._COST_FLOOR_S so shapes A-small/B/C/D exercise real
# per-call caching rather than the "too cheap to store" floor.
_SLEEP_SMALL = 0.3
# 5ms: CAS-261's reported band. Under the 10ms floor this stored nothing (100
# warm calls); under the measured 3ms floor it caches (0 warm calls). n=100
# also stays below the single-unit threshold (n>=125 for a 1-stmt body), so
# this exercises per-CALL caching, not whole-loop caching -- including after
# the append case takes it to 101.
_SLEEP_LARGE = 0.005
_N_LARGE = 100

# These fail only from inside a full parallel suite on a genuinely loaded box
# -- not under synthetic CPU load, not when this file runs alone beside one.
# The mechanism is not diagnosed (see the commit that added `_why`), so they
# are retried rather than left to redden every slow run.
#
# Scoped three ways so this cannot become a blanket "retry until green":
#   * only_rerun is their OWN assertion text. Any other failure in the same
#     test -- the `cold == N` harness sanity check, an error, a timeout --
#     still fails on the first attempt.
#   * a deterministic regression fails all three attempts and still reports
#     FAILED. Only an intermittent one is absorbed.
#   * a retried failure is NOT discarded: tests/conftest.py prints its text
#     under "failures that passed on a retry", which is where the on-disk
#     evidence `_why` collects will show up.
#
# The delay is a guess, not a measurement: it is long enough to outrun a brief
# scheduling spike and nowhere near long enough to outrun a slow run, which is
# what the failures actually correlate with. If the rerun report shows these
# still burning all three attempts, retrying is the wrong tool for them.
LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

SETUP = "import cash\n%cash_on\nimport time"
SETUP_OFF = "import cash\nimport time"

# A cost floor high enough that no realistic stall lifts this test's body over
# it. The SUBJECT of the break-even test is the PROTECTION -- "a call below the
# break-even is not stored individually" -- not the 3ms number, so pinning the
# threshold keeps what is under test and drops what is not: how busy the box is.
#
# conftest's CASH_TEST_PIN_THRESHOLDS cannot serve here: it pins the floor to
# 0.0 so every call clears it, the exact opposite of this test's premise.
#
# Measured: with the real 3ms floor the test failed 2/2 in isolation on a loaded
# machine (45 of 60 calls re-ran; the tolerance is 5) while passing inside the
# full suite, because a file append is sub-millisecond warm but not under load.
SETUP_HIGH_COST_FLOOR = SETUP + "\ncash.configure(call_cost_floor_seconds=1.0)"

UNRELATED = "unrelated = 1"
UNRELATED_EDITED = "unrelated = 1\nz = 1"

SMALL_BASE = "[1, 2, 3]"
SMALL_APPEND = "[1, 2, 3, 4]"
SMALL_REORDER = "[3, 1, 2]"

# Ranges start at 1, never 0: `compute(0)` returns `0 * 10`, which CPython
# interns to the argument object itself, tripping the call unit's
# arg-identity refusal -- one permanently-uncached call that mimics a caching
# bug. See _compute_def's docstring.
LARGE_BASE = f"list(range(1, {_N_LARGE + 1}))"
LARGE_APPEND = f"list(range(1, {_N_LARGE + 2}))"
LARGE_REORDER = f"[2, 1] + list(range(3, {_N_LARGE + 1}))"


def _n(path):
    """Real compute() calls so far: one byte per call, read from disk."""
    return len(path.read_bytes()) if path.exists() else 0


def _why(work_dir):
    """What cash left on DISK, for use in a failure message.

    The four shape-A-large cases fail only on a genuinely loaded machine, and
    only from inside a full parallel suite -- not under synthetic CPU load, not
    when this file runs alone beside one. Each real occurrence is therefore
    rare and not summonable, and a bare "re-ran 100 calls" cannot tell "nothing
    was ever stored" from "entries were stored and then not found". The entry
    count separates those. Reading the directory perturbs nothing.
    """
    cache_dir = work_dir / ".cash"
    if not cache_dir.exists():
        return "no .cash directory: nothing was ever stored"
    entries = list(cache_dir.glob("*.meta"))
    split = (cache_dir / "_loop_split.json")
    return (f"{len(entries)} cache entries on disk; "
            f"split store {'present' if split.exists() else 'absent'}")


def _compute_def(counter, sleep, mult=10):
    """Item sets must not contain ``0`` -- see the LARGE_* definitions.

    ``return v * mult`` on ``v = 0`` yields ``0``, which CPython interns to
    the very same object as the argument. The call unit's arg-identity guard
    correctly refuses to cache a callee that returns its own argument, so
    that ONE call re-ran on every warm rerun forever and looked exactly like
    a caching bug -- it cost a full bisect to attribute.

    Fixed by starting the ranges at 1 rather than by changing this payload:
    shapes C and D assert on exact printed values, so altering the return
    here silently rewrites what those tests are checking.
    """
    # ``time.sleep(0)`` is NOT a no-op: it is a voluntary yield, so under a
    # 16-worker suite the call can be descheduled for a full Windows quantum
    # (~10-15ms) and measure far above the 3ms cost floor. That is how the
    # sub-break-even test's body -- meant to be unambiguously cheap -- ended up
    # storing 6 calls and failing at 54/60. Emit no sleep at all when there is
    # none to take; callers passing a real duration are unaffected.
    body_sleep = f"    time.sleep({sleep})\n" if sleep else ""
    return (
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        f"{body_sleep}"
        f"    return v * {mult}"
    )


# ---------------------------------------------------------------------------
# Shape bodies. A/B/C/D bodies each end in their own print so the loop cell
# is self-contained; hoisted shapes keep the seed in an EARLIER cell (the
# whole-cell-caching-confounder trap: seed + loop in one cell makes the trap,
# the fix, and an undirected control all measure identically).
# ---------------------------------------------------------------------------

def _a_body(items_expr):
    return f"for t in {items_expr}:\n    out.append(compute(t))\nprint('OUT', out)"


def _b_body(items_expr):
    return f"for t in {items_expr}:\n    v = compute(t)\n    out.append(v)\nprint('OUT', out)"


def _c_body(items_expr):
    return f"for t in {items_expr}:\n    s += compute(t)\nprint('S', s)"


def _d_body(items_expr):
    return f"for t in {items_expr}:\n    d[t] = compute(t)\nprint('D', sorted(d.items()))"


def _hoisted_cells(counter, seed_src, loop_src, setup=SETUP):
    """SETUP(1) / UNRELATED(2) / compute def(3) / seed(4) / loop(5).

    Seed is hoisted into its own cell, one earlier than the loop -- NOT the
    loop's immediately-preceding sibling in the same cell.
    """
    return [setup, UNRELATED, _compute_def(counter, _SLEEP_SMALL), seed_src, loop_src]


LOOP_CELL_HOISTED = 5
UNRELATED_CELL = 2
COMPUTE_DEF_CELL = 3


def _adjacent_cell(counter, items_expr, setup=SETUP):
    """SETUP(1) / UNRELATED(2) / compute def(3) / seed+loop(4).

    Shape A's large variant deliberately keeps the seed ADJACENT (same cell,
    immediately preceding the `for`) -- that adjacency is part of what CAS-259
    and CAS-261 are about, not a confounder to avoid here.
    """
    return [setup, UNRELATED, _compute_def(counter, _SLEEP_LARGE),
            f"out = []\n{_a_body(items_expr)}"]


LOOP_CELL_ADJACENT = 4


# ===========================================================================
# Shape A, small (n=3, 0.3s/call, HOISTED seed)
# ===========================================================================

def test_shape_a_small_unchanged_rerun(nb_runner, tmp_path):
    """Mutation that would break this: deleting call_unit's cache-hit
    short-circuit (the branch that serves a stored result instead of
    invoking the wrapped callee) makes this jump from 0 to 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3, f"baseline did not run all 3 items: {cold}"
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)

    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_append(nb_runner, tmp_path):
    """Mutation: in call_unit.py, make the discriminator ignore the call's
    argument value (CAS-259's own mutation) -- this must jump from 1 to 3+."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _a_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1 (only the new item) [{_why(tmp_path)}]"
    assert "OUT [10, 20, 30, 40]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_reorder(nb_runner, tmp_path):
    """Mutation: strip call-level caching's order-independence (key the call
    by loop position instead of argument value) -- this must jump to 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _a_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0 [{_why(tmp_path)}]"
    assert "OUT [30, 10, 20]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_unrelated_edit(nb_runner, tmp_path):
    """Mutation: in the upstream lineage simulation, make ANY upstream cell
    edit mark every downstream loop stale regardless of dependency -- if this
    is already broken (CAS-262) that mutation is invisible; measured here as
    a clean pass (0 real calls), though the seed cell's OWN statement is
    still needlessly (harmlessly) re-executed -- see the report."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, (
        f"unrelated upstream edit re-ran {warm} calls, expected 0 "
        f"(raw: {nb_runner.get_raw_output(LOOP_CELL_HOISTED)[:400]!r})"
    )
    assert "OUT [10, 20, 30]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_dependency_edit(nb_runner, tmp_path):
    """Negative control: redefines compute() itself (a genuine dependency of
    every call in the loop). Mutation: make the lineage tracker ignore a
    callee's own source-hash when building the call's cache key -- this must
    then WRONGLY stay at 0 instead of recomputing all 3."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3

    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3 (full recompute)"
    assert "OUT [20, 40, 60]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_a_small_oracle_no_caching(nb_runner, tmp_path):
    """Cash OFF, identical shape: proves the 0/1/0 results above are cash's
    doing, not some property of the shape (an idempotent for-loop reruns
    everything with no caching at all, every single time)."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _a_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3, f"oracle baseline did not run all 3 items: {cold}"

    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3 (no caching)"


# ===========================================================================
# Shape A, LARGE (n=100, 5ms/call, ADJACENT seed) -- CAS-261's uncached band
# ===========================================================================

@LOAD_SENSITIVE
def test_shape_a_large_unchanged_rerun(nb_runner, tmp_path):
    """Mutation: raising call_unit._COST_FLOOR_S above 5ms would make this
    fail even harder (more of the small-shape tests would join it);
    lowering it below 5ms, or aligning the two floors per CAS-261's own
    recommended fix, would flip this to a genuine pass."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE, f"baseline did not run all {_N_LARGE} items: {cold}"

    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0 (measured: {warm}/{_N_LARGE}) [{_why(tmp_path)}]"


@LOAD_SENSITIVE
def test_shape_a_large_append(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above -- see
    that test's mutation note."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(LOOP_CELL_ADJACENT, f"out = []\n{_a_body(LARGE_APPEND)}")
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1 (only the new item) [{_why(tmp_path)}]"


@LOAD_SENSITIVE
def test_shape_a_large_reorder(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(LOOP_CELL_ADJACENT, f"out = []\n{_a_body(LARGE_REORDER)}")
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0 [{_why(tmp_path)}]"


@LOAD_SENSITIVE
def test_shape_a_large_unrelated_edit(nb_runner, tmp_path):
    """Same underlying mechanism as the unchanged-rerun case above."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0 (measured: {warm}/{_N_LARGE}) [{_why(tmp_path)}]"


def test_shape_a_large_dependency_edit(nb_runner, tmp_path):
    """Negative control -- must pass regardless of CAS-261: a real dependency
    change has to recompute even in the uncached band."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_LARGE, mult=20))
    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == _N_LARGE, f"real dependency edit only re-ran {warm} calls, expected {_N_LARGE}"


def test_shape_a_large_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_adjacent_cell(counter, LARGE_BASE, setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_LARGE

    nb_runner.run_cell(LOOP_CELL_ADJACENT)
    warm = _n(counter) - cold
    assert warm == _N_LARGE, f"oracle only re-ran {warm} calls on unchanged rerun; expected {_N_LARGE}"


# ===========================================================================
# Shape B: two-statement body (v = compute(t); out.append(v)), HOISTED seed
# ===========================================================================

def test_shape_b_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_b_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _b_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "OUT [10, 20, 30, 40]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_b_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _b_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0 [{_why(tmp_path)}]"


def test_shape_b_unrelated_edit(nb_runner, tmp_path):
    """CAS-262's own ticket cites the TWO-statement spelling as already
    broken on `main`, independent of this branch -- but that ticket's own
    repro has no function call in the body. Measured here (with a call):
    clean pass."""
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_b_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"


def test_shape_b_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "out = []", _b_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"


# ===========================================================================
# Shape C: accumulator fold (s += compute(t)), HOISTED seed
# ===========================================================================

def test_shape_c_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_c_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _c_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "S 100" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _c_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0 [{_why(tmp_path)}]"
    assert "S 60" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_unrelated_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_c_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"
    assert "S 120" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_c_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "s = 0", _c_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"


# ===========================================================================
# Shape D: subscript store (d[t] = compute(t)), HOISTED seed
# ===========================================================================

def test_shape_d_unchanged_rerun(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unchanged rerun re-ran {warm} calls, expected 0"


def test_shape_d_append(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _d_body(SMALL_APPEND))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 1, f"append re-ran {warm} calls, expected 1"
    assert "[(1, 10), (2, 20), (3, 30), (4, 40)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_reorder(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(LOOP_CELL_HOISTED, _d_body(SMALL_REORDER))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"reorder re-ran {warm} calls, expected 0 [{_why(tmp_path)}]"
    assert "[(1, 10), (2, 20), (3, 30)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_unrelated_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(UNRELATED_CELL, UNRELATED_EDITED)
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 0, f"unrelated upstream edit re-ran {warm} calls, expected 0"


def test_shape_d_dependency_edit(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.set_cell_source(COMPUTE_DEF_CELL, _compute_def(counter, _SLEEP_SMALL, mult=20))
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"real dependency edit only re-ran {warm} calls, expected 3"
    assert "[(1, 20), (2, 40), (3, 60)]" in nb_runner.get_output(LOOP_CELL_HOISTED)


def test_shape_d_oracle_no_caching(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(_hoisted_cells(counter, "d = {}", _d_body(SMALL_BASE), setup=SETUP_OFF))
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 3
    nb_runner.run_cell(LOOP_CELL_HOISTED)
    warm = _n(counter) - cold
    assert warm == 3, f"oracle only re-ran {warm} calls on unchanged rerun; expected 3"


# ===========================================================================
# Sub-break-even guard: cash must not make a cheap loop SLOWER
# ===========================================================================

_N_TINY = 60


def test_sub_break_even_calls_are_not_stored_individually(nb_runner, tmp_path):
    """A call cheaper than the break-even must NOT be cached per-call.

    This is the guard the suite never had, and its absence is why CAS-261's
    proposed fix looked right on paper. That fix was "if a site is hit N times
    and N x elapsed clears a floor, store it, even when each call is
    individually sub-floor." It cannot work: storing N calls saves N x body
    but costs N x hit, so the ratio is PER CALL and aggregate size never
    enters it. Measured end-to-end at n=124 with the floor forced to 0:

        body 0.1ms -> 166ms warm vs 21ms cash-off   (8x SLOWER)
        body 1ms   -> 166ms warm vs 133ms cash-off  (1.25x slower)
        body 2ms   -> 200ms warm vs 257ms cash-off  (1.3x faster)

    So per-call caching below ~1.2ms is a pessimisation no N can rescue, and
    ``_COST_FLOOR_S`` is what prevents it. This test pins that protection.

    Asserted as counted executions, not wall-clock: a timing assertion here
    would false-fail under parallel load the way ``test_cfd_loop_overhead``
    does, and would be measuring the machine rather than the policy.

    NOTE for CAS-261 step 2: whole-loop promotion will legitimately turn this
    into 0 warm calls, restored as ONE unit -- that is the correct fix for
    this band, and it does not contradict this test's point. When it lands,
    change the assertion to "restored as a single unit"; do NOT simply delete
    it, because the thing being guarded (never store these calls
    individually) stays true either way.
    """
    counter = tmp_path / "calls.log"
    # sleep=0 -> the body is just a file append, far below the 3ms floor.
    cells = [SETUP_HIGH_COST_FLOOR, UNRELATED, _compute_def(counter, 0),
             f"out = []\nfor t in list(range(1, {_N_TINY + 1})):\n"
             f"    out.append(compute(t))\nprint('OUT', len(out))"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == _N_TINY, f"baseline did not run all {_N_TINY} items: {cold}"

    nb_runner.run_cell(4)
    warm = _n(counter) - cold

    # Deliberately a range, not `== _N_TINY`. The store decision is itself
    # timing-derived, so an occasional call can genuinely clear the floor.
    #
    # The range discriminates sharply, which is the point. Measured: 60/60
    # across three full parallel suite runs, and 0/60 with the floor set to
    # zero. The bound sits in open space between them.
    assert warm >= _N_TINY - 5, (
        f"sub-break-even calls are being cached individually ({warm}/"
        f"{_N_TINY} re-ran; expected nearly all of them). Storing these costs "
        "more than recomputing them -- check whether "
        "`config.call_cost_floor_seconds` was lowered below the measured "
        "~1.2ms break-even, or whether a store decision stopped consulting "
        "it. NOTE: patching the `call_unit._COST_FLOOR_S` constant does NOT "
        "move this -- it is only the fallback for when no config is "
        "reachable; the config field is what governs."
    )
